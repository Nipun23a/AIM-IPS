import asyncio
import json
import os
import time
import logging
import uuid
from typing import Callable, Optional

# IPs allowed to set X-Forwarded-For (your Nginx / load-balancer on VPS).
# Comma-separated in .env: TRUSTED_PROXIES=127.0.0.1,10.0.0.1
_TRUSTED_PROXIES: set = {
    ip.strip()
    for ip in os.getenv("TRUSTED_PROXIES", "127.0.0.1").split(",")
    if ip.strip()
}

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from shared.schemas import RequestContext, LayerScore
from shared.constants import (
    LAYER_0, LAYER_1, ACTION_BLOCK, ACTION_CAPTCHA, ACTION_THROTTLE,
    ACTION_DELAY, ACTION_ALLOW, NORMAL_LABELS,
    KEY_BLACKLIST, KEY_RATE_LIMIT, KEY_NETWORK_THREAT,
    TTL_RATE_LIMIT, MAX_REQUESTS_PER_MINUTE,
)
from firewall.decisions import FirewallDecision
from firewall.regex_filter import RegexFilter
from utils.redis_client import RedisClient
from utils import redis_threat_store
from response.engine import ResponseEngine

logger = logging.getLogger(__name__)

DELAY_SECONDS    = 1.5
THROTTLE_SECONDS = 3.0


class IPSMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, skip_paths: list = None):
        super().__init__(app)
        self.skip_paths      = skip_paths or [
            "/health", "/docs", "/openapi.json", "/redoc", "/status"
        ]
        self.response_engine = ResponseEngine(auto_blacklist=True)
        self._redis          = None
        self._firewall       = None
        # RegexFilter cached once — avoids re-instantiation on every request
        self._regex_filter   = RegexFilter()
    
    @property
    def redis(self):
        if self._redis is None:
            try:
                self._redis = RedisClient.get_redis()
            except Exception:
                pass
        return self._redis
    
    @property
    def firewall(self):
        if self._firewall is None:
            try:
                from firewall.engine import StaticFirewall
                self._firewall = StaticFirewall()
                logger.info("[Middleware] StaticFirewall Loaded")
            except Exception as e:
                logger.error(f"[Middleware] StaticFirewall load failed : {e}")
        return self._firewall
    
    def _get_layer2(self,request:Request):
        return getattr(request.app.state,"layer2",None)
    
    async def dispatch(self, request: Request, call_next: Callable):
        if any(request.url.path.startswith(p) for p in self.skip_paths):
            return await call_next(request)

        ctx = await self._build_context(request)

        # ── Single Redis pipeline: blacklist + rate-limit + network score ──
        # One round-trip replaces three serial GET/INCR calls.
        is_blacklisted, rate_count, net_data = self._redis_bulk_read(ctx.ip)

        # Layer 0 — blacklist / rate-limit / static firewall
        ctx = self._run_layer0(ctx, is_blacklisted, rate_count)
        if ctx.short_circuited:
            asyncio.ensure_future(self._log_async(ctx))
            return self._blocked_response(ctx)

        # Layer 1 — regex filter (cached instance)
        ctx = self._run_layer1(ctx)
        if ctx.short_circuited:
            asyncio.ensure_future(self._log_async(ctx))
            return self._blocked_response(ctx)

        # Inject pre-fetched network score
        if net_data:
            ctx.network_threat = net_data
            ctx.network_score  = float(net_data.get(redis_threat_store.F_SCORE, 0.0))

        # Layer 2 — ML
        layer2 = self._get_layer2(request)
        if layer2:
            ctx = layer2.run(ctx)
        else:
            logger.warning("[Middleware] Layer2 not ready — skipping ML layer")
            ctx.add_score(LayerScore.clean(LAYER_0))

        decision, detail = self.response_engine.decide(ctx)
        asyncio.ensure_future(self._log_async(ctx))   # fire-and-forget
        return await self._apply_action(ctx, decision, request, call_next)
    

    def _redis_bulk_read(self, ip: str):
        """
        Single Redis pipeline replacing 3 serial calls:
          EXISTS blacklist:ip:{ip}
          INCR   ratelimit:ip:{ip}  +  EXPIRE
          GET    threat:ip:{ip}

        Returns (is_blacklisted, rate_count, net_data_dict_or_None).
        Falls back gracefully if Redis is unavailable.
        """
        r = self.redis
        if r is None:
            return False, 0, None
        try:
            pipe = r.raw.pipeline(transaction=False)
            pipe.exists(KEY_BLACKLIST.format(ip=ip))
            pipe.incr(KEY_RATE_LIMIT.format(ip=ip))
            pipe.expire(KEY_RATE_LIMIT.format(ip=ip), TTL_RATE_LIMIT)
            pipe.get(KEY_NETWORK_THREAT.format(ip=ip))
            results = pipe.execute()

            is_bl      = bool(results[0])
            rate_count = int(results[1])
            raw_net    = results[3]
            net_data   = json.loads(raw_net) if raw_net else None
            return is_bl, rate_count, net_data
        except Exception as e:
            logger.warning("[Middleware] Redis pipeline failed: %s", e)
            return False, 0, None

    def _run_layer0(self, ctx: RequestContext,
                    is_blacklisted: bool, rate_count: int) -> RequestContext:
        try:
            if is_blacklisted:
                ctx.add_score(LayerScore.hard_block(
                    layer=LAYER_0, label="blacklisted", reason="IP is blacklisted"
                ))
                ctx.short_circuited = True
                ctx.action          = ACTION_BLOCK
                ctx.block_reason    = "IP blacklisted"
                return ctx

            if rate_count > MAX_REQUESTS_PER_MINUTE:
                ctx.add_score(LayerScore.hard_block(layer=LAYER_0, label="rate_limited"))
                ctx.short_circuited = True
                ctx.action          = ACTION_BLOCK
                ctx.block_reason    = "Rate limit exceeded"
                return ctx

            if self.firewall:
                req_dict = self._ctx_to_dict(ctx)
                decision, reason = self.firewall.inspect(req_dict)

                if decision == FirewallDecision.MITIGATE:
                    ctx.add_score(LayerScore.hard_block(
                        layer=LAYER_0,
                        label=str(reason) if isinstance(reason, str) else "pattern_match",
                        reason=str(reason),
                    ))
                    ctx.short_circuited = True
                    ctx.action          = ACTION_BLOCK
                    ctx.block_reason    = str(reason)

                elif decision == FirewallDecision.FORWARD_TO_ML:
                    suspicion_score = 0.0
                    if isinstance(reason, dict):
                        suspicion_score = min(0.4, reason.get("suspicion_score", 0) * 0.10)
                    ctx.add_score(LayerScore(
                        score=suspicion_score, label="suspicious",
                        confidence=0.5, layer=LAYER_0, triggered=False,
                        metadata={"reason": str(reason)},
                    ))
                else:
                    ctx.add_score(LayerScore.clean(LAYER_0))
            else:
                ctx.add_score(LayerScore.clean(LAYER_0))

        except Exception as e:
            logger.error("[Middleware] Layer0 error: %s", e, exc_info=True)
            ctx.add_score(LayerScore.clean(LAYER_0))

        return ctx

    def _run_layer1(self, ctx: RequestContext) -> RequestContext:
        try:
            req_dict = self._ctx_to_dict(ctx)
            decision, reason = self._regex_filter.inspect(req_dict)

            if decision in (FirewallDecision.MITIGATE, FirewallDecision.BLOCK,
                            "BLOCK", "MITIGATE"):
                ctx.add_score(LayerScore.hard_block(
                    layer=LAYER_1,
                    label=str(reason) if reason else "regex_match",
                    reason=str(reason),
                ))
                ctx.short_circuited = True
                ctx.action          = ACTION_BLOCK
                ctx.block_reason    = str(reason)

            elif decision == FirewallDecision.FORWARD_TO_ML:
                conf  = reason.get("confidence", 0.65) if isinstance(reason, dict) else 0.65
                score = self._regex_filter.confidence_to_score(conf)
                label = reason.get("pattern_group", "regex_suspicious") if isinstance(reason, dict) else "regex_suspicious"
                ctx.add_score(LayerScore(
                    score=score, label=label,
                    confidence=conf, layer=LAYER_1, triggered=False,
                    metadata=reason if isinstance(reason, dict) else {"reason": str(reason)},
                ))
            else:
                ctx.add_score(LayerScore.clean(LAYER_1))

        except Exception as e:
            logger.error("[Middleware] Layer1 error: %s", e, exc_info=True)
            ctx.add_score(LayerScore.clean(LAYER_1))

        return ctx
    

    async def _log_async(self, ctx: RequestContext) -> None:
        """Fire-and-forget: log to Redis without blocking the response."""
        try:
            if self.redis:
                self.redis.log_request_scores(ctx.request_id, ctx.to_log_dict())
        except Exception as e:
            logger.error("[Middleware] Async log error: %s", e)

        # Push to admin events list (24 h TTL, max 10 000 events)
        # Also enqueue to db:queue for the background DB writer (fire-and-forget,
        # zero pipeline coupling — actual DB write happens in a separate async task).
        try:
            if self.redis:
                event = ctx.to_log_dict()
                event["latency_ms"] = round((time.time() - ctx.timestamp) * 1000, 1)
                raw_event = json.dumps(event)
                pipe = self.redis.raw.pipeline(transaction=False)
                pipe.lpush("admin:events", raw_event)
                pipe.ltrim("admin:events", 0, 9999)
                pipe.expire("admin:events", 86400)
                pipe.lpush("db:queue", raw_event)   # picked up by DBWriter
                pipe.ltrim("db:queue", 0, 49999)    # cap queue at 50k events
                pipe.execute()
        except Exception as e:
            logger.error("[Middleware] Admin event push error: %s", e)
    
    async def _apply_action(
        self,
        ctx:       RequestContext,
        decision:  FirewallDecision,
        request:   Request,
        call_next: Callable,
    ) -> Response:

        if decision == FirewallDecision.BLOCK:
            return self._blocked_response(ctx)

        if decision == FirewallDecision.CAPTCHA:
            return self._captcha_response(ctx)

        if decision == FirewallDecision.THROTTLE:
            await asyncio.sleep(THROTTLE_SECONDS)   # non-blocking sleep
            response = await call_next(request)
            return self._add_ips_headers(response, ctx, ACTION_THROTTLE)

        if decision == FirewallDecision.DELAY:
            await asyncio.sleep(DELAY_SECONDS)       # non-blocking sleep
            response = await call_next(request)
            return self._add_ips_headers(response, ctx, ACTION_DELAY)

        # ALLOW
        response = await call_next(request)
        return self._add_ips_headers(response, ctx, ACTION_ALLOW)

    def _blocked_response(self, ctx: RequestContext) -> JSONResponse:
        return JSONResponse(
            status_code=403,
            content={
                "error":      "Forbidden",
                "reason":     ctx.block_reason or "Security policy violation",
                "request_id": ctx.request_id,
            },
            headers={
                "X-IPS-Action": ACTION_BLOCK,
                "X-IPS-Score":  str(round(ctx.final_score, 3)),
            }
        )

    def _captcha_response(self, ctx: RequestContext) -> JSONResponse:
        return JSONResponse(
            status_code=429,
            content={
                "error":      "Challenge Required",
                "message":    "Please complete the security challenge",
                "request_id": ctx.request_id,
            },
            headers={
                "X-IPS-Action": ACTION_CAPTCHA,
                "X-IPS-Score":  str(round(ctx.final_score, 3)),
            }
        )

    def _add_ips_headers(
        self, response: Response, ctx: RequestContext, action: str
    ) -> Response:
        response.headers["X-IPS-Action"]    = action
        response.headers["X-IPS-Score"]     = str(round(ctx.final_score, 3))
        response.headers["X-IPS-RequestID"] = ctx.request_id
        return response
    


    async def _build_context(self, request: Request) -> RequestContext:
        try:
            body_bytes = await request.body()
            body       = body_bytes.decode("utf-8", errors="replace")
        except Exception:
            body = ""
        direct_ip = request.client.host if request.client else "0.0.0.0"
        if direct_ip in _TRUSTED_PROXIES:
            # Request came from a trusted proxy (Nginx/LB) — use the forwarded IP
            ip = (
                request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
                or request.headers.get("X-Real-IP", "")
                or direct_ip
            )
        else:
            # Direct connection — use the real socket IP, ignore forwarded headers
            # (prevents attackers from spoofing X-Forwarded-For to bypass blacklist)
            ip = direct_ip

        return RequestContext(
            ip           = ip,
            method       = request.method,
            path         = request.url.path,
            headers      = dict(request.headers),
            query_params = dict(request.query_params),
            body         = body,
            timestamp    = time.time(),
            request_id   = str(uuid.uuid4()),
        )

    def _ctx_to_dict(self, ctx: RequestContext) -> dict:
        return {
            "ip":           ctx.ip,
            "method":       ctx.method,
            "path":         ctx.path,
            "headers":      ctx.headers,
            "body":         ctx.body,
            "query_params": ctx.query_params,
        }



















        




        
                

import json 
import time
import logging
import redis
from typing import Optional
from shared.constants import (
    KEY_NETWORK_THREAT, KEY_BLACKLIST, KEY_RATE_LIMIT,
    KEY_CAPTCHA_SESSION, KEY_REQUEST_LOG,
    TTL_NETWORK_THREAT, TTL_BLACKLIST_TEMP, TTL_CAPTCHA,
    TTL_RATE_LIMIT, TTL_REQUEST_LOG,
    MAX_REQUESTS_PER_MINUTE
)

logger = logging.getLogger(__name__)

class RedisClient:
    _instance: Optional["RedisClient"] = None

    def __init__(
        self,
        host:     str  = "localhost",
        port:     int  = 6379,
        db:       int  = 0,
        password: Optional[str] = None,
        decode_responses: bool  = True,
    ):
        self._client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=decode_responses,
            socket_connect_timeout=2,
            socket_timeout=2,
            retry_on_timeout=True,
        )
        try:
            self._client.ping()
            logger.info(f"[Redis] Connected → {host}:{port} db={db}")
        except redis.ConnectionError as e:
            logger.error(f"[Redis] Cannot connect: {e}")
            raise

    @classmethod
    def get_instance(cls,host:str = "localhost",port:int = 6379,db:int = 0,password:Optional[str] = None) -> "RedisClient":
        if cls._instance is None:
            cls._instance = cls(host=host,port=port,db=db,password=password)
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        cls._instance = None
    
    @property
    def raw(self) -> redis.Redis:
        return self._client
    
    def set_network_threat_score(
        self,
        ip:          str,
        score:       float,
        net_lgbm:    float = 0.0,
        tcn:         float = 0.0,
        attack_type: str   = "Unknown",
        confidence:  float = 0.0,
        ttl:         int   = TTL_NETWORK_THREAT,
    ) -> bool:

        key = KEY_NETWORK_THREAT.format(ip=ip)

        payload = {
                "score":       round(float(score), 4),
                "net_lgbm":    round(float(net_lgbm), 4),
                "tcn":         round(float(tcn), 4),
                "attack_type": attack_type,
                "confidence":  round(float(confidence), 4),
                "timestamp":   round(time.time(), 3),
        }
        try:
            self._client.setex(key, ttl, json.dumps(payload))
            return True
        except redis.RedisError as e:
                logger.warning(f"[Redis] set_network_threat_score failed {ip}: {e}")
                return False
    
    def get_network_threat_score(self, ip: str) -> Optional[dict]:
        key = KEY_NETWORK_THREAT.format(ip=ip)
        try:
            raw = self._client.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except (redis.RedisError, json.JSONDecodeError) as e:
            logger.warning(f"[Redis] get_network_threat_score failed {ip}: {e}")
            return None
        
    def get_network_score_value(self, ip:str, default:float = 0.0) -> float:
        data = self.get_network_threat_score(ip)
        if data is None:
            return default
        return float(data.get("score", default))
    
    def blacklist_ip(self, ip: str, reason: str = "", permanent: bool = False) -> bool:
        key = KEY_BLACKLIST.format(ip=ip)
        payload = json.dumps({
            "reason": reason,
            "permanent": permanent,
            "timestamp": round(time.time(), 3)
        })
        try:
            if permanent:
                self._client.set(key, payload)
            else:
                self._client.setex(key, TTL_BLACKLIST_TEMP, payload)
            logger.info(f"[Redis] Blacklisted IP {ip} (permanent={permanent}) Reason: {reason}")
            return True
        except redis.RedisError as e:
            logger.warning(f"[Redis] blacklist_ip failed {ip}: {e}")
            return False
    
    def is_blacklisted(self, ip: str) -> bool:
        key = KEY_BLACKLIST.format(ip=ip)
        try:
            return self._client.exists(key) == 1
        except redis.RedisError as e:
            logger.warning(f"[Redis] is_blacklisted failed {ip}: {e}")
            return False
        
    def remove_from_blacklist(self, ip: str) -> bool:
        key = KEY_BLACKLIST.format(ip=ip)
        try:
            self._client.delete(key)
            logger.info(f"[Redis] Removed IP {ip} from blacklist")
            return True
        except redis.RedisError as e:
            logger.warning(f"[Redis] remove_from_blacklist failed {ip}: {e}")
            return False
        
    def get_blacklist_entry(self, ip:str) -> Optional[dict]:
        key = KEY_BLACKLIST.format(ip=ip)
        try:
            raw = self._client.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except (redis.RedisError, json.JSONDecodeError) as e:
            logger.warning(f"[Redis] get_blacklist_entry failed {ip}: {e}")
            return None
        

    def increment_request_count(self, ip: str, window: int = TTL_RATE_LIMIT) -> int:
        key = KEY_RATE_LIMIT.format(ip=ip)
        try:
            pipe = self._client.pipeline()
            pipe.incr(key)
            pipe.expire(key,window)
            result = pipe.execute()
            return int(result[0])
        except redis.RedisError as e:
            logger.warning(f"[Redis] increment_request_count failed {ip}: {e}")
            return 0
    
    def get_request_count(self, ip: str) -> int:
        key = KEY_RATE_LIMIT.format(ip=ip)
        try:
            count = self._client.get(key)
            return int(count) if count else 0
        except redis.RedisError as e:
            return 0
    
    def is_rate_limited(self, ip: str,limit:int = MAX_REQUESTS_PER_MINUTE) -> bool:
        count = self.increment_request_count(ip)
        return count > limit
    
    def set_captcha_session(self,ip:str) -> bool:
        key = KEY_CAPTCHA_SESSION.format(ip = ip)
        try:
            self._client.setex(key, TTL_CAPTCHA, "pending")
            return True
        except redis.RedisError : 
            return False
    
    def is_captcha_pending(self, ip:str) -> bool:
        key = KEY_CAPTCHA_SESSION.format(ip = ip)
        try:
            return self._client.get(key) =="pending"
        except redis.RedisError : 
            return False
    
    def resolve_captcha(self, ip:str) -> bool:
        key = KEY_CAPTCHA_SESSION.format(ip = ip)
        try:
            self._client.setex(key, TTL_CAPTCHA, "solved")
            return True
        except redis.RedisError :
            return False
    
    def is_captcha_solved(self, ip:str) -> bool:
        key = KEY_CAPTCHA_SESSION.format(ip = ip)
        try:
            return self._client.get(key) == "solved"
        except redis.RedisError :
            return False
    
    def log_request_scores(self, request_id:str, log_dict:dict) -> bool:
        key = KEY_REQUEST_LOG.format(request_id = request_id)
        try:
            self._client.setex(key, TTL_REQUEST_LOG, json.dumps(log_dict))
            return True
        except redis.RedisError as e:
            logger.warning(f"[Redis] log_request_scores failed {request_id}: {e}")
            return False
    
    def get_request_log(self,request_id:str,log_dict:dict) -> Optional[dict]:
        key = KEY_REQUEST_LOG.format(request_id = request_id)
        try:
            self._client.setex(key, TTL_REQUEST_LOG, json.dumps(log_dict))
            return True
        except redis.RedisError as e:
            logger.warning(f"[Redis] get_request_log failed {request_id}: {e}")
            return False
        
    def get_request_log(self, request_id:str) -> Optional[dict]:
        key = KEY_REQUEST_LOG.format(request_id = request_id)
        try:
            raw = self._client.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except (redis.RedisError, json.JSONDecodeError) as e:
            logger.warning(f"[Redis] get_request_log failed {request_id}: {e}")
            return None
        

    def ping(self) -> bool:
        try:
            return self._client.ping()
        except redis.RedisError as e:
            logger.warning(f"[Redis] ping failed: {e}")
            return False
    
    def get_stats(self) -> dict:
        try:
            info = self._client.info()
            return {
                "redis_version": info.get("redis_version", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "uptime_in_seconds": info.get("uptime_in_seconds", 0),
            }
        except redis.RedisError as e:
            logger.warning(f"[Redis] get_stats failed: {e}")
            return {}
    
    def get_redis(
            host: str = "localhost",
            port: int = 6379,
            db: int = 0,
            password: Optional[str] = None,
    ) -> "RedisClient":
        return RedisClient.get_instance(host=host,port=port,db=db,password=password)
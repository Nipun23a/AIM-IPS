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
    """
        Singleton Redis client for AIM-IPS.
    """

    _instance: Optional["RedisClient"] = None

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        decode_responses: bool = True
    ):
        self._client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=decode_responses
            socket_connect_timeout=2,
            socket_timeout=2
            retry_on_timeout=True,
        )
        try:
            self._client.ping()
            logger.info(f"[Redis] Connected -> {host}:{port} db = {db}")
        except redis.ConnectionError as e:
            logger.error(f"[Redis] Connection failed: {e}")
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
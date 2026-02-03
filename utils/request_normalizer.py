import time
import time

def normalize_request(raw):
    return {
        "id": raw.get("id"),
        "ip": raw.get("ip"),
        "method": raw.get("method"),
        "path": raw.get("path"),
        "headers": raw.get("headers", {}),
        "body": raw.get("body", ""),
        "timestamp": time.time(),
        "scores": {},
        "flags": {}
    }


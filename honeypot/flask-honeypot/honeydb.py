import os
import requests
from dotenv import load_dotenv

load_dotenv()

HONEYDB_ENABLED = os.getenv("HONEYDB_ENABLED", "false").lower() == "true"
HONEYDB_API_KEY = os.getenv("HONEYDB_API_KEY", "").strip()

def lookup_ip_reputation(ip: str):
    if not HONEYDB_ENABLED or not HONEYDB_API_KEY or not ip:
        return None
    try:
        BASE_URL = "https://api.honeydb.io/v1/ip"  # adjust to real endpoint
        headers = {"Authorization": f"Bearer {HONEYDB_API_KEY}"}
        resp = requests.get(f"{BASE_URL}/{ip}", headers=headers, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return {"ti_source": "honeydb", "ti_found": True, "ti_raw": data}
        return {"ti_source": "honeydb", "ti_found": False, "status_code": resp.status_code}
    except Exception as e:
        return {"ti_source": "honeydb", "ti_error": str(e)}

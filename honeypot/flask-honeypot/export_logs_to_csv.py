import json, csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data_collector"

LOG = DATA_DIR / "honeypot_log.json"
OUT = DATA_DIR / "honeypot_log.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)

def flatten(row):
    ti = row.get("threat_intel") or {}
    return {
        "timestamp": row.get("timestamp"),
        "ip": row.get("ip"),
        "method": row.get("method"),
        "path": row.get("path"),
        "query_params": json.dumps(row.get("query_params", {}), ensure_ascii=False),
        "form_data": json.dumps(row.get("form_data", {}), ensure_ascii=False),
        "raw_body": row.get("raw_body", ""),
        "tags": "|".join(row.get("tags", [])),
        "ti_found": str(ti.get("ti_found", False)),
        "ti_source": ti.get("ti_source", ""),
        "ti_raw": json.dumps(ti.get("ti_raw", {}), ensure_ascii=False)
        if ti and ti.get("ti_found") else ""
    }

def main():
    if not LOG.exists():
        print("No log file found:", LOG)
        return

    with LOG.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        print("No rows to export")
        return

    rows = [flatten(r) for r in data]

    # ✅ Overwrite safely (no unlink)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("Exported ->", OUT)

if __name__ == "__main__":
    main()

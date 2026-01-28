import json, os, csv

LOG = os.path.join("logs", "honeypot_log.json")
OUT_DIR = os.path.join("logs", "export")
OUT = os.path.join(OUT_DIR, "honeypot_log.csv")

os.makedirs(OUT_DIR, exist_ok=True)

def flatten(row):
    ti = row.get("threat_intel") or {}
    return {
        "timestamp": row.get("timestamp"),
        "ip": row.get("ip"),
        "method": row.get("method"),
        "path": row.get("path"),
        "query_params": json.dumps(row.get("query_params", {}), ensure_ascii=False),
        "form_data": json.dumps(row.get("form_data", {}), ensure_ascii=False),
        "raw_body": row.get("raw_body",""),
        "tags": "|".join(row.get("tags", [])),
        "ti_found": str(ti.get("ti_found", False)),
        "ti_source": ti.get("ti_source",""),
        "ti_raw": json.dumps(ti.get("ti_raw", {}), ensure_ascii=False) if ti and ti.get("ti_found") else ""
    }

def main():
    rows = []
    if os.path.exists(LOG):
        with open(LOG, "r", encoding="utf-8") as f:
            data = json.load(f)
            for r in data:
                rows.append(flatten(r))
    else:
        print("No log file found:", LOG)
        return

    if not rows:
        print("No rows to export")
        return

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Exported ->", OUT)

if __name__ == "__main__":
    main()

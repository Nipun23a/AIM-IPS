import json, os, time, argparse, requests

LOG = os.path.join("logs", "honeypot_log.json")

def main():
    ap = argparse.ArgumentParser(description="Replay honeypot traffic to a target")
    ap.add_argument("--target", required=True, help="Base URL, e.g. http://127.0.0.1:8080")
    ap.add_argument("--sleep", type=float, default=0.1, help="Seconds between requests")
    ap.add_argument("--limit", type=int, default=0, help="Max number of events to replay (0 = all)")
    args = ap.parse_args()

    if not os.path.exists(LOG):
        print("No log file:", LOG); return

    with open(LOG, "r", encoding="utf-8") as f:
        data = json.load(f)

    count = 0
    for r in data:
        url = args.target.rstrip("/") + (r.get("path") or "/")
        method = (r.get("method") or "GET").upper()
        params = r.get("query_params") or {}
        headers = r.get("headers") or {}
        body = r.get("raw_body") or None

        try:
            if method == "GET":
                requests.get(url, params=params, headers=headers, timeout=5)
            else:
                # attempt to mimic form vs raw body
                if r.get("form_data"):
                    requests.post(url, data=r.get("form_data"), params=params, headers=headers, timeout=5)
                else:
                    requests.request(method, url, params=params, data=body, headers=headers, timeout=5)
        except Exception as e:
            print("ERR:", e)

        count += 1
        if args.limit and count >= args.limit:
            break
        time.sleep(args.sleep)

    print(f"Replayed {count} requests to {args.target}")

if __name__ == "__main__":
    main()

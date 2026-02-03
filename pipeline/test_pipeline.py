import time
from firewall.decisions import FirewallDecision
from main import pipeline


TEST_REQUESTS = [
    {
        "name": "Normal API",
        "request": {
            "ip": "10.0.0.1",
            "method": "GET",
            "path": "/api/users?id=10",
            "headers": {"User-Agent": "Mozilla/5.0"},
            "body": ""
        }
    },
    {
        "name": "SQL Injection",
        "request": {
            "ip": "10.0.0.2",
            "method": "POST",
            "path": "/api/login",
            "headers": {"User-Agent": "curl"},
            "body": "admin' UNION SELECT password FROM users--"
        }
    },
    {
        "name": "Encoded SQLi (FORWARD_TO_ML)",
        "request": {
            "ip": "10.0.0.3",
            "method": "POST",
            "path": "/api/search",
            "headers": {"User-Agent": "python-requests"},
            "body": "id=1%27%20or%201%3D1"
        }
    },
    {
        "name": "Zero-day-like Payload",
        "request": {
            "ip": "10.0.0.4",
            "method": "POST",
            "path": "/api/data",
            "headers": {"User-Agent": "Mozilla"},
            "body": "A9f@#!!9283///==%00%00"
        }
    }
]


def run_pipeline_tests():
    print("\n" + "=" * 90)
    print("AIM-IPS PIPELINE – END-TO-END FUNCTIONAL TEST")
    print("=" * 90)

    for test in TEST_REQUESTS:
        print(f"\n▶ Test: {test['name']}")
        start = time.perf_counter()

        decision, info = pipeline.process(test["request"])

        total_time = (time.perf_counter() - start) * 1000

        print(f"Decision      : {decision}")
        print(f"Scores / Info : {info}")
        print(f"Latency       : {total_time:.3f} ms")

    print("\n✅ PIPELINE FUNCTIONAL TEST COMPLETED\n")


if __name__ == "__main__":
    run_pipeline_tests()

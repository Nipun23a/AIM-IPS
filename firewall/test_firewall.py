from firewall.engine import StaticFirewall
# Test request
test_request = {
    "ip": "10.0.0.1",
    "method": "POST",
    "path": "/login",
    "headers": {"User-Agent": "sqlmap"},
    "body": "SELECT * FROM users",
    "payload_size": 200
}

fw = StaticFirewall()
decision, reason = fw.inspect(test_request)

print("Decision:", decision)
print("Reason:", reason)

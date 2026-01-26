def apply_mitigation(request, threat_type):
    request["mitigation_applied"] = True
    request["threat_type"] = threat_type

    if threat_type == "SQL Injection":
        request["body"] = "[SANITIZED]"
        request["log_level"] = "HIGH"

    elif threat_type == "XSS":
        request["sanitize_output"] = True

    elif threat_type == "Rate Limit Exceeded":
        request["throttle"] = True

    else:
        request["monitor_only"] = True

    return request

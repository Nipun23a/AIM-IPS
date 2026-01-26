import re

# SQL Injection
SQLI_PATTERNS = [
    r"union\s+select",
    r"or\s+1\s*=\s*1",
    r"drop\s+table",
    r"insert\s+into",
    r"delete\s+from",
    r"update\s+\w+\s+set",
    r"--",
    r";\s*--",
]


# XSS
XSS_PATTERNS = [
    r"<script>",
    r"javascript:",
    r"onerror=",
    r"onload=",
]

# Path traversal
PATH_TRAVERSAL = [
    r"\.\./",
    r"\.\.\\",
]

# Bad user agents
BAD_USER_AGENTS = [
    "sqlmap",
    "nikto",
    "nmap",
    "curl",
    "masscan",
]

# Thresholds
MAX_REQUESTS_PER_MINUTE = 120
MAX_PAYLOAD_SIZE = 10_000  # bytes

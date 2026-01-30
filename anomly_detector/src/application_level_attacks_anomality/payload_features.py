import math
import re
from collections import defaultdict, Counter

SQL_KW = re.compile(r"(select|union|insert|drop|or\s+1=1)", re.I)
XSS_KW = re.compile(r"(<script|onerror=|alert\()", re.I)
PATH_KW = re.compile(r"(\.\./|\.\.\\|/etc/passwd)", re.I)
CMD_KW = re.compile(r"(;|\|\||&&|\b(cat|ls|whoami)\b)", re.I)

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    probs = [c/len(s) for c in counts.values()]
    return -sum(p*math.log(p) for p in probs)

def extract_payload_features(payload: str) -> dict:
    payload = str(payload)

    return {
        "payload_len": len(payload),
        "digit_count": sum(c.isdigit() for c in payload),
        "alpha_count": sum(c.isalpha() for c in payload),
        "special_char_count": sum(not c.isalnum() for c in payload),

        "slash_count": payload.count("/"),
        "dot_count": payload.count("."),
        "percent_count": payload.count("%"),
        "equals_count": payload.count("="),
        "question_count": payload.count("?"),
        "amp_count": payload.count("&"),

        "has_sql_kw": int(bool(SQL_KW.search(payload))),
        "has_xss_kw": int(bool(XSS_KW.search(payload))),
        "has_path_kw": int(bool(PATH_KW.search(payload))),
        "has_cmd_kw": int(bool(CMD_KW.search(payload))),

        "entropy": shannon_entropy(payload),
    }


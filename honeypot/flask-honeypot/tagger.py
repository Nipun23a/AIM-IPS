import re
from typing import List, Tuple

PATTERNS = {
    "SQLi": [
        r"(?i)(\bor\b|\band\b)\s+\d+\s*=\s*\d+",
        r"(?i)(union\s+select|sleep\()",
        r"(?i)(--|#|/\*)\s*\w*",
        r"(?i)('|\")\s*or\s*('|\")?1('|\")?\s*=\s*('|\")?1"
    ],
    "XSS": [
        r"(?i)<script[^>]*>",
        r"(?i)on\w+\s*=",
        r"(?i)javascript:",
        r"(?i)alert\s*\("
    ],
    "PathTraversal": [r"\.\./\.\./", r"(?i)/etc/passwd", r"(?i)system32\\\\cmd\.exe"],
    "RFI_LFI": [r"(?i)(file|php|data):\/\/", r"(?i)(\?|&)file=", r"(?i)(\?|&)(path|page|include|inc|template)="],
    "CmdInjection": [r"(\||;|&&|\bexec\b|\bcurl\b|\bwget\b|\bnslookup\b)", r"(?i)(;|\|\|)\s*(cat|ls|whoami|id)"],
    "SSRF": [r"(?i)(\?|&)(url|target|dest|u)=https?://", r"(?i)169\.254\.169\.254"],
    "AuthBruteforce": [r"(?i)/wp-login\.php|/admin|/login", r"(?i)password=\w{0,}"],
}

def _score(patterns: List[str], text: str) -> float:
    if not text:
        return 0.0
    hits = sum(1 for p in patterns if re.search(p, text))
    return min(1.0,hits / max(1, len(patterns)))

def tag_payload(payload: str) -> List[Tuple[str, float]]:
    tags: List[Tuple[str, float]] = []
    for family, pats in PATTERNS.items():
        score = _score(pats, payload)
        if score >= 0.25:
            tags.append((family, round(score, 2)))
    return sorted(tags, key=lambda x: x[1], reverse=True)

def tag_request(path: str, query: dict, form: dict, headers: dict, body: str) -> List[str]:
    parts = [
        path or "",
        "&".join([f"{k}={v}" for k,v in (query or {}).items()]),
        "&".join([f"{k}={v}" for k,v in (form or {}).items()]),
        " ".join([f"{k}:{v}" for k,v in (headers or {}).items()]),
        body or ""
    ]
    text = "\n".join(parts)
    tagged = tag_payload(text)
    return [f"{fam}:{score}" for fam, score in tagged]
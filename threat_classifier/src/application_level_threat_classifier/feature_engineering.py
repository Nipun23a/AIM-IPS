import pandas as pd
import numpy as np
import re
from urllib.parse import unquote
from pathlib import Path
import joblib

class ThreatFeatureExtractor:
    def __init__(self):
        self.feature_names = None

    def extract_features(self, payload):
        if pd.isna(payload) or payload == '':
            payload = ''
        
        try:
            decoded = unquote(str(payload))
        except Exception:
            decoded = str(payload)
        
        features = {}

        length = len(payload)
        features['length'] = length
        features['decoded_length'] = len(decoded)

        # ============ CHARACTER DISTRIBUTION ============
        num_special = sum(not c.isalnum() and not c.isspace() for c in payload)
        num_digits = sum(c.isdigit() for c in payload)
        num_upper = sum(c.isupper() for c in payload)
        num_lower = sum(c.islower() for c in payload)
        num_spaces = payload.count(' ')

        features['num_special_chars'] = num_special
        features['num_digits'] = num_digits
        features['num_uppercase'] = num_upper
        features['num_lowercase'] = num_lower
        features['num_spaces'] = num_spaces

        if length > 0:
            features['special_char_ratio'] = num_special / length
            features['digit_ratio'] = num_digits / length
            features['upper_ratio'] = num_upper / length
            features['lower_ratio'] = num_lower / length
        else:
            features['special_char_ratio'] = 0
            features['digit_ratio'] = 0
            features['upper_ratio'] = 0
            features['lower_ratio'] = 0

        # ============ ENTROPY ============
        if payload:
            prob = [payload.count(c) / length for c in set(payload)]
            features['entropy'] = -sum(p * np.log2(p) for p in prob if p > 0)
        else:
            features['entropy'] = 0
        
        payload_lower = payload.lower()
        
        # ============ SUPER ENHANCED SQL INJECTION DETECTION ============
        
        # Basic SQL keywords
        sql_dml = ['select', 'insert', 'update', 'delete']
        sql_ddl = ['create', 'drop', 'alter', 'truncate']
        sql_dcl = ['grant', 'revoke']
        sql_other = ['union', 'exec', 'execute', 'declare', 'cast', 'convert']
        
        all_sql_keywords = sql_dml + sql_ddl + sql_dcl + sql_other + [
            'from', 'where', 'order', 'group', 'having', 'limit', 'join', 'table', 'database'
        ]
        
        features['sql_keyword_count'] = sum(1 for kw in all_sql_keywords if kw in payload_lower)
        
        # SQL Comments (CRITICAL SQLi indicator)
        features['has_sql_comment'] = int('--' in payload or '/*' in payload or '#' in payload)
        
        # Core SQL operations
        features['has_union'] = int('union' in payload_lower)
        features['has_select'] = int('select' in payload_lower)
        features['has_insert'] = int('insert' in payload_lower)
        features['has_delete'] = int('delete' in payload_lower)
        features['has_drop'] = int('drop' in payload_lower)
        
        # *** CRITICAL: UNION-based SQLi (this should catch admin' UNION...) ***
        features['is_union_injection'] = int(
            'union' in payload_lower and 
            'select' in payload_lower and
            ("'" in payload or '"' in payload or '--' in payload)
        )
        
        # Quote + SQL keyword (strong indicator)
        features['has_quote_and_sql'] = int(
            ("'" in payload or '"' in payload) and
            features['sql_keyword_count'] >= 1
        )
        
        # Quote + comment (classic bypass)
        features['has_quote_comment'] = int(
            ("'--" in payload or '"--' in payload or 
             "'#" in payload or '"#' in payload or
             "'/*" in payload or '"/*' in payload)
        )
        
        # SQL with NULL keyword (UNION SELECT NULL pattern)
        features['has_select_null'] = int(
            'select' in payload_lower and 
            'null' in payload_lower
        )
        
        # Multiple SQL keywords (strong injection indicator)
        features['has_multiple_sql_keywords'] = int(features['sql_keyword_count'] >= 3)
        
        # Classic SQLi patterns
        features['has_always_true'] = int(
            "'1'='1'" in payload_lower or 
            '"1"="1"' in payload_lower or 
            " 1=1" in payload_lower or
            "'=''" in payload or
            " or 1=1" in payload_lower
        )
        
        # OR/AND clauses with quotes (injection pattern)
        features['has_or_with_quote'] = int(
            (" or " in payload_lower or "' or '" in payload_lower or '" or "' in payload_lower) and
            ("'" in payload or '"' in payload)
        )
        
        features['has_and_with_quote'] = int(
            (" and " in payload_lower or "' and '" in payload_lower or '" and "' in payload_lower) and
            ("'" in payload or '"' in payload)
        )
        
        # Time-based SQLi
        features['has_sleep'] = int(
            'sleep(' in payload_lower or 
            'benchmark(' in payload_lower or 
            'pg_sleep' in payload_lower or 
            'waitfor' in payload_lower
        )
        
        # SQL Functions
        sql_functions = ['concat', 'substring', 'ascii', 'char', 'version', 
                         'database', 'user', 'system_user', 'current_user']
        features['sql_function_count'] = sum(1 for f in sql_functions if f in payload_lower)
        
        # Stacked queries
        features['has_stacked_query'] = int(
            ';' in payload and
            any(kw in payload_lower for kw in ['select', 'insert', 'update', 'delete', 'drop'])
        )
        
        # *** NEW: Detect normal SQL vs SQLi ***
        # Normal SQL: starts with SELECT/INSERT/UPDATE, no quotes mid-statement, no comments
        features['is_normal_sql'] = int(
            bool(re.match(r'^\s*(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER)\s', payload, re.IGNORECASE)) and
            '--' not in payload and
            '/*' not in payload and
            (payload.count("'") <= 2) and  # At most 2 quotes for normal string values
            not bool(re.search(r"'\s*(or|and|union)", payload_lower))
        )
        
        # ============ XSS PATTERNS ============
        xss_tags = ['<script', 'javascript:', 'onerror=', 'onload=', 'onclick=', 
                    'onmouseover=', 'onfocus=', 'alert(', 'prompt(', 'confirm(',
                    'eval(', '<iframe', '<object', '<embed', '<img', '<svg',
                    'onabort=', 'onblur=', 'onchange=', 'ondblclick=']
        
        features['xss_pattern_count'] = sum(1 for pat in xss_tags if pat in payload_lower)
        features['has_html_tags'] = int(bool(re.search(r'<[^>]+>', payload)))
        features['has_javascript'] = int('javascript:' in payload_lower)
        features['has_event_handler'] = int(bool(re.search(r'on\w+\s*=', payload_lower)))
        features['has_script_tag'] = int('<script' in payload_lower)
        features['html_tag_count'] = len(re.findall(r'<[^>]+>', payload))

        # ============ COMMAND INJECTION PATTERNS ============
        cmd_chars = ['|', ';', '&&', '||', '`', '$']
        features['cmd_char_count'] = sum(payload.count(c) for c in cmd_chars)
        features['has_pipe'] = int('|' in payload)
        features['has_semicolon'] = int(';' in payload)
        features['has_backtick'] = int('`' in payload)
        features['has_dollar'] = int('$(' in payload or '${' in payload)
        
        cmd_keywords = ['/bin/', '/etc/passwd', '/etc/shadow', 
                        'cat ', 'ls ', 'wget ', 'curl ', 'nc ', 'netcat',
                        'bash', 'sh ', 'chmod', 'chown', 'rm ', 'kill', 'whoami']
        features['cmd_keyword_count'] = sum(1 for kw in cmd_keywords if kw in payload_lower)
        
        # CMDi with shell commands
        features['is_cmd_injection'] = int(
            (features['has_pipe'] or features['has_semicolon'] or features['has_backtick']) and
            features['cmd_keyword_count'] > 0
        )

        # ============ PATH TRAVERSAL PATTERNS ============
        features['has_dotdot'] = int('..' in payload)
        features['dotdot_count'] = payload.count('..')
        features['has_etc_passwd'] = int('/etc/passwd' in payload_lower or '/etc/shadow' in payload_lower)
        features['has_windows_path'] = int('c:\\' in payload_lower or 'windows\\' in payload_lower)
        
        slash_count = payload.count('/') + payload.count('\\')
        features['slash_count'] = slash_count
        
        # *** CRITICAL: Distinguish normal paths from traversal ***
        
        # Normal REST API path pattern: /api/resource or /path/to/resource?param=value
        features['is_rest_api_path'] = int(
            bool(re.match(r'^/[a-zA-Z0-9_-]+(/[a-zA-Z0-9_-]+)*(\?[a-zA-Z0-9_=&-]+)?$', payload)) and
            features['dotdot_count'] == 0 and
            features['has_etc_passwd'] == 0 and
            '<' not in payload and  # No HTML
            "'" not in payload and  # No SQL quotes
            features['sql_keyword_count'] == 0  # No SQL keywords
        )
        
        # Malicious path traversal: multiple .., or system paths, or mixed with other attacks
        features['is_path_traversal_attack'] = int(
            (features['dotdot_count'] >= 2) or
            features['has_etc_passwd'] == 1 or
            features['has_windows_path'] == 1 or
            (features['dotdot_count'] >= 1 and slash_count >= 3)
        )

        # ============ ENCODING DETECTION ============
        features['has_url_encoding'] = int('%' in payload)
        features['has_html_encoding'] = int('&' in payload and ';' in payload)
        features['encoding_count'] = payload.count('%')
        features['hex_encoding_count'] = len(re.findall(r'\\x[0-9a-fA-F]{2}', payload))
        features['unicode_encoding'] = int('\\u' in payload)

        # ============ SUSPICIOUS CHARACTERS ============
        features['has_quotes'] = int("'" in payload or '"' in payload)
        features['quote_count'] = payload.count("'") + payload.count('"')
        features['has_parentheses'] = int('(' in payload and ')' in payload)
        features['parentheses_count'] = payload.count('(') + payload.count(')')
        features['has_brackets'] = int('[' in payload or '{' in payload)
        features['has_equals'] = int('=' in payload)
        features['equals_count'] = payload.count('=')

        # ============ NULL BYTE INJECTION ============
        features['has_null_byte'] = int('%00' in payload or '\\x00' in payload or '\x00' in payload)

        return features
    
    def extract_batch_features(self, payloads):
        print(f"[INFO] Extracting features from {len(payloads)} payloads...")
        
        features_list = []
        for idx, payload in enumerate(payloads):
            if idx % 10000 == 0 and idx > 0:
                print(f"  Progress: {idx}/{len(payloads)} ({idx/len(payloads)*100:.1f}%)")
            
            features = self.extract_features(payload)
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        self.feature_names = features_df.columns.tolist()
        
        print(f"[INFO] Feature extraction complete. Shape: {features_df.shape}")
        print(f"[INFO] Number of features: {len(self.feature_names)}")
        
        return features_df
    
    def save_feature_config(self, path):
        config = {"feature_names": self.feature_names}
        joblib.dump(config, path)
        print(f"[INFO] Feature configuration saved to {path}")

    def load_feature_config(self, path):
        config = joblib.load(path)
        self.feature_names = config.get("feature_names")
        print(f"[INFO] Feature configuration loaded from {path}. Number of features: {len(self.feature_names)}")


if __name__ == "__main__":
    extractor = ThreatFeatureExtractor()
    
    # Test the problematic cases
    test_cases = [
        ("SQLi UNION (should be sqli)", "admin' UNION SELECT NULL, password FROM users--"),
        ("Normal API (should be norm)", "/api/users/profile?id=456"),
        ("SQLi Classic (should be sqli)", "' OR '1'='1' --"),
        ("Normal SQL (should be norm)", "SELECT * FROM products WHERE id=123"),
        ("CMDi (should be cmdi)", "; cat /etc/passwd"),
        ("Path Traversal (should be path-traversal)", "../../../../etc/passwd"),
    ]

    print("\n" + "="*100)
    print("CRITICAL FEATURE EXTRACTION TEST")
    print("="*100)
    
    for label, payload in test_cases:
        print(f"\n{'='*100}")
        print(f"{label}")
        print(f"Payload: {payload}")
        print("-"*100)
        
        features = extractor.extract_features(payload)
        
        # Show CRITICAL distinguishing features
        critical = [
            'is_union_injection', 'is_normal_sql', 'is_rest_api_path',
            'is_path_traversal_attack', 'is_cmd_injection',
            'has_quote_comment', 'has_select_null', 'sql_keyword_count',
            'dotdot_count', 'slash_count', 'quote_count'
        ]
        
        print("Critical features:")
        for feat in critical:
            if feat in features:
                print(f"  {feat:30s}: {features[feat]}")
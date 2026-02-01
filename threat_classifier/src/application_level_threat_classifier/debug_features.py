# threat_classifier/src/application_level_threat_classifier/debug_features.py

from threat_classifier.src.application_level_threat_classifier.feature_engineering import ThreatFeatureExtractor
import pandas as pd

extractor = ThreatFeatureExtractor()

# The two problematic payloads
problematic_cases = [
    ("SQLi UNION - Misclassified as norm", "admin' UNION SELECT NULL, password FROM users--"),
    ("Normal API - Misclassified as path-traversal", "/api/users/profile?id=456"),
]

# Similar correct cases for comparison
correct_cases = [
    ("SQLi Classic - Correctly classified", "' OR '1'='1' --"),
    ("Normal Query - Correctly classified", "SELECT * FROM products WHERE id=123"),
]

print("\n" + "="*100)
print("FEATURE DEBUGGING - WHY ARE THESE MISCLASSIFIED?")
print("="*100)

for category, payload in problematic_cases + correct_cases:
    print(f"\n{'='*100}")
    print(f"{category}")
    print(f"Payload: {payload}")
    print("="*100)
    
    features = extractor.extract_features(payload)
    
    # Show ALL non-zero features
    non_zero = {k: v for k, v in features.items() if v != 0}
    
    print(f"\nNon-zero features ({len(non_zero)}):")
    for feature, value in sorted(non_zero.items()):
        print(f"  {feature:30s}: {value}")
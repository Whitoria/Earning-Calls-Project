#!/usr/bin/env python3
"""
Quick import test for FinBERT tutorial
Run this to verify all required packages are installed
"""

print("="*60)
print("Testing Tutorial Imports")
print("="*60)

all_passed = True

# Test pandas
try:
    import pandas as pd
    print(f"✅ pandas - version {pd.__version__}")
except ImportError as e:
    print(f"❌ pandas - NOT INSTALLED")
    all_passed = False

# Test scipy
try:
    import scipy
    print(f"✅ scipy - version {scipy.__version__}")
except ImportError:
    print(f"❌ scipy - NOT INSTALLED")
    all_passed = False

# Test seaborn
try:
    import seaborn as sns
    print(f"✅ seaborn - version {sns.__version__}")
except ImportError:
    print(f"❌ seaborn - NOT INSTALLED")
    all_passed = False

# Test matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib
    print(f"✅ matplotlib.pyplot - version {matplotlib.__version__}")
except ImportError:
    print(f"❌ matplotlib - NOT INSTALLED")
    all_passed = False

# Test sklearn
try:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
    import sklearn
    print(f"✅ sklearn.metrics - version {sklearn.__version__}")
except ImportError:
    print(f"❌ scikit-learn - NOT INSTALLED")
    all_passed = False

# Test torch
try:
    import torch
    print(f"✅ torch - version {torch.__version__}")
    if torch.cuda.is_available():
        print(f"   GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print(f"   Running on CPU")
except ImportError:
    print(f"❌ torch - NOT INSTALLED")
    all_passed = False

# Test transformers
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import transformers
    print(f"✅ transformers - version {transformers.__version__}")
except ImportError:
    print(f"❌ transformers - NOT INSTALLED")
    all_passed = False

print("\n" + "="*60)

if all_passed:
    print("🎉 All imports successful! You're ready to follow the tutorial.")
else:
    print("⚠️  Some imports failed. Install missing packages with:")
    print("\npip install pandas scipy seaborn matplotlib scikit-learn torch transformers")

print("="*60)
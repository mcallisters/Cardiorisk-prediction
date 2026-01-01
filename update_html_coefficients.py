"""
===============================================================================
UPDATE HTML PREDICTOR WITH ACTUAL MODEL COEFFICIENTS
Automatically updates heart_disease_predictor.html with trained model values
===============================================================================
"""

import json
import re

print("="*80)
print("UPDATING HTML PREDICTOR WITH MODEL COEFFICIENTS")
print("="*80)

# ===============================================================================
# STEP 1: LOAD MODEL CONFIGURATION
# ===============================================================================

print("\nStep 1: Loading model configuration...")

try:
    with open('model_config.json', 'r') as f:
        config = json.load(f)
    print("✓ Loaded model_config.json")
except FileNotFoundError:
    print("❌ ERROR: model_config.json not found!")
    print("   Please run complete_pipeline.py first to train the model.")
    exit(1)

coefficients = config['coefficients']
intercept = config['intercept']

print(f"\nModel Coefficients:")
for feature, coef in coefficients.items():
    print(f"  {feature:30s}: {coef:+.6f}")
print(f"  {'Intercept':30s}: {intercept:+.6f}")

# ===============================================================================
# STEP 2: READ HTML FILE
# ===============================================================================

print("\nStep 2: Reading HTML file...")

try:
    with open('heart_disease_predictor.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    print("✓ Loaded heart_disease_predictor.html")
except FileNotFoundError:
    print("❌ ERROR: heart_disease_predictor.html not found!")
    exit(1)

# ===============================================================================
# STEP 3: UPDATE COEFFICIENTS
# ===============================================================================

print("\nStep 3: Updating coefficients in HTML...")

# Update each coefficient
for feature, coef in coefficients.items():
    # Pattern to match: 'feature_name': 0.1234,  // any comment
    pattern = rf"('{re.escape(feature)}':\s*)-?\d+\.?\d*"
    replacement = rf"\g<1>{coef:.6f}"
    html_content = re.sub(pattern, replacement, html_content)
    print(f"  ✓ Updated {feature}")

# Update intercept
intercept_pattern = r"(const intercept =\s*)-?\d+\.?\d*"
intercept_replacement = rf"\g<1>{intercept:.6f}"
html_content = re.sub(intercept_pattern, intercept_replacement, html_content)
print(f"  ✓ Updated intercept")

# ===============================================================================
# STEP 4: SAVE UPDATED HTML
# ===============================================================================

print("\nStep 4: Saving updated HTML...")

with open('heart_disease_predictor.html', 'w', encoding='utf-8') as f:
    f.write(html_content)
print("✓ Saved updated heart_disease_predictor.html")

# ===============================================================================
# STEP 5: CREATE BACKUP
# ===============================================================================

print("\nStep 5: Creating backup...")

with open('heart_disease_predictor_backup.html', 'w', encoding='utf-8') as f:
    f.write(html_content)
print("✓ Created backup: heart_disease_predictor_backup.html")

# ===============================================================================
# VERIFICATION
# ===============================================================================

print("\n" + "="*80)
print("UPDATE COMPLETE!")
print("="*80)

print("\nVerification:")
print("-" * 60)

# Verify the coefficients are in the file
all_found = True
for feature in coefficients.keys():
    if feature in html_content:
        print(f"✓ {feature} found in HTML")
    else:
        print(f"✗ {feature} NOT found in HTML")
        all_found = False

if all_found:
    print("\n✓ All coefficients successfully updated!")
    print("\nYou can now open heart_disease_predictor.html in your browser.")
    print("The predictions will use your actual trained model coefficients.")
else:
    print("\n⚠️  Some coefficients may not have been updated correctly.")
    print("   Please check the HTML file manually.")

print("\nNext Steps:")
print("  1. Open heart_disease_predictor.html in your web browser")
print("  2. Test with example patient data")
print("  3. Compare results with Python predictions for accuracy")

print("\n" + "="*80)

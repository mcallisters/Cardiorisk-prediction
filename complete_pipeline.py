"""
===============================================================================
COMPLETE END-TO-END PIPELINE: HEART DISEASE PREDICTION
From cleaned CSV → Trained 7-feature model → Ready for predictions
===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score, 
                            precision_score, recall_score, confusion_matrix,
                            roc_curve, classification_report)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")

print("="*80)
print("HEART DISEASE PREDICTION - COMPLETE PIPELINE")
print("From Cleaned Data to Trained 7-Feature Model")
print("="*80)

# ===============================================================================
# STEP 1: LOAD THE CLEANED DATA
# ===============================================================================

print("\n" + "="*80)
print("STEP 1: LOADING CLEANED DATA")
print("="*80)

# Load the cleaned dataset
df = pd.read_csv('cleveland_heart_cleaned.csv')

print(f"\n✓ Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nFirst few rows:")
print(df.head())

print(f"\nColumn names:")
print(df.columns.tolist())

print(f"\nData types:")
print(df.dtypes)

print(f"\nTarget variable distribution:")
print(df['target'].value_counts())

# ===============================================================================
# STEP 2: PREPROCESS THE DATA
# ===============================================================================

print("\n" + "="*80)
print("STEP 2: PREPROCESSING DATA")
print("="*80)

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

print(f"\n✓ Features shape: {X.shape}")
print(f"✓ Target shape: {y.shape}")

# Identify categorical columns that need one-hot encoding
categorical_columns = X.select_dtypes(include=['object', 'bool']).columns.tolist()
print(f"\nCategorical columns to encode: {categorical_columns}")

# One-hot encode categorical variables
X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=False)

# Convert any remaining boolean columns to int
for col in X_encoded.columns:
    if X_encoded[col].dtype == 'bool':
        X_encoded[col] = X_encoded[col].astype(int)

print(f"\n✓ After encoding: {X_encoded.shape[0]} rows × {X_encoded.shape[1]} columns")
print(f"\nEncoded column names:")
for i, col in enumerate(X_encoded.columns, 1):
    print(f"  {i:3d}. {col}")

# ===============================================================================
# STEP 3: IDENTIFY THE 7 OPTIMAL FEATURES
# ===============================================================================

print("\n" + "="*80)
print("STEP 3: IDENTIFYING 7 OPTIMAL FEATURES IN YOUR DATA")
print("="*80)

# The 7 optimal features we want
OPTIMAL_FEATURES_TEMPLATE = [
    'cp_atypical angina',
    'sex_Male',
    'fbs_missing',
    'slope_flat',
    'cp_non-anginal',
    'exang_True',
    'oldpeak'
]

# Check which of these exist in the encoded data
available_features = []
missing_features = []

for feature in OPTIMAL_FEATURES_TEMPLATE:
    if feature in X_encoded.columns:
        available_features.append(feature)
        print(f"✓ Found: {feature}")
    else:
        missing_features.append(feature)
        print(f"✗ Missing: {feature}")

if missing_features:
    print(f"\n⚠️  Warning: {len(missing_features)} features not found in data:")
    for feat in missing_features:
        print(f"  - {feat}")
    
    # Try to find similar column names
    print(f"\n💡 Looking for similar column names...")
    for missing_feat in missing_features:
        # Extract the key part (e.g., 'cp', 'sex', 'fbs', etc.)
        key_parts = missing_feat.split('_')
        print(f"\n  For '{missing_feat}', found these similar columns:")
        for col in X_encoded.columns:
            if any(part in col.lower() for part in key_parts):
                print(f"    - {col}")

# Use only available features for now
OPTIMAL_FEATURES = available_features

print(f"\n✓ Using {len(OPTIMAL_FEATURES)} available features for the model")

# ===============================================================================
# STEP 4: TRAIN-TEST SPLIT
# ===============================================================================

print("\n" + "="*80)
print("STEP 4: TRAIN-TEST SPLIT")
print("="*80)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"\n✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")
print(f"\nTarget distribution in training set:")
print(y_train.value_counts())

# ===============================================================================
# STEP 5: PREPARE 7-FEATURE SUBSETS (UNSCALED)
# ===============================================================================

print("\n" + "="*80)
print("STEP 5: PREPARING 7-FEATURE SUBSETS")
print("="*80)

# Extract the 7 optimal features (unscaled for the scaler to fit on)
X_train_7features = X_train[OPTIMAL_FEATURES].copy()
X_test_7features = X_test[OPTIMAL_FEATURES].copy()

print(f"\n✓ Training subset: {X_train_7features.shape}")
print(f"✓ Test subset: {X_test_7features.shape}")

print(f"\nFeature statistics (training set):")
print(X_train_7features.describe())

# ===============================================================================
# STEP 6: SCALE THE FEATURES
# ===============================================================================

print("\n" + "="*80)
print("STEP 6: FEATURE SCALING")
print("="*80)

# Create and fit scaler on the 7 features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_7features)
X_test_scaled = scaler.transform(X_test_7features)

# Convert back to DataFrames for easier handling
X_train_scaled = pd.DataFrame(X_train_scaled, columns=OPTIMAL_FEATURES, index=X_train_7features.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=OPTIMAL_FEATURES, index=X_test_7features.index)

print(f"\n✓ Scaled training data: {X_train_scaled.shape}")
print(f"✓ Scaled test data: {X_test_scaled.shape}")

print(f"\nScaled feature statistics (training set):")
print(X_train_scaled.describe())

# ===============================================================================
# STEP 7: TRAIN THE MODEL
# ===============================================================================

print("\n" + "="*80)
print("STEP 7: TRAINING 7-FEATURE LOGISTIC REGRESSION MODEL")
print("="*80)

# Train the optimized model
model = LogisticRegression(
    max_iter=2000,
    random_state=42,
    class_weight='balanced'
)

print("\nTraining model...")
model.fit(X_train_scaled, y_train)
print("✓ Model trained successfully")

# Get predictions
y_pred_train = model.predict(X_train_scaled)
y_prob_train = model.predict_proba(X_train_scaled)[:, 1]
y_pred_test = model.predict(X_test_scaled)
y_prob_test = model.predict_proba(X_test_scaled)[:, 1]

# ===============================================================================
# STEP 8: EVALUATE PERFORMANCE
# ===============================================================================

print("\n" + "="*80)
print("STEP 8: MODEL EVALUATION")
print("="*80)

# Calculate metrics
train_metrics = {
    'ROC-AUC': roc_auc_score(y_train, y_prob_train),
    'Accuracy': accuracy_score(y_train, y_pred_train),
    'Precision': precision_score(y_train, y_pred_train, zero_division=0),
    'Recall': recall_score(y_train, y_pred_train, zero_division=0),
    'F1 Score': f1_score(y_train, y_pred_train, zero_division=0)
}

test_metrics = {
    'ROC-AUC': roc_auc_score(y_test, y_prob_test),
    'Accuracy': accuracy_score(y_test, y_pred_test),
    'Precision': precision_score(y_test, y_pred_test, zero_division=0),
    'Recall': recall_score(y_test, y_pred_test, zero_division=0),
    'F1 Score': f1_score(y_test, y_pred_test, zero_division=0)
}

print("\nPerformance Metrics:")
print("-" * 80)
print(f"{'Metric':<15} {'Training':<15} {'Test':<15} {'Difference':<15}")
print("-" * 80)
for metric in train_metrics.keys():
    diff = train_metrics[metric] - test_metrics[metric]
    print(f"{metric:<15} {train_metrics[metric]:<15.4f} {test_metrics[metric]:<15.4f} {diff:<15.4f}")

# Cross-validation
print("\nCross-Validation:")
print("-" * 80)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
print(f"5-Fold CV ROC-AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Classification report
print("\nClassification Report (Test Set):")
print("-" * 80)
print(classification_report(y_test, y_pred_test, 
                          target_names=['No Disease', 'Has Disease'],
                          digits=4))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
print("\nConfusion Matrix:")
print("-" * 80)
print(f"                 Predicted Negative    Predicted Positive")
print(f"Actual Negative        {cm[0,0]:5d}               {cm[0,1]:5d}")
print(f"Actual Positive        {cm[1,0]:5d}               {cm[1,1]:5d}")

# ===============================================================================
# STEP 9: FEATURE IMPORTANCE
# ===============================================================================

print("\n" + "="*80)
print("STEP 9: FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Get feature coefficients
feature_coefficients = pd.DataFrame({
    'Feature': OPTIMAL_FEATURES,
    'Coefficient': model.coef_[0],
    'Abs_Coefficient': np.abs(model.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print("\nFeature Coefficients (sorted by magnitude):")
print("-" * 80)
print(f"{'Feature':<30} {'Coefficient':>12} {'Impact':>20}")
print("-" * 80)
for _, row in feature_coefficients.iterrows():
    direction = "↑ Increases Risk" if row['Coefficient'] > 0 else "↓ Decreases Risk"
    print(f"{row['Feature']:<30} {row['Coefficient']:>12.4f} {direction:>20}")

print(f"\nModel Intercept: {model.intercept_[0]:.4f}")

# ===============================================================================
# STEP 10: VISUALIZATIONS
# ===============================================================================

print("\n" + "="*80)
print("STEP 10: GENERATING VISUALIZATIONS")
print("="*80)

# Figure 1: Model Performance
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('7-Feature Model Performance', fontsize=16, fontweight='bold')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob_test)
axes[0, 0].plot(fpr, tpr, linewidth=3, label=f'Model (AUC = {test_metrics["ROC-AUC"]:.4f})', color='#3498db')
axes[0, 0].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random', alpha=0.5)
axes[0, 0].fill_between(fpr, tpr, alpha=0.3, color='#3498db')
axes[0, 0].set_xlabel('False Positive Rate', fontweight='bold')
axes[0, 0].set_ylabel('True Positive Rate', fontweight='bold')
axes[0, 0].set_title('ROC Curve', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=axes[0, 1],
            xticklabels=['No Disease', 'Has Disease'],
            yticklabels=['No Disease', 'Has Disease'])
axes[0, 1].set_xlabel('Predicted', fontweight='bold')
axes[0, 1].set_ylabel('Actual', fontweight='bold')
axes[0, 1].set_title('Confusion Matrix', fontweight='bold')

# Feature Importance
colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in feature_coefficients['Coefficient']]
axes[1, 0].barh(feature_coefficients['Feature'], feature_coefficients['Coefficient'], 
                color=colors, alpha=0.8)
axes[1, 0].axvline(x=0, color='black', linewidth=1.5)
axes[1, 0].set_xlabel('Coefficient Value', fontweight='bold')
axes[1, 0].set_title('Feature Importance', fontweight='bold')
axes[1, 0].grid(axis='x', alpha=0.3)

# Probability Distribution
axes[1, 1].hist(y_prob_test[y_test == 0], bins=20, alpha=0.6, 
                label='No Disease', color='#2ecc71', edgecolor='black')
axes[1, 1].hist(y_prob_test[y_test == 1], bins=20, alpha=0.6, 
                label='Has Disease', color='#e74c3c', edgecolor='black')
axes[1, 1].axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
axes[1, 1].set_xlabel('Predicted Probability', fontweight='bold')
axes[1, 1].set_ylabel('Count', fontweight='bold')
axes[1, 1].set_title('Probability Distribution', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('model_performance_visualization.png', dpi=300, bbox_inches='tight')
print("✓ Saved: model_performance_visualization.png")
plt.show()

# ===============================================================================
# STEP 11: SAVE MODEL AND ARTIFACTS
# ===============================================================================

print("\n" + "="*80)
print("STEP 11: SAVING MODEL AND ARTIFACTS")
print("="*80)

# Save the trained model
joblib.dump(model, 'heart_disease_model_7features.pkl')
print("✓ Saved: heart_disease_model_7features.pkl")

# Save the scaler
joblib.dump(scaler, 'scaler_7features.pkl')
print("✓ Saved: scaler_7features.pkl")

# Save feature names
with open('optimal_7_features.txt', 'w') as f:
    f.write("OPTIMAL 7 FEATURES FOR HEART DISEASE PREDICTION\n")
    f.write("="*60 + "\n\n")
    for i, feat in enumerate(OPTIMAL_FEATURES, 1):
        f.write(f"{i}. {feat}\n")
print("✓ Saved: optimal_7_features.txt")

# Save model configuration
model_config = {
    'features': OPTIMAL_FEATURES,
    'model_path': 'heart_disease_model_7features.pkl',
    'scaler_path': 'scaler_7features.pkl',
    'performance': test_metrics,
    'coefficients': dict(zip(OPTIMAL_FEATURES, model.coef_[0].tolist())),
    'intercept': float(model.intercept_[0]),
    'training_samples': len(y_train),
    'test_samples': len(y_test)
}

with open('model_config.json', 'w') as f:
    json.dump(model_config, f, indent=4)
print("✓ Saved: model_config.json")

# Save comprehensive summary
with open('model_summary.txt', 'w') as f:
    f.write("HEART DISEASE PREDICTION MODEL - 7-FEATURE OPTIMIZED\n")
    f.write("="*80 + "\n\n")
    
    f.write("MODEL INFORMATION\n")
    f.write("-"*80 + "\n")
    f.write(f"Model Type: Logistic Regression\n")
    f.write(f"Number of Features: {len(OPTIMAL_FEATURES)}\n")
    f.write(f"Training Samples: {len(y_train)}\n")
    f.write(f"Test Samples: {len(y_test)}\n\n")
    
    f.write("OPTIMAL FEATURES\n")
    f.write("-"*80 + "\n")
    for i, feat in enumerate(OPTIMAL_FEATURES, 1):
        f.write(f"{i}. {feat}\n")
    f.write("\n")
    
    f.write("TEST SET PERFORMANCE\n")
    f.write("-"*80 + "\n")
    for metric, value in test_metrics.items():
        f.write(f"{metric:<15}: {value:.4f}\n")
    f.write("\n")
    
    f.write("CROSS-VALIDATION\n")
    f.write("-"*80 + "\n")
    f.write(f"5-Fold CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n\n")
    
    f.write("FEATURE COEFFICIENTS\n")
    f.write("-"*80 + "\n")
    for _, row in feature_coefficients.iterrows():
        direction = "↑" if row['Coefficient'] > 0 else "↓"
        f.write(f"{row['Feature']:<30} {row['Coefficient']:>10.4f} {direction}\n")
    f.write(f"\nIntercept: {model.intercept_[0]:.4f}\n")

print("✓ Saved: model_summary.txt")

# ===============================================================================
# STEP 12: TEST THE PREDICTION FUNCTION
# ===============================================================================

print("\n" + "="*80)
print("STEP 12: TESTING PREDICTION FUNCTION")
print("="*80)

def predict_heart_disease(patient_data):
    """
    Make a prediction for a new patient
    
    Parameters:
    -----------
    patient_data : dict
        Dictionary with the 7 feature values (UNSCALED)
    
    Returns:
    --------
    dict : Prediction results
    """
    # Create DataFrame
    input_df = pd.DataFrame([patient_data])
    
    # Ensure correct feature order
    input_df = input_df[OPTIMAL_FEATURES]
    
    # Scale
    input_scaled = scaler.transform(input_df)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0, 1]
    
    # Risk category
    if probability < 0.3:
        risk_category, risk_color = "Low Risk", "🟢"
    elif probability < 0.6:
        risk_category, risk_color = "Moderate Risk", "🟡"
    else:
        risk_category, risk_color = "High Risk", "🔴"
    
    return {
        'prediction': 'Heart Disease' if prediction == 1 else 'No Heart Disease',
        'probability': float(probability),
        'risk_category': risk_category,
        'risk_color': risk_color,
        'confidence': float(max(probability, 1 - probability))
    }

# Test with a sample from the test set
print("\nTesting with a sample patient from test set:")
print("-" * 60)

sample_idx = 0
example_patient = X_test_7features.iloc[sample_idx].to_dict()

print("\nPatient Data:")
for feature, value in example_patient.items():
    print(f"  {feature:25s}: {value}")

result = predict_heart_disease(example_patient)
actual = "Heart Disease" if y_test.iloc[sample_idx] == 1 else "No Heart Disease"

print(f"\nPrediction:")
print(f"  {result['risk_color']} {result['prediction']}")
print(f"  Probability: {result['probability']:.1%}")
print(f"  Risk Category: {result['risk_category']}")
print(f"  Actual: {actual}")
print(f"  {'✓ CORRECT' if result['prediction'] == actual else '✗ INCORRECT'}")

# ===============================================================================
# FINAL SUMMARY
# ===============================================================================

print("\n" + "="*80)
print("PIPELINE COMPLETE!")
print("="*80)

print("\n📊 Model Performance Summary:")
print(f"  • Test ROC-AUC: {test_metrics['ROC-AUC']:.4f}")
print(f"  • Test Accuracy: {test_metrics['Accuracy']:.4f}")
print(f"  • Test F1 Score: {test_metrics['F1 Score']:.4f}")
print(f"  • CV ROC-AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

print("\n📁 Files Created:")
print("  ✓ heart_disease_model_7features.pkl")
print("  ✓ scaler_7features.pkl")
print("  ✓ optimal_7_features.txt")
print("  ✓ model_config.json")
print("  ✓ model_summary.txt")
print("  ✓ model_performance_visualization.png")

print("\n🎯 Features Used:")
for i, feat in enumerate(OPTIMAL_FEATURES, 1):
    print(f"  {i}. {feat}")

print("\n🚀 Next Steps:")
print("  1. Review model_summary.txt for detailed performance")
print("  2. Check model_performance_visualization.png for visualizations")
print("  3. Use heart_disease_prediction_platform_v2.py to make predictions")
print("  4. Open heart_disease_predictor.html for web interface")

print("\n" + "="*80)

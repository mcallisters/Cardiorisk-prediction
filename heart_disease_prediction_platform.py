"""
===============================================================================
HEART DISEASE PREDICTION PLATFORM - USING SAVED 7-FEATURE MODEL
Loads pre-trained model and scaler for predictions
===============================================================================
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

print("="*80)
print("HEART DISEASE PREDICTION PLATFORM")
print("7-Feature Optimized Model")
print("="*80)

# ===============================================================================
# LOAD MODEL AND ARTIFACTS
# ===============================================================================

print("\n" + "="*80)
print("LOADING MODEL AND ARTIFACTS")
print("="*80)

try:
    # Load the trained model
    model = joblib.load('heart_disease_model_7features.pkl')
    print("✓ Loaded: heart_disease_model_7features.pkl")
    
    # Load the scaler
    scaler = joblib.load('scaler_7features.pkl')
    print("✓ Loaded: scaler_7features.pkl")
    
    # Load model configuration
    with open('model_config.json', 'r') as f:
        config = json.load(f)
    print("✓ Loaded: model_config.json")
    
    OPTIMAL_FEATURES = config['features']
    
    print(f"\nModel Information:")
    print(f"  • Features: {len(OPTIMAL_FEATURES)}")
    print(f"  • ROC-AUC: {config['performance']['ROC-AUC']:.4f}")
    print(f"  • Accuracy: {config['performance']['Accuracy']:.4f}")
    
except FileNotFoundError as e:
    print(f"\n❌ ERROR: Model files not found!")
    print(f"   {e}")
    print("\nPlease run 'train_7feature_model.py' first to train and save the model.")
    print("="*80)
    exit(1)

# ===============================================================================
# FEATURE INFORMATION
# ===============================================================================

print("\n" + "="*80)
print("MODEL FEATURES")
print("="*80)

FEATURE_DESCRIPTIONS = {
    'cp_atypical angina': {
        'type': 'Binary (0/1)',
        'description': 'Atypical angina chest pain',
        'impact': 'Increases risk',
        'coefficient': config['coefficients']['cp_atypical angina']
    },
    'sex_Male': {
        'type': 'Binary (0/1)', 
        'description': 'Male biological sex',
        'impact': 'Increases risk',
        'coefficient': config['coefficients']['sex_Male']
    },
    'fbs_missing': {
        'type': 'Binary (0/1)',
        'description': 'Fasting blood sugar data missing',
        'impact': 'Decreases risk',
        'coefficient': config['coefficients']['fbs_missing']
    },
    'slope_flat': {
        'type': 'Binary (0/1)',
        'description': 'Flat ST segment slope on ECG',
        'impact': 'Increases risk',
        'coefficient': config['coefficients']['slope_flat']
    },
    'cp_non-anginal': {
        'type': 'Binary (0/1)',
        'description': 'Non-anginal chest pain',
        'impact': 'Decreases risk',
        'coefficient': config['coefficients']['cp_non-anginal']
    },
    'exang_True': {
        'type': 'Binary (0/1)',
        'description': 'Exercise-induced angina',
        'impact': 'Increases risk',
        'coefficient': config['coefficients']['exang_True']
    },
    'oldpeak': {
        'type': 'Continuous',
        'description': 'ST depression value (typically 0-6)',
        'impact': 'Increases risk',
        'coefficient': config['coefficients']['oldpeak']
    }
}

print("\nFeatures used by the model:")
for i, feature in enumerate(OPTIMAL_FEATURES, 1):
    info = FEATURE_DESCRIPTIONS[feature]
    arrow = "↑" if "Increase" in info['impact'] else "↓"
    print(f"\n{i}. {feature}")
    print(f"   Type: {info['type']}")
    print(f"   Description: {info['description']}")
    print(f"   Impact: {arrow} {info['impact']}")
    print(f"   Coefficient: {info['coefficient']:.4f}")

# ===============================================================================
# PREDICTION FUNCTION
# ===============================================================================

def predict_heart_disease(patient_data, show_details=True):
    """
    Make a heart disease prediction for a new patient
    
    Parameters:
    -----------
    patient_data : dict
        Dictionary containing the 7 feature values (UNSCALED original values)
        Required keys:
            - cp_atypical angina (0 or 1)
            - sex_Male (0 or 1)
            - fbs_missing (0 or 1)
            - slope_flat (0 or 1)
            - cp_non-anginal (0 or 1)
            - exang_True (0 or 1)
            - oldpeak (continuous, typically 0-6)
    
    show_details : bool, default=True
        Whether to print detailed prediction information
        
    Returns:
    --------
    dict : Prediction results including:
        - prediction: 'Heart Disease' or 'No Heart Disease'
        - probability: Probability of heart disease (0-1)
        - risk_category: 'Low Risk', 'Moderate Risk', or 'High Risk'
        - risk_color: Emoji indicator
        - confidence: Model confidence (0-1)
        - feature_contributions: Dict of feature contributions to prediction
    """
    # Validate input
    missing_features = set(OPTIMAL_FEATURES) - set(patient_data.keys())
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Create DataFrame from input
    input_df = pd.DataFrame([patient_data])
    
    # Select features in correct order
    input_df = input_df[OPTIMAL_FEATURES]
    
    # Scale the input using the saved scaler
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0, 1]
    
    # Determine risk category
    if probability < 0.3:
        risk_category = "Low Risk"
        risk_color = "🟢"
    elif probability < 0.6:
        risk_category = "Moderate Risk"
        risk_color = "🟡"
    else:
        risk_category = "High Risk"
        risk_color = "🔴"
    
    # Calculate feature contributions (coefficient * scaled_value)
    feature_contributions = {}
    for i, feature in enumerate(OPTIMAL_FEATURES):
        feature_contributions[feature] = config['coefficients'][feature] * input_scaled[0, i]
    
    result = {
        'prediction': 'Heart Disease' if prediction == 1 else 'No Heart Disease',
        'probability': float(probability),
        'risk_category': risk_category,
        'risk_color': risk_color,
        'confidence': float(max(probability, 1 - probability)),
        'feature_contributions': feature_contributions
    }
    
    if show_details:
        print("\n" + "="*80)
        print("PREDICTION RESULTS")
        print("="*80)
        print(f"\n{result['risk_color']} Prediction: {result['prediction']}")
        print(f"   Risk Category: {result['risk_category']}")
        print(f"   Probability: {result['probability']:.1%}")
        print(f"   Confidence: {result['confidence']:.1%}")
        
        print("\nFeature Contributions to Prediction:")
        print("-" * 80)
        sorted_contributions = sorted(feature_contributions.items(), 
                                     key=lambda x: abs(x[1]), reverse=True)
        for feature, contribution in sorted_contributions:
            direction = "→ Increases" if contribution > 0 else "→ Decreases"
            print(f"  {feature:30s} {direction:15s} risk  (contrib: {contribution:+.4f})")
    
    return result


def predict_batch(patients_df, show_summary=True):
    """
    Make predictions for multiple patients
    
    Parameters:
    -----------
    patients_df : pandas.DataFrame
        DataFrame with columns matching the 7 required features
    
    show_summary : bool, default=True
        Whether to print summary statistics
        
    Returns:
    --------
    pandas.DataFrame : Results with predictions and probabilities
    """
    results = []
    
    for idx, row in patients_df.iterrows():
        patient_data = row[OPTIMAL_FEATURES].to_dict()
        result = predict_heart_disease(patient_data, show_details=False)
        
        results.append({
            'patient_id': idx,
            'prediction': result['prediction'],
            'probability': result['probability'],
            'risk_category': result['risk_category'],
            'confidence': result['confidence']
        })
    
    results_df = pd.DataFrame(results)
    
    if show_summary:
        print("\n" + "="*80)
        print("BATCH PREDICTION SUMMARY")
        print("="*80)
        print(f"\nTotal Patients: {len(results_df)}")
        print(f"\nRisk Distribution:")
        print(results_df['risk_category'].value_counts())
        print(f"\nPrediction Distribution:")
        print(results_df['prediction'].value_counts())
        print(f"\nAverage Risk Probability: {results_df['probability'].mean():.1%}")
        print(f"High Risk Patients (>60%): {(results_df['probability'] > 0.6).sum()}")
        print(f"Low Risk Patients (<30%): {(results_df['probability'] < 0.3).sum()}")
    
    return results_df


def interactive_prediction():
    """
    Interactive command-line tool for making predictions
    """
    print("\n" + "="*80)
    print("INTERACTIVE HEART DISEASE PREDICTION TOOL")
    print("="*80)
    print("\nEnter patient information (or 'q' to quit, 'h' for help):\n")
    
    while True:
        try:
            patient_data = {}
            
            print("\nEnter feature values:")
            print("-" * 60)
            
            for feature in OPTIMAL_FEATURES:
                info = FEATURE_DESCRIPTIONS[feature]
                prompt = f"{feature}\n  ({info['type']} - {info['description']}): "
                
                value = input(prompt)
                
                if value.lower() == 'q':
                    print("\nExiting...")
                    return
                elif value.lower() == 'h':
                    print_help()
                    continue
                
                patient_data[feature] = float(value)
            
            # Make prediction
            result = predict_heart_disease(patient_data, show_details=True)
            
            # Ask if user wants to continue
            cont = input("\nMake another prediction? (y/n): ")
            if cont.lower() != 'y':
                break
                
        except ValueError as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or enter 'q' to quit.\n")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            return


def print_help():
    """Print help information"""
    print("\n" + "="*80)
    print("HELP - FEATURE DESCRIPTIONS")
    print("="*80)
    
    for feature in OPTIMAL_FEATURES:
        info = FEATURE_DESCRIPTIONS[feature]
        print(f"\n{feature}:")
        print(f"  Type: {info['type']}")
        print(f"  Description: {info['description']}")
        print(f"  Impact: {info['impact']}")
        
        if info['type'] == 'Binary (0/1)':
            print(f"  Valid values: 0 (No/False) or 1 (Yes/True)")
        else:
            print(f"  Valid values: Decimal number (typically 0-6)")
    
    print("\n" + "="*80)


# ===============================================================================
# EXAMPLE USAGE
# ===============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("EXAMPLE PREDICTIONS")
    print("="*80)
    
    # Example 1: High-Risk Patient
    print("\n" + "-"*80)
    print("EXAMPLE 1: HIGH-RISK PATIENT")
    print("-"*80)
    
    high_risk_patient = {
        'cp_atypical angina': 1,
        'sex_Male': 1,
        'fbs_missing': 0,
        'slope_flat': 1,
        'cp_non-anginal': 0,
        'exang_True': 1,
        'oldpeak': 2.5
    }
    
    print("\nPatient Profile:")
    for feature, value in high_risk_patient.items():
        print(f"  {feature}: {value}")
    
    result1 = predict_heart_disease(high_risk_patient)
    
    # Example 2: Low-Risk Patient
    print("\n" + "-"*80)
    print("EXAMPLE 2: LOW-RISK PATIENT")
    print("-"*80)
    
    low_risk_patient = {
        'cp_atypical angina': 0,
        'sex_Male': 0,
        'fbs_missing': 0,
        'slope_flat': 0,
        'cp_non-anginal': 1,
        'exang_True': 0,
        'oldpeak': 0.2
    }
    
    print("\nPatient Profile:")
    for feature, value in low_risk_patient.items():
        print(f"  {feature}: {value}")
    
    result2 = predict_heart_disease(low_risk_patient)
    
    # Example 3: Batch Prediction
    print("\n" + "-"*80)
    print("EXAMPLE 3: BATCH PREDICTION")
    print("-"*80)
    
    patients_df = pd.DataFrame([
        high_risk_patient,
        low_risk_patient,
        {
            'cp_atypical angina': 1,
            'sex_Male': 1,
            'fbs_missing': 1,
            'slope_flat': 0,
            'cp_non-anginal': 0,
            'exang_True': 0,
            'oldpeak': 1.0
        }
    ])
    
    batch_results = predict_batch(patients_df)
    print("\nDetailed Results:")
    print(batch_results.to_string(index=False))
    
    # Show usage instructions
    print("\n" + "="*80)
    print("HOW TO USE THIS PLATFORM")
    print("="*80)
    print("""
1. SINGLE PREDICTION (Programmatic):
   
   result = predict_heart_disease({
       'cp_atypical angina': 1,
       'sex_Male': 1,
       'fbs_missing': 0,
       'slope_flat': 1,
       'cp_non-anginal': 0,
       'exang_True': 1,
       'oldpeak': 2.5
   })
   
   print(result['prediction'])
   print(result['probability'])

2. BATCH PREDICTION:
   
   results_df = predict_batch(patients_df)
   results_df.to_csv('predictions.csv')

3. INTERACTIVE MODE:
   
   interactive_prediction()

4. WEB INTERFACE:
   
   Open 'heart_disease_predictor.html' in your browser
   (Make sure to update it with the correct model file paths)
    """)
    
    print("="*80)
    print("READY FOR PREDICTIONS!")
    print("="*80)
    print("\nTo start interactive mode, uncomment the line below and run again:")
    print("# interactive_prediction()")
    print("="*80)

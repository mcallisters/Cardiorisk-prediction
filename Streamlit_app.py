"""
===============================================================================
HEART DISEASE PREDICTION - STREAMLIT APP
7-Feature Optimized Logistic Regression Model
===============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

# ===============================================================================
# PAGE CONFIGURATION
# ===============================================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================================================================
# CUSTOM CSS
# ===============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .risk-low {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
    }
    .risk-moderate {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
    }
    .risk-high {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ===============================================================================
# INITIALIZE SESSION STATE
# ===============================================================================

# Initialize session state for form fields
if 'chest_pain_type' not in st.session_state:
    st.session_state.chest_pain_type = "Typical Angina"
if 'sex' not in st.session_state:
    st.session_state.sex = "Female"
if 'fbs_status' not in st.session_state:
    st.session_state.fbs_status = "Available"
if 'slope_type' not in st.session_state:
    st.session_state.slope_type = "Upsloping"
if 'exang' not in st.session_state:
    st.session_state.exang = "No"
if 'oldpeak' not in st.session_state:
    st.session_state.oldpeak = 1.0

# ===============================================================================
# LOAD MODEL AND CONFIGURATION
# ===============================================================================

@st.cache_resource
def load_model_artifacts():
    """Load model, scaler, and configuration"""
    try:
        model = joblib.load('heart_disease_model_7features.pkl')
        scaler = joblib.load('scaler_7features.pkl')
        
        with open('model_config.json', 'r') as f:
            config = json.load(f)
        
        return model, scaler, config
    except FileNotFoundError as e:
        st.error(f"⚠️ Model files not found: {e}")
        st.info("Please ensure model files are in the repository.")
        st.stop()

# Load artifacts
model, scaler, config = load_model_artifacts()

OPTIMAL_FEATURES = config['features']
COEFFICIENTS = config['coefficients']
INTERCEPT = config['intercept']

# ===============================================================================
# HELPER FUNCTIONS
# ===============================================================================

def predict_heart_disease(patient_data):
    """Make prediction for a patient"""
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
        risk_category = "Low Risk"
        risk_color = "🟢"
        risk_class = "risk-low"
    elif probability < 0.6:
        risk_category = "Moderate Risk"
        risk_color = "🟡"
        risk_class = "risk-moderate"
    else:
        risk_category = "High Risk"
        risk_color = "🔴"
        risk_class = "risk-high"
    
    # Feature contributions
    feature_contributions = {}
    for i, feature in enumerate(OPTIMAL_FEATURES):
        feature_contributions[feature] = COEFFICIENTS[feature] * input_scaled[0, i]
    
    return {
        'prediction': 'Heart Disease' if prediction == 1 else 'No Heart Disease',
        'probability': float(probability),
        'risk_category': risk_category,
        'risk_color': risk_color,
        'risk_class': risk_class,
        'confidence': float(max(probability, 1 - probability)),
        'feature_contributions': feature_contributions
    }

def load_high_risk_example():
    """Load high risk example into session state"""
    st.session_state.chest_pain_type = "Atypical Angina"
    st.session_state.sex = "Male"
    st.session_state.fbs_status = "Available"
    st.session_state.slope_type = "Flat"
    st.session_state.exang = "Yes"
    st.session_state.oldpeak = 2.5

def load_low_risk_example():
    """Load low risk example into session state"""
    st.session_state.chest_pain_type = "Non-anginal Pain"
    st.session_state.sex = "Female"
    st.session_state.fbs_status = "Available"
    st.session_state.slope_type = "Upsloping"
    st.session_state.exang = "No"
    st.session_state.oldpeak = 0.2

# ===============================================================================
# MAIN APP
# ===============================================================================

# Header
st.markdown('<div class="main-header">❤️ Heart Disease Prediction Platform</div>', 
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">7-Feature Optimized Logistic Regression Model</div>', 
            unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/heart-with-pulse.png", width=100)
    
    st.markdown("## 📊 Model Information")
    st.info(f"""
    **Model Type:** Logistic Regression  
    **Input Fields:** 6  
    **Model Features:** 7 (Chest Pain creates 2 binary features with a single input)  
    **ROC-AUC:** 0.9012  
    **Accuracy:** 0.8098  
    **F1 Score:** 0.8241
    """)
    
    st.markdown("## 📖 About")
    st.markdown("""
    This model predicts heart disease risk using 7 key clinical features 
    identified through sequential feature selection analysis.
    
    The model achieves high accuracy while maintaining simplicity and interpretability.
    """)
    
    st.markdown("## ⚠️ Disclaimer")
    st.warning("""
    **For educational purposes only.**  
    This tool is NOT a substitute for professional medical advice.  
    Always consult healthcare professionals.
    """)

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction", "📊 Model Insights", "📈 Batch Prediction", "ℹ️ User Guide"])

# ===============================================================================
# TAB 1: PREDICTION
# ===============================================================================

with tab1:
    st.markdown("### Enter Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Clinical Features")
        
        # Chest Pain Type
        chest_pain_type = st.selectbox(
            "Chest Pain Type",
            ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
            key="chest_pain_type",
            help="Type of chest pain experienced by the patient"
        )
        
        # Sex
        sex = st.selectbox(
            "Sex",
            ["Female", "Male"],
            key="sex",
            help="Biological sex of the patient"
        )
        
        # FBS Missing
        fbs_status = st.selectbox(
            "Fasting Blood Sugar Data",
            ["Available", "Missing"],
            key="fbs_status",
            help="Whether fasting blood sugar measurement is available"
        )
        
    with col2:
        st.markdown("#### ECG & Exercise Features")
        
        # ST Slope
        slope_type = st.selectbox(
            "ST Slope (ECG)",
            ["Upsloping", "Flat", "Downsloping"],
            key="slope_type",
            help="Slope of the ST segment on ECG during exercise"
        )
        
        # Exercise Induced Angina
        exang = st.selectbox(
            "Exercise Induced Angina",
            ["No", "Yes"],
            key="exang",
            help="Whether exercise triggers angina symptoms"
        )
        
        # Oldpeak
        oldpeak = st.number_input(
            "ST Depression (oldpeak)",
            min_value=0.0,
            max_value=10.0,
            value=st.session_state.oldpeak,
            step=0.1,
            key="oldpeak",
            help="ST depression induced by exercise relative to rest (typically 0-6)"
        )
    
    # Example buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("📋 Load High Risk Example", use_container_width=True):
            load_high_risk_example()
            st.rerun()
    
    with col2:
        if st.button("📋 Load Low Risk Example", use_container_width=True):
            load_low_risk_example()
            st.rerun()
    
    # Predict button
    st.markdown("---")
    if st.button("🔍 Predict Heart Disease Risk", type="primary", use_container_width=True):
        # Convert inputs to binary features
        cp_atypical = 1 if chest_pain_type == "Atypical Angina" else 0
        cp_non_anginal = 1 if chest_pain_type == "Non-anginal Pain" else 0
        sex_male = 1 if sex == "Male" else 0
        fbs_missing = 1 if fbs_status == "Missing" else 0
        slope_flat = 1 if slope_type == "Flat" else 0
        exang_true = 1 if exang == "Yes" else 0
        
        # Prepare patient data
        patient_data = {
            'cp_atypical angina': cp_atypical,
            'sex_Male': sex_male,
            'fbs_missing': fbs_missing,
            'slope_flat': slope_flat,
            'cp_non-anginal': cp_non_anginal,
            'exang_True': exang_true,
            'oldpeak': oldpeak
        }
        
        # Make prediction
        result = predict_heart_disease(patient_data)
        
        # Display results
        st.markdown("---")
        st.markdown("## Prediction Results")
        
        # Risk indicator
        risk_html = f"""
        <div class="{result['risk_class']}">
            <h2>{result['risk_color']} {result['risk_category']}</h2>
            <h3>{result['prediction']}</h3>
        </div>
        """
        st.markdown(risk_html, unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Risk Probability", f"{result['probability']*100:.1f}%")
        
        with col2:
            st.metric("Confidence", f"{result['confidence']*100:.1f}%")
        
        with col3:
            classification = "Positive" if result['probability'] >= 0.5 else "Negative"
            st.metric("Classification", classification)
        
        # Probability gauge
        st.markdown("### Risk Probability Gauge")
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = result['probability'] * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Heart Disease Risk (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 60], 'color': "yellow"},
                    {'range': [60, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature contributions
        st.markdown("### Feature Contributions")
        st.info("ℹ️ Note: 7 features are considered because 'Chest Pain Type' generates 2 separate binary features (atypical angina and non-anginal pain).")
        
        contributions_df = pd.DataFrame([
            {'Feature': k, 'Contribution': v}
            for k, v in result['feature_contributions'].items()
        ]).sort_values('Contribution', key=abs, ascending=False)
        
        fig = px.bar(
            contributions_df,
            x='Contribution',
            y='Feature',
            orientation='h',
            color='Contribution',
            color_continuous_scale=['green', 'white', 'red'],
            color_continuous_midpoint=0,
            title="How Each Feature Affects the Prediction"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendation
        st.markdown("### 💡 Recommendation")
        if result['probability'] >= 0.6:
            st.error("⚠️ **High risk detected.** Immediate medical consultation is strongly recommended.")
        elif result['probability'] >= 0.3:
            st.warning("⚠️ **Moderate risk detected.** Consider consulting with a healthcare provider.")
        else:
            st.success("✅ **Low risk detected.** Maintain a healthy lifestyle and regular check-ups.")

# ===============================================================================
# TAB 2: MODEL INSIGHTS
# ===============================================================================

with tab2:
    st.markdown("### Model Feature Importance")
    
    # Feature importance chart
    importance_df = pd.DataFrame([
        {'Feature': k, 'Coefficient': v}
        for k, v in COEFFICIENTS.items()
    ]).sort_values('Coefficient', key=abs, ascending=False)
    
    fig = px.bar(
        importance_df,
        x='Coefficient',
        y='Feature',
        orientation='h',
        color='Coefficient',
        color_continuous_scale=['green', 'white', 'red'],
        color_continuous_midpoint=0,
        title="Feature Coefficients (Impact on Heart Disease Risk)"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Feature Descriptions")
    
    feature_info = {
        'cp_atypical angina': {
            'description': 'Atypical angina chest pain',
            'impact': '↑ Increases risk',
            'clinical': 'Chest pain with some but not all characteristics of typical angina'
        },
        'sex_Male': {
            'description': 'Male biological sex',
            'impact': '↑ Increases risk',
            'clinical': 'Males have statistically higher heart disease risk'
        },
        'fbs_missing': {
            'description': 'Fasting blood sugar data missing',
            'impact': '↓ Decreases risk',
            'clinical': 'Missing FBS data (statistical artifact in this dataset)'
        },
        'slope_flat': {
            'description': 'Flat ST segment slope on ECG',
            'impact': '↑ Increases risk',
            'clinical': 'Indicates potential cardiac ischemia'
        },
        'cp_non-anginal': {
            'description': 'Non-anginal chest pain',
            'impact': '↓ Decreases risk',
            'clinical': 'Chest pain not related to cardiac origin'
        },
        'exang_True': {
            'description': 'Exercise-induced angina',
            'impact': '↑ Increases risk',
            'clinical': 'Classic symptom of coronary artery disease'
        },
        'oldpeak': {
            'description': 'ST depression value (0-6 typical)',
            'impact': '↑ Increases risk',
            'clinical': 'Higher values indicate more severe cardiac ischemia'
        }
    }
    
    for feature, info in feature_info.items():
        with st.expander(f"**{feature}** ({info['impact']})"):
            st.write(f"**Description:** {info['description']}")
            st.write(f"**Clinical Significance:** {info['clinical']}")
            st.write(f"**Coefficient:** {COEFFICIENTS[feature]:.4f}")
    
    st.markdown("### Model Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("ROC-AUC", f"{config['performance']['ROC-AUC']:.4f}")
        st.metric("Accuracy", f"{config['performance']['Accuracy']:.4f}")
    
    with col2:
        st.metric("Precision", f"{config['performance']['Precision']:.4f}")
        st.metric("Recall", f"{config['performance']['Recall']:.4f}")
    
    with col3:
        st.metric("F1 Score", f"{config['performance']['F1 Score']:.4f}")
        st.metric("Features Used", len(OPTIMAL_FEATURES))

# ===============================================================================
# TAB 3: BATCH PREDICTION
# ===============================================================================

with tab3:
    st.markdown("### Batch Prediction")
    st.info("Upload a CSV file with patient data to get predictions for multiple patients at once.")
    
    # Show expected format
    with st.expander("📋 View Expected CSV Format"):
        example_df = pd.DataFrame([{
            'cp_atypical angina': 1,
            'sex_Male': 1,
            'fbs_missing': 0,
            'slope_flat': 1,
            'cp_non-anginal': 0,
            'exang_True': 1,
            'oldpeak': 2.5
        }])
        st.dataframe(example_df)
        
        st.download_button(
            label="📥 Download Template CSV",
            data=example_df.to_csv(index=False),
            file_name="heart_disease_template.csv",
            mime="text/csv"
        )
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Loaded {len(df)} patients")
            
            # Validate columns
            missing_cols = set(OPTIMAL_FEATURES) - set(df.columns)
            if missing_cols:
                st.error(f"❌ Missing columns: {missing_cols}")
            else:
                # Make predictions
                results = []
                for idx, row in df.iterrows():
                    patient_data = row[OPTIMAL_FEATURES].to_dict()
                    result = predict_heart_disease(patient_data)
                    results.append({
                        'Patient_ID': idx + 1,
                        'Prediction': result['prediction'],
                        'Risk_Category': result['risk_category'],
                        'Probability': f"{result['probability']*100:.1f}%",
                        'Confidence': f"{result['confidence']*100:.1f}%"
                    })
                
                results_df = pd.DataFrame(results)
                
                # Display results
                st.markdown("### Prediction Results")
                st.dataframe(results_df, use_container_width=True)
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    high_risk = (results_df['Risk_Category'] == 'High Risk').sum()
                    st.metric("High Risk Patients", high_risk)
                
                with col2:
                    moderate_risk = (results_df['Risk_Category'] == 'Moderate Risk').sum()
                    st.metric("Moderate Risk Patients", moderate_risk)
                
                with col3:
                    low_risk = (results_df['Risk_Category'] == 'Low Risk').sum()
                    st.metric("Low Risk Patients", low_risk)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name="heart_disease_predictions.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

# ===============================================================================
# TAB 4: USER GUIDE
# ===============================================================================

with tab4:
    st.markdown("### 📖 How to Use This App")
    
    st.markdown("""
    #### 1. Single Patient Prediction
    
    - Go to the **Prediction** tab
    - Enter patient information in the form
    - Click **Load Example** buttons to see sample data
    - Click **Predict Heart Disease Risk**
    - View the results, including:
        - Risk category (Low/Moderate/High)
        - Probability percentage
        - Feature contributions
        - Recommendations
    
    #### 2. Batch Prediction
    
    - Go to the **Batch Prediction** tab
    - Download the CSV template
    - Fill in patient data
    - Upload the CSV file
    - Download the prediction results
    
    #### 3. Understanding the Results
    
    **Risk Categories:**
    - 🟢 **Low Risk** (<30%): Low probability of heart disease
    - 🟡 **Moderate Risk** (30-60%): Moderate probability, consider consultation
    - 🔴 **High Risk** (>60%): High probability, medical consultation recommended
    
    **Feature Contributions:**
    - Shows how each feature affects the prediction
    - Positive values increase risk
    - Negative values decrease risk
    - Larger absolute values = stronger influence
    
    #### 4. Model Features
    
    The model uses 7 features derived from 6 input fields:
    
    1. **Chest Pain Type** → Creates 2 binary features:
       - Atypical angina (yes/no)
       - Non-anginal pain (yes/no)
    2. **Sex**: Male (yes/no)
    3. **Fasting Blood Sugar**: Data missing (yes/no)
    4. **ST Slope**: Flat slope (yes/no)
    5. **Exercise Induced Angina**: Yes/no
    6. **ST Depression (oldpeak)**: Continuous value (0-6 typical)
    
    **Why 7 features from 6 inputs?**  
    Chest Pain Type has 4 options (Typical Angina, Atypical Angina, Non-anginal, Asymptomatic). 
    The model uses 2 of these as separate predictive features, which is why you see 7 total features 
    in the Feature Contributions chart even though you only fill in 6 input fields.
    
    #### 5. Important Notes
    
    ⚠️ **Medical Disclaimer:**
    - This tool is for **educational purposes only**
    - NOT a substitute for professional medical advice
    - Always consult qualified healthcare professionals
    - Do not make medical decisions based solely on this tool
    
    #### 6. Technical Details
    
    - **Model**: Logistic Regression
    - **Features**: 7 optimized features
    - **Training**: Sequential feature removal strategy
    - **Performance**: High accuracy with minimal complexity
    
    """)
    
    st.markdown("### 📞 Support")
    st.info("""
    For questions or issues:
    - Check the Model Insights tab for feature details
    - Review the expected CSV format in Batch Prediction
    - Ensure all required features are provided
    """)

# ===============================================================================
# FOOTER
# ===============================================================================

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Heart Disease Prediction Platform | 7-Feature Optimized Model | "
    "For Educational Purposes Only"
    "</div>",
    unsafe_allow_html=True
)
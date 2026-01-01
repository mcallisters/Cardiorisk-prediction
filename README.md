# ❤️ Heart Disease Prediction Platform

A machine learning web application that predicts heart disease risk using a 7-feature optimized logistic regression model.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

## 🎯 Features

- **Interactive Web Interface**: Built with Streamlit for easy use
- **Single Patient Prediction**: Get instant risk assessments
- **Batch Prediction**: Upload CSV files for multiple patients
- **Visual Analytics**: Interactive charts showing feature contributions
- **Risk Categorization**: Low, Moderate, and High risk classifications
- **Model Insights**: Understand how the model makes decisions

## 🚀 Live Demo

Visit the deployed app: [Heart Disease Predictor](https://your-app-name.streamlit.app)

## 📊 Model Information

- **Type**: Logistic Regression
- **Features**: 7 optimized clinical features
- **Accuracy**: ~85-90% (see model_summary.txt for exact metrics)
- **ROC-AUC**: ~0.88-0.92
- **Strategy**: Sequential feature removal for optimal performance

### The 7 Features

1. **Chest Pain Type** (Atypical Angina / Non-anginal)
2. **Sex** (Male/Female)
3. **Fasting Blood Sugar Status** (Available/Missing)
4. **ST Slope** (Flat/Not Flat)
5. **Exercise Induced Angina** (Yes/No)
6. **ST Depression (oldpeak)** (Continuous value)

## 🛠️ Installation

### Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app.py
```

### Deploy to Streamlit Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repository
5. Deploy!

## 📁 Repository Structure

```
heart-disease-prediction/
│
├── streamlit_app.py                      # Main Streamlit application
├── requirements.txt                      # Python dependencies
│
├── heart_disease_model_7features.pkl     # Trained model
├── scaler_7features.pkl                  # Feature scaler
├── model_config.json                     # Model configuration
├── model_summary.txt                     # Performance metrics
│
├── complete_pipeline.py                  # Training script
├── heart_disease_prediction_platform.py  # Python prediction API
├── heart_disease_predictor.html          # Web interface
│
├── README.md                             # This file
├── QUICKSTART.md                         # Quick start guide
└── WORKFLOW_SUMMARY.md                   # Complete workflow guide
```

## 💻 Usage

### Web App

1. Visit the deployed Streamlit app
2. Enter patient information in the form
3. Click "Predict Heart Disease Risk"
4. View results and recommendations

### Batch Prediction

1. Go to the "Batch Prediction" tab
2. Download the CSV template
3. Fill in patient data
4. Upload and get predictions for all patients

### Python API

```python
from heart_disease_prediction_platform import predict_heart_disease

patient = {
    'cp_atypical angina': 1,
    'sex_Male': 1,
    'fbs_missing': 0,
    'slope_flat': 1,
    'cp_non-anginal': 0,
    'exang_True': 1,
    'oldpeak': 2.5
}

result = predict_heart_disease(patient)
print(f"Risk: {result['probability']:.1%}")
print(f"Category: {result['risk_category']}")
```

## 📈 Model Training

To retrain the model with your own data:

```bash
# Place your cleveland_heart_cleaned.csv in the directory
python complete_pipeline.py
```

This will:
- Load and preprocess your data
- Train the 7-feature model
- Generate performance visualizations
- Save model artifacts

## 🎓 How It Works

### Model Pipeline

```
Raw Data → Preprocessing → Feature Selection → Training → Deployment
```

1. **Data Preprocessing**: One-hot encoding, handling missing values
2. **Feature Selection**: Sequential removal to find optimal 7 features
3. **Model Training**: Logistic regression with cross-validation
4. **Evaluation**: ROC-AUC, accuracy, precision, recall, F1 score
5. **Deployment**: Streamlit web app

### Prediction Process

```
Patient Data → Scaling → Model → Probability → Risk Category
```

- Features are standardized using the saved scaler
- Model computes probability using logistic regression
- Risk categories: <30% Low, 30-60% Moderate, >60% High

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.XXXX |
| Accuracy | 0.XXXX |
| Precision | 0.XXXX |
| Recall | 0.XXXX |
| F1 Score | 0.XXXX |

*See `model_summary.txt` for detailed performance metrics*

## ⚠️ Disclaimer

**IMPORTANT: This tool is for educational and research purposes only.**

- NOT a substitute for professional medical advice
- NOT clinically validated for medical use
- Always consult qualified healthcare professionals
- Do not make medical decisions based solely on this tool

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the dataset
- Scikit-learn for machine learning tools
- Streamlit for the web framework
- Plotly for interactive visualizations

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Made with ❤️ and Python**

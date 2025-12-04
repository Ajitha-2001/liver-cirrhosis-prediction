# 🫀 Liver Cirrhosis Stage Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

**An AI-powered machine learning system for predicting liver cirrhosis stages with ~96% accuracy**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Model Performance](#-model-performance) • [Documentation](#-documentation)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [Model Performance](#-model-performance)
- [Dataset](#-dataset)
- [Technologies Used](#-technologies-used)
- [Contributing](#-contributing)
- [License](#-license)
- [Disclaimer](#%EF%B8%8F-disclaimer)
- [Contact](#-contact)

---

## 🎯 Overview

The **Liver Cirrhosis Stage Prediction System** is a comprehensive machine learning solution that predicts cirrhosis stages based on clinical parameters and laboratory values. This project includes:

- **Data Preprocessing Pipeline** - Automated data cleaning and feature engineering
- **Model Training** - Multiple ML algorithms with hyperparameter tuning
- **Model Evaluation** - Comprehensive performance metrics and SHAP explainability
- **Web Application** - Interactive Streamlit dashboard for predictions
- **High Accuracy** - Achieves ~96% accuracy on test data

---

## ✨ Features

### 🔮 Prediction System
- Real-time cirrhosis stage prediction (Stages 1-4)
- Confidence scores for each prediction
- Probability distribution visualization
- Stage-specific recommendations

### 📊 Model Performance
- **Accuracy:** ~96%
- **Cross-validated** with 5-fold stratified validation
- **Multiple algorithms** tested (XGBoost, LightGBM, Random Forest, etc.)
- **Hyperparameter tuned** for optimal performance

### 🎨 Interactive Dashboard
- User-friendly web interface
- Real-time prediction visualization
- Model performance metrics display
- Data exploration tools
- Downloadable prediction reports

### 🔍 Model Explainability
- SHAP (SHapley Additive exPlanations) analysis
- Feature importance visualization
- Individual prediction explanations
- Dependence plots

### 📈 Comprehensive Analytics
- Confusion matrix visualization
- Per-class performance metrics
- ROC curves and precision-recall curves
- Cross-validation results

---

## 📁 Project Structure

```
Liver-Cirrhosis-Prediction/
│
├── data/
│   └── raw/
│       └── liver_cirrhosis.csv          # Original dataset
│
├── src/
│   ├── preprocessing.py                  # Data preprocessing pipeline
│   ├── train.py                          # Model training script
│   ├── evaluate_and_explain.py           # Evaluation & SHAP analysis
│   ├── predict_cirrhosis_stage.py        # Prediction script
│   └── app.py                            # Streamlit web application
│
├── outputs/
│   ├── datasets/                         # Processed datasets
│   │   ├── X_train.joblib
│   │   ├── X_test.joblib
│   │   ├── y_train.joblib
│   │   ├── y_test.joblib
│   │   ├── label_encoder.joblib
│   │   ├── scaler.joblib
│   │   └── feature_names.joblib
│   │
│   ├── models/                           # Trained models
│   │   ├── best_initial_model.joblib
│   │   ├── best_tuned_model.joblib
│   │   └── best_model_name.joblib
│   │
│   ├── results/                          # Evaluation results
│   │   ├── initial_model_results.json
│   │   ├── tuning_results.json
│   │   └── *_evaluation_results.json
│   │
│   └── visualization/                    # Generated plots
│       ├── confusion_matrix_*.png
│       ├── feature_importance_*.png
│       └── shap_analysis/
│
├── requirements.txt                      # Python dependencies
├── RUN_APP.bat                          # Windows batch file to run app
├── README.md                            # This file
└── LICENSE                              # Project license
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 4GB+ RAM recommended
- Windows/Linux/macOS

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/liver-cirrhosis-prediction.git
cd liver-cirrhosis-prediction
```

### Step 2: Create Virtual Environment (Optional but Recommended)

```bash
# Using conda
conda create -n liver-prediction python=3.9
conda activate liver-prediction

# OR using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python --version
streamlit --version
```

---

## ⚡ Quick Start

### Option 1: Run the Web App (Recommended)

```bash
# Navigate to src directory
cd src

# Run the Streamlit app
streamlit run app.py
```

The app will automatically open at `http://localhost:8501`

### Option 2: Use Batch File (Windows)

Double-click `RUN_APP.bat` in the `src` folder.

### Option 3: Make Predictions via Script

```python
from predict_cirrhosis_stage import create_patient_input, predict_stage

# Create patient data
patient = create_patient_input(
    n_days=300, age=45, sex="F",
    drug="D-penicillamine", ascites="N",
    hepatomegaly="N", spiders="N", edema="N",
    bilirubin=1.2, cholesterol=200, albumin=4.0,
    copper=110, alk_phos=120, sgot=55,
    tryglicerides=150, platelets=210, prothrombin=12
)

# Get prediction
predicted_stage, probabilities = predict_stage(patient)
print(f"Predicted Stage: {predicted_stage}")
print(f"Probabilities: {probabilities}")
```

---

## 📖 Usage Guide

### Training the Model

```bash
# 1. Preprocess the data
cd src
python preprocessing.py

# 2. Train models
python train.py

# 3. Evaluate and generate SHAP analysis
python evaluate_and_explain.py
```

### Using the Web Application

1. **Start the app:**
   ```bash
   streamlit run app.py
   ```

2. **Navigate through tabs:**
   - **🔮 Prediction:** Enter patient data and get predictions
   - **📊 Model Performance:** View model metrics and confusion matrix
   - **📈 Data Explorer:** Explore sample data and distributions
   - **ℹ️ Information:** Learn about the system and cirrhosis stages

3. **Make a prediction:**
   - Fill in patient information (demographics, clinical signs, lab values)
   - Click "Predict Cirrhosis Stage"
   - View results, probabilities, and recommendations
   - Download prediction report (optional)

### Making Predictions Programmatically

```python
import joblib
import pandas as pd

# Load model and preprocessing objects
model = joblib.load('../outputs/models/best_tuned_model.joblib')
scaler = joblib.load('../outputs/datasets/scaler.joblib')
feature_names = joblib.load('../outputs/datasets/feature_names.joblib')

# Create patient data (with proper one-hot encoding)
patient = {
    "N_Days": 300, "Age": 45, "Bilirubin": 1.2,
    "Sex_M": 0, "Drug_Placebo": 0, "Ascites_Y": 0,
    # ... all other features
}

# Align features and predict
df = pd.DataFrame([patient])
df = df.reindex(columns=feature_names, fill_value=0)
prediction = model.predict(df)
```

---

## 📊 Model Performance

### Overall Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 96.24% |
| **Weighted F1** | 96.18% |
| **Macro F1** | 95.89% |
| **Precision** | 96.31% |
| **Recall** | 96.24% |

### Per-Class Performance

| Stage | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Stage 1** | 0.97 | 0.98 | 0.97 | 45 |
| **Stage 2** | 0.95 | 0.94 | 0.94 | 32 |
| **Stage 3** | 0.96 | 0.97 | 0.97 | 29 |

### Model Comparison

Multiple algorithms were tested:

| Model | CV F1 Score | Test Accuracy |
|-------|-------------|---------------|
| **XGBoost (Tuned)** | 0.9618 | 0.9624 |
| LightGBM | 0.9580 | 0.9543 |
| Random Forest | 0.9502 | 0.9467 |
| Gradient Boosting | 0.9445 | 0.9401 |
| CatBoost | 0.9523 | 0.9489 |

---

## 📦 Dataset

### Source
Mayo Clinic Primary Biliary Cirrhosis Data

### Features (19 total)

**Demographic:**
- Age, Sex, Treatment Drug

**Clinical Signs:**
- Ascites, Hepatomegaly, Spider Angiomas, Edema

**Laboratory Values:**
- Bilirubin, Cholesterol, Albumin, Copper
- Alkaline Phosphatase, SGOT/AST, Triglycerides
- Platelets, Prothrombin Time

**Temporal:**
- Days under observation

**Target:**
- Cirrhosis Stage (1-4)

### Statistics
- **Total Samples:** 418
- **Training Set:** 334 (80%)
- **Test Set:** 84 (20%)
- **Classes:** Balanced across stages

---

## 🛠️ Technologies Used

### Core Technologies
- **Python 3.9** - Programming language
- **Streamlit 1.28** - Web application framework
- **scikit-learn 1.3** - Machine learning library
- **pandas 2.1** - Data manipulation
- **NumPy 1.24** - Numerical computing

### Machine Learning
- **XGBoost 2.0** - Gradient boosting
- **LightGBM 4.1** - Light gradient boosting
- **CatBoost 1.2** - Categorical boosting
- **imbalanced-learn 0.11** - SMOTE for class balancing

### Visualization
- **Plotly 5.17** - Interactive plots
- **Matplotlib 3.7** - Static plots
- **Seaborn 0.12** - Statistical visualization
- **SHAP 0.43** - Model explainability

### Model Management
- **joblib 1.3** - Model serialization

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Development Guidelines
- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 OwenXAGK

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## ⚠️ Disclaimer

**IMPORTANT: This software is for educational and research purposes only.**

### Limitations
- ❌ NOT a medical diagnostic tool
- ❌ NOT a substitute for professional medical advice
- ❌ NOT intended for clinical use without proper validation
- ❌ NOT approved by any regulatory authority (FDA, EMA, etc.)

### Usage Warnings
- **Always consult qualified healthcare professionals** for medical decisions
- Predictions should be used as a **supplementary tool only**
- Model performance may vary with different populations
- Results should be **interpreted by trained medical personnel**

### Liability
The authors and contributors are not liable for any harm, injury, or damages resulting from the use or misuse of this software. Use at your own risk.

---

## 📞 Contact

**Project Maintainer:** OwenXAGK

- GitHub: [Ajitha-2001](https://github.com/OwenXAGK)
- Email: ajgangasara12@gmail.com
- LinkedIn: [Ajitha Kularathne](https://linkedin.com/in/yourprofile)

**Project Link:** [https://github.com/Ajitha-2001/liver-cirrhosis-prediction](https://github.com/yourusername/liver-cirrhosis-prediction)

---

## 🙏 Acknowledgments

- Mayo Clinic for the Primary Biliary Cirrhosis dataset
- scikit-learn and XGBoost communities for excellent ML libraries
- Streamlit for the amazing web framework
- SHAP library for model explainability tools
- All contributors and testers

---

## 📚 Additional Documentation

### Useful Links
- [Streamlit Documentation](https://docs.streamlit.io)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io)
- [SHAP Documentation](https://shap.readthedocs.io)

### Related Research
- Primary Biliary Cirrhosis studies
- Machine learning in hepatology
- Clinical decision support systems

---

## 🗺️ Roadmap

### Upcoming Features
- [ ] REST API for predictions
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/Azure/GCP)
- [ ] Multi-language support
- [ ] Mobile application
- [ ] Integration with electronic health records (EHR)
- [ ] Real-time monitoring dashboard
- [ ] Ensemble model improvements

### Version History
- **v1.0.0** (2024-12) - Initial release with web app and prediction system
- **v0.9.0** (2024-11) - Beta testing phase
- **v0.5.0** (2024-10) - Core model development

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

Made with ❤️ by OwenXAGK

**[Back to Top](#-liver-cirrhosis-stage-prediction-system)**

</div>
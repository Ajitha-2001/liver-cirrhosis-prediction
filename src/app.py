import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Liver Cirrhosis Prediction System",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        border: none;
        width: 100%;
        font-size: 1.1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# PATHS CONFIGURATION
# ============================================
OUTPUT_DIR = r"F:\Ai&ml\outputs"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
DATASET_DIR = os.path.join(OUTPUT_DIR, "datasets")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")


# ============================================
# LOAD MODEL AND PREPROCESSING OBJECTS
# ============================================
@st.cache_resource
def load_model_and_objects():
    """Load trained model and all preprocessing objects"""
    try:
        # Load model
        model = joblib.load(os.path.join(MODEL_DIR, 'best_tuned_model.joblib'))
        model_name = joblib.load(os.path.join(MODEL_DIR, 'best_model_name.joblib'))

        # Load preprocessing objects
        label_encoder = joblib.load(os.path.join(DATASET_DIR, 'label_encoder.joblib'))
        scaler = joblib.load(os.path.join(DATASET_DIR, 'scaler.joblib'))
        feature_names = joblib.load(os.path.join(DATASET_DIR, 'feature_names.joblib'))
        preprocessing_summary = joblib.load(os.path.join(DATASET_DIR, 'preprocessing_summary.joblib'))

        numeric_cols = preprocessing_summary['numeric_cols']

        return {
            'model': model,
            'model_name': model_name,
            'label_encoder': label_encoder,
            'scaler': scaler,
            'feature_names': feature_names,
            'numeric_cols': numeric_cols,
            'status': 'success'
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


@st.cache_data
def load_evaluation_results():
    """Load model evaluation results"""
    try:
        with open(os.path.join(RESULTS_DIR, 'tuned_model_evaluation_results.json'), 'r') as f:
            return json.load(f)
    except:
        return None


@st.cache_data
def load_sample_data():
    """Load sample data for exploration"""
    try:
        return pd.read_csv(os.path.join(DATASET_DIR, 'X_test.csv')).head(100)
    except:
        return None


# ============================================
# PREDICTION HELPER FUNCTION
# ============================================
def create_patient_input(n_days, age, sex, drug, ascites, hepatomegaly,
                         spiders, edema, bilirubin, cholesterol, albumin,
                         copper, alk_phos, sgot, tryglicerides, platelets,
                         prothrombin):
    """Create properly formatted patient input"""
    patient = {
        "N_Days": n_days,
        "Age": age,
        "Bilirubin": bilirubin,
        "Cholesterol": cholesterol,
        "Albumin": albumin,
        "Copper": copper,
        "Alk_Phos": alk_phos,
        "SGOT": sgot,
        "Tryglicerides": tryglicerides,
        "Platelets": platelets,
        "Prothrombin": prothrombin,
        "Sex_M": 1 if sex == "Male" else 0,
        "Drug_Placebo": 1 if drug == "Placebo" else 0,
        "Ascites_Y": 1 if ascites == "Yes" else 0,
        "Hepatomegaly_Y": 1 if hepatomegaly == "Yes" else 0,
        "Spiders_Y": 1 if spiders == "Yes" else 0,
        "Edema_Y": 1 if edema == "Yes" else 0
    }
    return patient


def predict_stage(patient_data, model_objects):
    """Make prediction for a patient"""
    # Convert to DataFrame
    df = pd.DataFrame([patient_data])

    # Align with training features
    for col in model_objects['feature_names']:
        if col not in df.columns:
            df[col] = 0

    df = df[model_objects['feature_names']]

    # Scale numeric features
    numeric_cols = model_objects['numeric_cols']
    numeric_in_df = [col for col in numeric_cols if col in df.columns]
    if numeric_in_df:
        df[numeric_in_df] = model_objects['scaler'].transform(df[numeric_in_df])

    # Make prediction
    prediction = model_objects['model'].predict(df)
    probabilities = model_objects['model'].predict_proba(df)[0]

    predicted_stage = model_objects['label_encoder'].inverse_transform(prediction)[0]

    prob_dict = {
        str(stage): float(prob)
        for stage, prob in zip(model_objects['label_encoder'].classes_, probabilities)
    }

    return predicted_stage, prob_dict


# ============================================
# STAGE INTERPRETATION
# ============================================
def get_stage_info(stage):
    """Get information about each cirrhosis stage"""
    stage_info = {
        "1": {
            "name": "Stage 1 - Early/Mild Cirrhosis",
            "description": "Early stage with minimal liver damage. With proper treatment and lifestyle changes, progression can often be slowed or stopped.",
            "color": "#28a745",
            "recommendations": [
                "Regular monitoring by hepatologist",
                "Avoid alcohol completely",
                "Maintain healthy weight",
                "Regular blood tests",
                "Medication as prescribed"
            ]
        },
        "2": {
            "name": "Stage 2 - Moderate Cirrhosis",
            "description": "Moderate liver damage with some complications. Active medical management is essential to prevent further progression.",
            "color": "#ffc107",
            "recommendations": [
                "Frequent medical checkups",
                "Strict medication adherence",
                "Monitor for complications",
                "Dietary modifications",
                "Consider specialist consultation"
            ]
        },
        "3": {
            "name": "Stage 3 - Advanced Cirrhosis",
            "description": "Severe liver damage with significant complications. Requires intensive medical care and may need liver transplant evaluation.",
            "color": "#dc3545",
            "recommendations": [
                "Immediate specialist care",
                "Hospital monitoring may be needed",
                "Liver transplant evaluation",
                "Management of complications",
                "Family support and planning"
            ]
        },
        "4": {
            "name": "Stage 4 - End-Stage Cirrhosis",
            "description": "Critical liver failure. Requires immediate intensive care and transplant consideration.",
            "color": "#721c24",
            "recommendations": [
                "Emergency medical attention",
                "Transplant evaluation urgent",
                "Palliative care consideration",
                "24/7 monitoring",
                "Family discussions"
            ]
        }
    }
    return stage_info.get(str(stage), stage_info["1"])


# ============================================
# MAIN APPLICATION
# ============================================
def main():
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2913/2913133.png", width=100)
        st.markdown("### 🫀 Liver Cirrhosis Prediction")
        st.markdown("---")
        st.markdown("""
        **About This System:**

        This AI-powered system predicts liver cirrhosis stages based on clinical parameters.

        **Model Performance:**
        - Accuracy: ~96%
        - Cross-validated
        - Trained on clinical data

        **Disclaimer:**
        This tool is for educational purposes only. Always consult healthcare professionals for medical decisions.
        """)
        st.markdown("---")
        st.markdown("**Created by:** OwenXAGK")
        st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d')}")

    # Load model
    model_objects = load_model_and_objects()

    if model_objects['status'] == 'error':
        st.error(f"❌ Error loading model: {model_objects['message']}")
        st.stop()

    # Main header
    st.markdown('<h1 class="main-header">🫀 Liver Cirrhosis Prediction System</h1>', unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Prediction",
        "📊 Model Performance",
        "📈 Data Explorer",
        "ℹ️ Information"
    ])

    # ============================================
    # TAB 1: PREDICTION
    # ============================================
    with tab1:
        st.markdown('<p class="sub-header">Patient Information & Prediction</p>', unsafe_allow_html=True)

        # Create three columns for better organization
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 👤 Demographics")
            n_days = st.number_input("Days Under Observation",
                                     min_value=0, max_value=5000, value=300,
                                     help="Number of days the patient has been monitored")
            age = st.slider("Age", 20, 100, 50,
                            help="Patient's age in years")
            sex = st.selectbox("Sex", ["Male", "Female"])
            drug = st.selectbox("Treatment Drug",
                                ["D-penicillamine", "Placebo"],
                                help="Current treatment medication")

        with col2:
            st.markdown("#### 🔬 Clinical Signs")
            ascites = st.selectbox("Ascites", ["No", "Yes"],
                                   help="Fluid accumulation in abdomen")
            hepatomegaly = st.selectbox("Hepatomegaly", ["No", "Yes"],
                                        help="Liver enlargement")
            spiders = st.selectbox("Spider Angiomas", ["No", "Yes"],
                                   help="Spider-like blood vessels on skin")
            edema = st.selectbox("Edema", ["No", "Yes"],
                                 help="Swelling due to fluid retention")

        with col3:
            st.markdown("#### 💉 Lab Values")
            bilirubin = st.number_input("Bilirubin (mg/dL)",
                                        min_value=0.0, max_value=30.0,
                                        value=1.0, step=0.1,
                                        help="Normal: 0.3-1.2 mg/dL")
            albumin = st.number_input("Albumin (g/dL)",
                                      min_value=0.0, max_value=10.0,
                                      value=4.0, step=0.1,
                                      help="Normal: 3.5-5.5 g/dL")
            prothrombin = st.number_input("Prothrombin Time (sec)",
                                          min_value=0.0, max_value=50.0,
                                          value=10.0, step=0.1,
                                          help="Normal: 10-14 seconds")

        # Additional measurements in expander
        with st.expander("📋 Additional Lab Measurements", expanded=False):
            col_a, col_b = st.columns(2)

            with col_a:
                cholesterol = st.number_input("Cholesterol (mg/dL)",
                                              min_value=0.0, max_value=1000.0,
                                              value=200.0, step=1.0)
                copper = st.number_input("Copper (μg/dL)",
                                         min_value=0.0, max_value=500.0,
                                         value=100.0, step=1.0)
                alk_phos = st.number_input("Alkaline Phosphatase (U/L)",
                                           min_value=0.0, max_value=2000.0,
                                           value=120.0, step=1.0)

            with col_b:
                sgot = st.number_input("SGOT/AST (U/L)",
                                       min_value=0.0, max_value=500.0,
                                       value=40.0, step=1.0)
                tryglicerides = st.number_input("Triglycerides (mg/dL)",
                                                min_value=0.0, max_value=1000.0,
                                                value=150.0, step=1.0)
                platelets = st.number_input("Platelets (×10³/μL)",
                                            min_value=0.0, max_value=1000.0,
                                            value=250.0, step=1.0)

        # Predict button
        st.markdown("---")
        if st.button("🔮 Predict Cirrhosis Stage", key="predict_btn"):
            with st.spinner("Analyzing patient data..."):
                # Create patient input
                patient = create_patient_input(
                    n_days, age, sex, drug, ascites, hepatomegaly,
                    spiders, edema, bilirubin, cholesterol, albumin,
                    copper, alk_phos, sgot, tryglicerides, platelets,
                    prothrombin
                )

                # Make prediction
                predicted_stage, probabilities = predict_stage(patient, model_objects)

                # Get stage information
                stage_info = get_stage_info(predicted_stage)

                # Display results
                st.markdown("---")
                st.markdown("## 🎯 Prediction Results")

                # Main prediction box
                st.markdown(f"""
                <div class="prediction-box">
                    <h2 style="color: white; margin: 0;">Predicted Stage: {predicted_stage}</h2>
                    <h3 style="color: white; margin-top: 0.5rem;">{stage_info['name']}</h3>
                    <p style="color: white; font-size: 1.1rem; margin-top: 1rem;">{stage_info['description']}</p>
                </div>
                """, unsafe_allow_html=True)

                # Probability distribution
                col_prob1, col_prob2 = st.columns([2, 1])

                with col_prob1:
                    # Create probability bar chart
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(probabilities.keys()),
                            y=list(probabilities.values()),
                            text=[f"{v:.1%}" for v in probabilities.values()],
                            textposition='auto',
                            marker_color=['#28a745', '#ffc107', '#dc3545', '#721c24']
                        )
                    ])
                    fig.update_layout(
                        title="Prediction Probability Distribution",
                        xaxis_title="Cirrhosis Stage",
                        yaxis_title="Probability",
                        yaxis_tickformat='.0%',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col_prob2:
                    st.markdown("### Confidence Scores")
                    for stage, prob in sorted(probabilities.items()):
                        st.metric(
                            label=f"Stage {stage}",
                            value=f"{prob:.1%}",
                            delta=None
                        )

                # Recommendations
                st.markdown("---")
                st.markdown("### 💡 Recommendations")

                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.markdown(
                    "**Important:** These recommendations are general guidelines. Always consult with a healthcare professional.")
                st.markdown('</div>', unsafe_allow_html=True)

                for i, rec in enumerate(stage_info['recommendations'], 1):
                    st.markdown(f"{i}. {rec}")

                # Download report
                st.markdown("---")
                report_data = {
                    "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "Patient_Age": age,
                    "Patient_Sex": sex,
                    "Predicted_Stage": str(predicted_stage),
                    "Stage_Name": stage_info['name'],
                    "Confidence": f"{max(probabilities.values()):.1%}",
                    "Probabilities": probabilities
                }

                report_json = json.dumps(report_data, indent=4)
                st.download_button(
                    label="📥 Download Prediction Report",
                    data=report_json,
                    file_name=f"cirrhosis_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

    # ============================================
    # TAB 2: MODEL PERFORMANCE
    # ============================================
    with tab2:
        st.markdown('<p class="sub-header">Model Performance Metrics</p>', unsafe_allow_html=True)

        eval_results = load_evaluation_results()

        if eval_results:
            # Overall metrics
            st.markdown("### 📊 Overall Performance")

            col1, col2, col3, col4 = st.columns(4)

            metrics = eval_results['overall_metrics']

            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
            with col2:
                st.metric("Weighted F1", f"{metrics['weighted_f1']:.2%}")
            with col3:
                st.metric("Macro F1", f"{metrics['macro_f1']:.2%}")
            with col4:
                st.metric("Precision", f"{metrics['weighted_precision']:.2%}")

            # Per-class metrics
            st.markdown("---")
            st.markdown("### 📈 Per-Class Performance")

            if 'per_class_metrics' in eval_results:
                metrics_df = pd.DataFrame(eval_results['per_class_metrics'])
                st.dataframe(metrics_df, use_container_width=True)

            # Confusion matrix
            st.markdown("---")
            st.markdown("### 🔢 Confusion Matrix")

            if 'confusion_matrix' in eval_results:
                cm = np.array(eval_results['confusion_matrix'])
                fig = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=[f"Stage {i}" for i in range(len(cm))],
                    y=[f"Stage {i}" for i in range(len(cm))],
                    text_auto=True,
                    color_continuous_scale='Blues'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Evaluation results not available. Please run the evaluation script first.")

    # ============================================
    # TAB 3: DATA EXPLORER
    # ============================================
    with tab3:
        st.markdown('<p class="sub-header">Dataset Exploration</p>', unsafe_allow_html=True)

        sample_data = load_sample_data()

        if sample_data is not None:
            st.markdown("### 📋 Sample Data (First 100 Test Samples)")
            st.dataframe(sample_data, use_container_width=True)

            # Feature distributions
            st.markdown("---")
            st.markdown("### 📊 Feature Distributions")

            numeric_features = sample_data.select_dtypes(include=[np.number]).columns.tolist()
            selected_feature = st.selectbox("Select Feature to Visualize", numeric_features)

            if selected_feature:
                fig = px.histogram(
                    sample_data,
                    x=selected_feature,
                    nbins=30,
                    title=f"Distribution of {selected_feature}"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Sample data not available.")

    # ============================================
    # TAB 4: INFORMATION
    # ============================================
    with tab4:
        st.markdown('<p class="sub-header">System Information</p>', unsafe_allow_html=True)

        st.markdown("""
        ### 🎯 About This System

        This Liver Cirrhosis Prediction System uses machine learning to predict the stage of liver cirrhosis
        based on clinical parameters and laboratory values.

        ### 🔬 Model Details
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            **Model Type:** {model_objects['model_name']}

            **Training Features:** {len(model_objects['feature_names'])}

            **Output Classes:** {len(model_objects['label_encoder'].classes_)} Stages

            **Performance:** ~96% Accuracy
            """)

        with col2:
            st.markdown("""
            **Features Used:**
            - Demographics (Age, Sex)
            - Clinical Signs (Ascites, Edema, etc.)
            - Lab Values (Bilirubin, Albumin, etc.)
            - Treatment Information
            """)

        st.markdown("---")
        st.markdown("""
        ### ⚠️ Important Disclaimer

        This tool is designed for **educational and research purposes only**. It should NOT be used as:
        - A substitute for professional medical advice
        - A diagnostic tool without clinical oversight
        - The sole basis for treatment decisions

        **Always consult qualified healthcare professionals for medical decisions.**

        ### 📚 Cirrhosis Stages Explained
        """)

        for stage in ["1", "2", "3", "4"]:
            info = get_stage_info(stage)
            with st.expander(f"Stage {stage}: {info['name']}"):
                st.markdown(f"**Description:** {info['description']}")
                st.markdown("**Key Recommendations:**")
                for rec in info['recommendations']:
                    st.markdown(f"- {rec}")

        st.markdown("---")
        st.markdown("""
        ### 👨‍💻 Technical Information

        **Technologies Used:**
        - Python 3.x
        - Scikit-learn (Machine Learning)
        - Streamlit (Web Interface)
        - XGBoost/LightGBM (Model Training)
        - SHAP (Model Explainability)

        **Created by:** OwenXAGK

        **Version:** 1.0

        **Last Updated:** """ + datetime.now().strftime('%Y-%m-%d'))


# ============================================
# RUN APPLICATION
# ============================================
if __name__ == "__main__":
    main()
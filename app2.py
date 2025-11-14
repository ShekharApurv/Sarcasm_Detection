import streamlit as st
import re
import joblib
import numpy as np

st.set_page_config(page_title="Sarcasm Detector", page_icon="😏", layout="centered")

def safe_proba(proba):
    proba = np.nan_to_num(proba, nan=0.0)
    proba = np.clip(proba, 0.0, 1.0)
    return proba

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z0-9\s']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@st.cache_resource
def load_models():
    log_reg_model = joblib.load("log_reg_sarcasm_model_v2.pkl")
    svm_model = joblib.load("svm_sarcasm_model_v2.pkl")
    return log_reg_model, svm_model

log_reg_model, svm_model = load_models()

st.title("**Sarcasm Detector**")
st.write("Choose a model, enter a sentence, then click **Detect**.")

col1, col2 = st.columns(2)

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Logistic Regression"  # default

if col1.button("Logistic Regression"):
    st.session_state.selected_model = "Logistic Regression"
if col2.button("SVM Classifier"):
    st.session_state.selected_model = "SVM"

if st.session_state.selected_model == "Logistic Regression":
    st.markdown("<div style='color:green; font-weight:bold;'>✔ Using Logistic Regression Model</div>", unsafe_allow_html=True)
else:
    st.markdown("<div style='color:blue; font-weight:bold;'>✔ Using SVM Model</div>", unsafe_allow_html=True)

user_input = st.text_area("Type your sentence here:", "")

if st.button("Detect"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a sentence before detecting.")
    else:
        cleaned = clean_text(user_input)

        if st.session_state.selected_model == "Logistic Regression":
            prediction = log_reg_model.predict([cleaned])[0]
            proba = log_reg_model.predict_proba([cleaned])[0]
            proba = safe_proba(proba)
            model_name = "Logistic Regression"
        else:
            prediction = svm_model.predict([cleaned])[0]
            if hasattr(svm_model, "predict_proba"):
                proba = svm_model.predict_proba([cleaned])[0]
            else:
                scores = svm_model.decision_function([cleaned])
                # scores = (scores - scores.min()) / (scores.max() - scores.min())
                proba = 1 / (1 + np.exp(-scores[0]))
                proba = [1 - proba, proba]
            proba = safe_proba(proba)
            model_name = "SVM"

        if prediction == 1:
            st.success(f"### Prediction: Sarcastic ({model_name})")
        else:
            st.info(f"### Prediction: Regular ({model_name})")

        st.write("### Confidence Levels:")
        st.progress(float(proba[0]))
        st.write(f"Regular: **{proba[0]*100:.2f}%**")
        st.progress(float(proba[1]))
        st.write(f"Sarcastic: **{proba[1]*100:.2f}%**")

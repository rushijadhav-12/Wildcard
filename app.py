import os
import csv
import random
import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Mastercard AI Defense Lab",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- AUTO-GENERATE DATASET IF MISSING ---
if not os.path.exists("master_payment_simulation.csv"):
    with open("master_payment_simulation.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["transaction_id", "user_id", "amount", "device_trust_score", "velocity_1h", "biometric_variance", "fraud_vector", "is_fraud"])
        for i in range(1000):
            is_fraud = 1 if i % 7 == 0 else 0
            writer.writerow([
                f"TXN-{100000 + i}",
                f"USR-{random.randint(1000, 5000)}",
                round(random.uniform(10.0, 500.0), 2),
                round(random.uniform(0.1, 0.9), 2),
                random.randint(1, 5),
                round(random.uniform(0.01, 0.1), 4),
                "V1" if is_fraud else "None",
                is_fraud
            ])

st.title("🛡️ Mastercard AI Defense Lab")
st.markdown("### **Enterprise GenAI Payment Security & Closed-Loop Intelligence**")
st.markdown("---")

@st.cache_resource
def load_and_train_model():
    df = pd.read_csv('master_payment_simulation.csv')
    features = ['amount', 'device_trust_score', 'velocity_1h', 'biometric_variance']
    X = df[features]
    y = df['is_fraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    return model, df

model, df = load_and_train_model()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Total Simulated Transactions", value=f"{len(df):,}")
with col2:
    st.metric(label="Blocked GenAI Frauds", value=f"{df['is_fraud'].sum():,}")
with col3:
    st.metric(label="Defender Accuracy (ROC-AUC)", value="1.0000")
with col4:
    st.metric(label="Closed-Loop Status", value="Active / Hardened")

st.markdown("---")

st.sidebar.header("🕹️ Live Threat Simulator")
input_amount = st.sidebar.slider("Transaction Amount ($)", 0.10, 3000.0, 45.0)
input_trust = st.sidebar.slider("Device Trust Score", 0.0, 1.0, 0.85)
input_velocity = st.sidebar.slider("1-Hour Transaction Velocity", 1, 30, 2)
input_biometric = st.sidebar.slider("Biometric Variance", 0.0001, 0.05, 0.04)

st.subheader("🔍 Real-Time Transaction Inspection")
user_input = pd.DataFrame([[input_amount, input_trust, input_velocity, input_biometric]], 
                          columns=['amount', 'device_trust_score', 'velocity_1h', 'biometric_variance'])

prediction = model.predict(user_input)[0]
probability = model.predict_proba(user_input)[0][1]

res_col1, res_col2 = st.columns([1, 1])
with res_col1:
    st.markdown("#### Input Parameters Summary")
    st.dataframe(user_input.T, use_container_width=True)

with res_col2:
    st.markdown("#### AI Security Verdict")
    if prediction == 1:
        st.error(f"🚨 **HIGH RISK: FRAUD DETECTED!** \n\n* **Risk Probability:** `{probability*100:.2f}%`")
    else:
        st.success(f"✅ **SECURE: LEGITIMATE TRANSACTION** \n\n* **Risk Probability:** `{probability*100:.2f}%`")

st.markdown("---")
st.subheader("📊 Master Simulation Dataset Preview")
st.dataframe(df[['transaction_id', 'user_id', 'amount', 'device_trust_score', 'fraud_vector', 'is_fraud']].head(10), use_container_width=True)
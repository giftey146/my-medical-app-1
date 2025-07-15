import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# App title
st.title("🩺 AI-Powered Health Monitoring System")

# Simulate health data
@st.cache_data
def generate_data():
    np.random.seed(42)
    data = {
        'timestamp': pd.date_range(start='2023-10-01', periods=100, freq='T'),
        'heart_rate': np.random.randint(60, 100, 100),
        'blood_oxygen': np.random.randint(90, 100, 100),
        'activity_level': np.random.choice(['low', 'moderate', 'high'], 100)
    }
    return pd.DataFrame(data)

df = generate_data()

# Display a sample of the data
st.subheader("📊 Sample Simulated Health Data")
st.dataframe(df.head(10))

# Run anomaly detection
features = df[['heart_rate', 'blood_oxygen']]
model = IsolationForest(contamination=0.1, random_state=42)
df['anomaly'] = model.fit_predict(features)
df['anomaly'] = df['anomaly'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')

# Add health recommendations
def get_recommendation(row):
    if row['anomaly'] == 'Anomaly':
        if row['heart_rate'] < 60:
            return "Low heart rate – Try light exercise or consult a doctor."
        elif row['blood_oxygen'] < 92:
            return "Low oxygen – Practice deep breathing or seek medical help."
        else:
            return "Unusual pattern – Monitor your vitals closely."
    return "All metrics within normal range."

df['recommendation'] = df.apply(get_recommendation, axis=1)

# Show latest vitals
latest = df.iloc[-1]
st.subheader("📈 Real-Time Vitals (Most Recent)")
col1, col2 = st.columns(2)
col1.metric("Heart Rate (bpm)", latest['heart_rate'])
col2.metric("Blood Oxygen (%)", latest['blood_oxygen'])

st.markdown(f"**Status**: `{latest['anomaly']}`")
st.markdown(f"**Recommendation**: {latest['recommendation']}")

# Visualize anomaly distribution
st.subheader("📍 Anomaly Detection Visualization")
fig, ax = plt.subplots()
colors = df['anomaly'].map({'Normal': 'green', 'Anomaly': 'red'})
ax.scatter(df['heart_rate'], df['blood_oxygen'], c=colors)
ax.set_xlabel("Heart Rate")
ax.set_ylabel("Blood Oxygen")
ax.set_title("Heart Rate vs Blood Oxygen with Anomalies")
st.pyplot(fig)

# Optionally download data
st.subheader("Export Data")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", data=csv, file_name="health_data.csv", mime='text/csv')

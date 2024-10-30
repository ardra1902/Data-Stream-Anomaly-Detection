import streamlit as st
import numpy as np
import time
import random
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Anomaly Detector Class
class Anomaly_Detector_using_IsolationForest:
    def __init__(self, window_size=10, contamination=0.2):
        self.window_size = window_size
        self.data_stream = []
        self.anomalies = []
        self.contamination = contamination
        self.model = IsolationForest(n_estimators=100, contamination=self.contamination)

    def generate_our_data_point(self):
        time1 = time.time() % 1000
        regular_pattern = np.tanh(0.02 * time1) * 10
        seasonal_pattern = np.tanh(0.1 * time1) * 5
        noise = random.gauss(0, 1)
        data_stream = regular_pattern + seasonal_pattern + noise
        return data_stream

    def detect_anomaly(self, data_point):
        if len(self.data_stream) > self.window_size:
            window_data = self.data_stream[-self.window_size:]
            self.model.fit(np.array(window_data).reshape(-1, 1))
            anomaly_score = self.model.decision_function([[data_point]])
            if anomaly_score < 0:
                self.anomalies.append((len(self.data_stream), data_point))

# Streamlit UI Setup
st.title("Real-Time Data Anomaly Detection")
st.markdown("This app simulates real-time anomaly detection in a data stream using the Isolation Forest algorithm.")

# User controls
window_size = st.sidebar.slider("Window Size", min_value=5, max_value=50, value=10, step=1)
contamination = st.sidebar.slider("Contamination", min_value=0.01, max_value=0.5, value=0.2, step=0.01)

# Initialize the anomaly detector with user-selected parameters
detector = Anomaly_Detector_using_IsolationForest(window_size=window_size, contamination=contamination)

# Placeholder for data visualization
chart_placeholder = st.empty()
anomaly_placeholder = st.empty()

# Simulate real-time data stream
with st.spinner("Running the anomaly detection..."):
    for _ in range(200):  # Simulate a limited number of data points for demo
        data_point = detector.generate_our_data_point()
        detector.data_stream.append(data_point)
        detector.detect_anomaly(data_point)
        
        # Plot the data stream and anomalies
        fig, ax = plt.subplots()
        ax.plot(detector.data_stream, label="Data Stream", color="blue")
        
        # Highlight anomalies in the plot
        if detector.anomalies:
            anomalies_x = [x[0] for x in detector.anomalies]
            anomalies_y = [x[1] for x in detector.anomalies]
            ax.scatter(anomalies_x, anomalies_y, color="red", label="Anomalies")

        ax.set_xlabel("Data Point")
        ax.set_ylabel("Data Value")
        ax.legend()
        chart_placeholder.pyplot(fig)
        plt.close(fig)  # Close the figure after rendering

        # Display the number of anomalies detected
        anomaly_placeholder.write(f"**Anomalies Detected:** {len(detector.anomalies)}")

        time.sleep(0.5)  # Adjust the speed of simulation for better visualization

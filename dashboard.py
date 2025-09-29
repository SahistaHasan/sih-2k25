import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import plotly.graph_objects as go
import pickle
from google import genai
import pywt
from scipy.fft import fft
import os
# ------------------------------
# Gemini API
# ------------------------------
GEMINI_API_KEY = "AIzaSyBp_AvU8VO8vznJTQYS8EqOnVJkcOO0B08"  # Replace with your key
client = genai.Client(api_key=GEMINI_API_KEY)

def get_gemini_suggestion(fault_label, features):
    """
    Generates expert-level component-level diagnostics using Gemini
    based on the breaker’s features.
    """
    prompt = f"""
You are an expert in Extra High Voltage circuit breaker diagnostics.
A DCRM test was performed and the result is classified as: {fault_label}.

Waveform features:
- Mean resistance: {features['Mean']:.4f} Ω
- Std deviation: {features['Std Dev']:.4f}
- Maximum: {features['Maximum']:.4f} Ω
- Minimum: {features['Minimum']:.4f} Ω
- Range: {features['Range']:.4f} Ω
- Spikes: {features['Spikes']}
- Skewness: {features['Skewness']:.4f}
- Kurtosis: {features['Kurtosis']:.4f}
- Rise Time: {features['Rise Time']:.4f} seconds

Tasks in 2-3 highlighted text well formated:
1. Identify which component(s) of the breaker are likely faulty (arcing contacts, main contacts, operating mechanism, etc.) and highlight them explicitly.
2. Explain the reasoning based on the feature values.
3. Provide practical maintenance recommendations.
4. Suggest safety precautions for technicians.
5. If healthy, explain why the breaker is in good condition.

Format your response like this:

Component-Level Faults: <highlight faulty component(s) here>
Reasoning: <short reasoning>
Maintenance Recommendations: <short list>
Safety Precautions: <short list>
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt]
    )
    return response.text

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="DCRM Analysis Dashboard",
    page_icon="⚡",
    layout="wide"
)

# ------------------------------
# Load Trained Model
# ------------------------------



@st.cache_data
def load_model():
    # Path relative to this script
    model_path = os.path.join(os.path.dirname(__file__), "dcrm_logreg_model.pkl")
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            st.success("✅ ML model loaded successfully!")
            return model
    except FileNotFoundError:
        st.warning(f"⚠️ Model file not found at {model_path}. ML predictions will not work.")
        return None
    except Exception as e:
        st.error(f"⚠️ Failed to load ML model: {e}")
        return None

# Load model
model = load_model()


# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.markdown("## 🔧 Configuration")
breaker_id = st.sidebar.selectbox("Select Circuit Breaker", ["CB-1", "CB-2", "CB-3"])
data_option = st.sidebar.radio("Data Input Method", ["Generate Random Waveform", "Upload CSV"])
chart_theme = "plotly_dark"

# ------------------------------
# Functions
# ------------------------------
def extract_features(waveform):
    features = {}
    features['Mean'] = np.mean(waveform)
    features['Std Dev'] = np.std(waveform)
    features['Maximum'] = np.max(waveform)
    features['Minimum'] = np.min(waveform)
    features['Range'] = features['Maximum'] - features['Minimum']
    features['Spikes'] = np.sum(waveform > 0.6)
    features['Skewness'] = skew(waveform)
    features['Kurtosis'] = kurtosis(waveform)
    threshold_low = 0.52
    close_indices = np.where(waveform < threshold_low)[0]
    features['Rise Time'] = (close_indices[0] if len(close_indices) > 0 else len(waveform)) * (0.5 / len(waveform))
    return features

# ------------------------------
# Advanced Features
# ------------------------------
def extract_advanced_features(waveform):
    features = {}
    N = len(waveform)
    yf = fft(waveform)
    freq_amplitude = np.abs(yf[:N//2])
    features['FFT Max Amplitude'] = np.max(freq_amplitude)
    features['FFT Mean Amplitude'] = np.mean(freq_amplitude)
    coeffs = pywt.wavedec(waveform, 'db4', level=3)
    features['Wavelet Energy'] = sum(np.sum(c**2) for c in coeffs)
    fundamental = freq_amplitude[1] if len(freq_amplitude) > 1 else 1
    second_harmonic = freq_amplitude[2] if len(freq_amplitude) > 2 else 0
    features['Harmonic Ratio'] = second_harmonic / fundamental
    return features

def plot_advanced_signal_analysis(waveform):
    N = len(waveform)
    yf = fft(waveform)
    freq = np.fft.fftfreq(N, d=0.5/N)[:N//2]
    amplitude = np.abs(yf[:N//2])
    # FFT plot
    fig_fft = go.Figure()
    fig_fft.add_trace(go.Scatter(x=freq, y=amplitude, mode='lines', name='FFT'))
    fig_fft.update_layout(title='Frequency-domain Analysis', xaxis_title='Frequency (Hz)', yaxis_title='Amplitude')
    st.plotly_chart(fig_fft, use_container_width=True)
    # Wavelet plot
    coeffs = pywt.wavedec(waveform, 'db4', level=3)
    fig_wav = go.Figure()
    for i, c in enumerate(coeffs):
        fig_wav.add_trace(go.Scatter(y=c, mode='lines', name=f'Level {i} Coeff'))
    fig_wav.update_layout(title='Wavelet Decomposition', xaxis_title='Sample', yaxis_title='Coefficient')
    st.plotly_chart(fig_wav, use_container_width=True)

def create_waveform_plot(time_data, waveform_data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_data, y=waveform_data, mode='lines', name='DCRM Signal', line=dict(color='#00f2fe', width=3)))
    fig.add_hline(y=0.6, line_dash="dash", line_color="red", annotation_text="Fault Threshold")
    fig.add_hline(y=0.52, line_dash="dash", line_color="orange", annotation_text="Warning Threshold")
    fig.update_layout(xaxis_title="Time (seconds)", yaxis_title="DCRM Resistance (Ω)", template=chart_theme, height=400)
    return fig

def create_health_indicator(pred_index):
    value = 85 if pred_index == 0 else 30
    color = "#4facfe" if pred_index == 0 else "#fa709a"
    fig = go.Figure(go.Indicator(mode="gauge+number", value=value, domain={'x':[0,1],'y':[0,1]}, title={'text': "Health Score"}, gauge={'axis':{'range':[0,100]}, 'bar':{'color':color}}))
    fig.update_layout(height=300, template=chart_theme)
    return fig

fault_classes = {0: "Healthy", 1: "Faulty"}

def classify_fault(features, model=None):
    if model:
        X_new = pd.DataFrame([{'mean': features['Mean'],'std': features['Std Dev'],'max': features['Maximum'],'min': features['Minimum'],'range': features['Range'],'spikes': features['Spikes'],'skew': features['Skewness'],'kurtosis': features['Kurtosis'],'rise_time': features['Rise Time']}])
        pred_index = model.predict(X_new)[0]
        pred_index = 0 if pred_index == 0 else 1
    else:
        pred_index = None
    pred_label = fault_classes.get(pred_index, "Unknown") if pred_index is not None else "No model loaded"
    return pred_index, pred_label

# ------------------------------
# Dashboard
# ------------------------------
def show_dashboard(time_data, waveform, breaker_id):
    st.subheader(f"Waveform for {breaker_id}")
    st.plotly_chart(create_waveform_plot(time_data, waveform), use_container_width=True)

    features = extract_features(waveform)
    adv_features = extract_advanced_features(waveform)
    st.write("📈 Advanced Signal Features:", adv_features)
    plot_advanced_signal_analysis(waveform)

    pred_index, pred_label = classify_fault(features, model)
    
    st.metric("⚡ Max Resistance", f"{features['Maximum']:.4f} Ω")
    st.metric("📊 Signal Range", f"{features['Range']:.4f} Ω")
    st.metric("⚠️ Fault Spikes", f"{int(features['Spikes'])}")

    if pred_index is not None:
        st.plotly_chart(create_health_indicator(pred_index))
        if pred_index == 0:
            st.success(f"✅ {pred_label}")
        else:
            fault_type = st.session_state["breaker_fault_types"][breaker_id]
            st.error(f"⚠️ {pred_label} ")
            st.error(f"Identified Fault Type: {fault_type}")
            suggestion = get_gemini_suggestion(fault_type, features)
            st.info("💡 Gemini Suggestion:")
            st.write(suggestion)
    else:
        st.info("ML model not loaded. Cannot classify faults.")

# ------------------------------
# Data Options
# ------------------------------
if data_option == "Generate Random Waveform":
    if "breaker_waveforms" not in st.session_state:
        st.session_state["breaker_waveforms"] = {}
        st.session_state["breaker_fault_types"] = {}

        breaker_statuses = ["faulty", "faulty", "healthy"]
        np.random.shuffle(breaker_statuses)

        fault_types = [
            "Arcing Contact Damage",
            "Main Contact Wear",
            "Operating Mechanism Delay",
            "Insulation Deterioration"
        ]

        for cb, status in zip(["CB-1", "CB-2", "CB-3"], breaker_statuses):
            t = np.linspace(0, 0.5, 500)
            # Base normal waveform
            base = 0.5 + 0.02 * np.sin(20 * t) + 0.005 * np.random.randn(500)

            fault_label = "Healthy"

            if status == "faulty":
                # Add large random spikes
                num_spikes = np.random.randint(5, 15)
                spike_indices = np.random.choice(500, num_spikes, replace=False)
                base[spike_indices] += np.random.uniform(0.05, 0.15, num_spikes)

                # Optional: small high-frequency noise for realism
                base += 0.01 * np.sin(50 * t)

                # Assign a random fault type
                fault_label = np.random.choice(fault_types)

            st.session_state["breaker_waveforms"][cb] = (t, base)
            st.session_state["breaker_fault_types"][cb] = fault_label

    time_data, waveform = st.session_state["breaker_waveforms"][breaker_id]
    fault_label = st.session_state["breaker_fault_types"][breaker_id]
    show_dashboard(time_data, waveform, breaker_id)
    st.sidebar.info(f"Assigned Fault Type: {fault_label}")

elif data_option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload DCRM Data", type="csv")
    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        waveform = df_upload.iloc[:,0].values
        time_data = np.linspace(0, 0.5, len(waveform))
       
        show_dashboard(time_data, waveform, breaker_id)

import numpy as np
import pandas as pd
import pickle
from scipy.stats import skew, kurtosis

# ------------------------------
# 1. Load Trained Model
# ------------------------------
with open("dcrm_logreg_model.pkl", "rb") as f:
    model = pickle.load(f)
print("Model loaded successfully!")

# ------------------------------
# 2. Generate a New Random Waveform
# ------------------------------
def generate_waveform(label='random', points=500):
    t = np.linspace(0, 0.5, points)
    base = 0.5 + 0.02*np.sin(20*t)  # baseline
    base += 0.005*np.random.randn(points)  # small noise

    wf = base.copy()

    # Add random fault pattern for demonstration
    fault_pattern = np.random.choice(['none', 'spike', 'bump', 'oscillation'])
    
    if fault_pattern == 'spike':
        start = np.random.randint(50, 400)
        length = np.random.randint(10, 20)
        wf[start:start+length] += np.random.uniform(0.1, 0.25)
    elif fault_pattern == 'bump':
        wf += 0.05*np.random.rand(points)
    elif fault_pattern == 'oscillation':
        wf += 0.03*np.sin(50*t)

    return wf

# ------------------------------
# 3. Feature Extraction
# ------------------------------
def extract_features(waveform):
    features = {}
    features['mean'] = np.mean(waveform)
    features['std'] = np.std(waveform)
    features['max'] = np.max(waveform)
    features['min'] = np.min(waveform)
    features['range'] = features['max'] - features['min']
    features['spikes'] = np.sum(waveform > 0.6)
    features['skew'] = skew(waveform)
    features['kurtosis'] = kurtosis(waveform)
    
    threshold_low = 0.52
    close_indices = np.where(waveform < threshold_low)[0]
    features['rise_time'] = (close_indices[0] if len(close_indices)>0 else len(waveform)) * (0.5 / len(waveform))
    
    return features

# ------------------------------
# 4. Generate Waveform & Extract Features
# ------------------------------
waveform = generate_waveform()
features = extract_features(waveform)
features_list = list(features.values())

# ------------------------------
# 5. Predict Healthy/Faulty
# ------------------------------
prediction = model.predict([features_list])
if prediction[0] == 0:
    print("Predicted: Healthy waveform ✅")
else:
    print("Predicted: Faulty waveform ⚠️")

# ------------------------------
# Optional: Print Features
# ------------------------------
print("\nExtracted Features:")
for k, v in features.items():
    print(f"{k}: {v:.4f}")

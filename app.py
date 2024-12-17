from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import warnings

# Flask app initialization
app = Flask(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Function to generate noisy signals with varying modulation
def generate_signal(modulation_type, num_samples=1000, snr_dB=20, carrier_freq=5, amplitude=1.0, phase_shift=0):
    # Enforce type safety for input parameters
    num_samples = int(num_samples)
    snr_dB = float(snr_dB)
    carrier_freq = float(carrier_freq)
    amplitude = float(amplitude)
    phase_shift = float(phase_shift)

    t = np.arange(num_samples) / num_samples

    if modulation_type == 'BPSK':
        data = np.random.randint(0, 2, num_samples).astype(float)
        modulated_signal = (2 * data - 1) * amplitude * np.cos(2 * np.pi * carrier_freq * t + phase_shift)

    elif modulation_type == 'QAM':
        # 16-QAM with clean I and Q components
        data = np.random.randint(0, 16, num_samples).astype(float)
        I = (2 * (data // 4) - 3).astype(float)  # Real part
        Q = (2 * (data % 4) - 3).astype(float)  # Imaginary part
        modulated_signal = amplitude * (I * np.cos(2 * np.pi * carrier_freq * t) - 
                                        Q * np.sin(2 * np.pi * carrier_freq * t))

    elif modulation_type == 'FSK':
        data = np.random.randint(0, 2, num_samples).astype(float)
        modulated_signal = amplitude * np.cos(2 * np.pi * (carrier_freq + data) * t + phase_shift)

    elif modulation_type == 'ASK':
        data = np.random.randint(0, 2, num_samples).astype(float)
        modulated_signal = amplitude * data * np.cos(2 * np.pi * carrier_freq * t + phase_shift)

    elif modulation_type == 'PSK':
        data = np.random.randint(0, 2, num_samples).astype(float)
        modulated_signal = amplitude * np.cos(2 * np.pi * carrier_freq * t + phase_shift + data * np.pi)

    else:
        raise ValueError(f"Unsupported modulation type: {modulation_type}")

    # Add Gaussian noise
    snr_linear = 10 ** (snr_dB / 10.0)
    power_signal = np.mean(np.abs(modulated_signal)**2)
    power_noise = power_signal / snr_linear
    noise = np.sqrt(power_noise) * np.random.randn(len(modulated_signal)).astype(float)
    noisy_signal = modulated_signal + noise

    return noisy_signal

# Function to extract features from a signal
def extract_features(signal):
    amplitude = np.abs(signal)
    phase = np.angle(signal)
    features = [
        np.mean(amplitude),
        np.var(amplitude),
        np.mean(phase),
        np.var(phase),
        np.mean(np.abs(np.fft.fft(signal)))  # FFT magnitude
    ]
    return features

# Dataset creation for training
num_samples = 1000
modulation_types = ['BPSK', 'QAM', 'FSK', 'ASK', 'PSK']
num_signals_per_class = 500

features = []
labels = []

for modulation in modulation_types:
    for _ in range(num_signals_per_class):
        snr_dB = np.random.uniform(5, 30)  # Random SNR
        carrier_freq = np.random.uniform(1, 10)  # Random frequency
        amplitude = np.random.uniform(0.5, 2.0)  # Random amplitude
        phase_shift = np.random.uniform(0, np.pi)  # Random phase shift
        signal = generate_signal(modulation, num_samples, snr_dB, carrier_freq, amplitude, phase_shift)
        feature = extract_features(signal)
        features.append(feature)
        labels.append(modulation)

# Convert to NumPy arrays
X = np.array(features)
y = np.array(labels)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Generate signal using the input parameters
        signal = generate_signal('BPSK', 1000, data['snr'], data['freq'], data['amplitude'], data['phase'])
        
        # Extract features
        features = extract_features(signal)
        features_scaled = scaler.transform([features])
        
        # Predict using the trained model
        prediction = clf.predict(features_scaled)[0]

        # Plot signal
        plt.figure(figsize=(5, 2))
        plt.plot(signal[:200])
        plt.title("Generated Signal")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        img_base64 = base64.b64encode(img.getvalue()).decode()

        return jsonify({'prediction': prediction, 'plot': img_base64})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

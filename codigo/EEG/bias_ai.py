from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, InputLayer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from bias import BiasClass
from scipy.signal import welch
from scipy.stats import skew, kurtosis
import pywt
import numpy as np

def main():
    n = 1000
    duration = 2
    fs = 500
    online = True
    number_of_channels = 4
    port = '/dev/serial0'
    baudrate = 115200
    timeout = 1
    biasInstance = BiasClass(n=n, fs=fs, channels=number_of_channels, port=port, baudrate=baudrate, timeout=timeout)
    biasAI = AIBias(n=n, fs=fs, channels=number_of_channels)
    biasAI.collect_and_train(biasInstance, biasInstance._commands)
    # Generate synthetic data
    synthetic_data = generate_synthetic_eeg(n_samples=n, n_channels=number_of_channels, duration=duration, fs=fs)
    
    # Wrap synthetic data in the format expected by your model
    signals = {ch: {'alpha': synthetic_data[ch], 'beta': synthetic_data[ch], 
                        'theta': synthetic_data[ch], 'delta': synthetic_data[ch], 
                        'gamma': synthetic_data[ch]} for ch in range(number_of_channels)}
    
    #signals = biasInstance._biasReception.get_real_data(channels=number_of_channels, n=n)
    filtered_data = biasInstance._biasFilter.filter_signals(signals)
    # Process data
    times, eeg_signals = biasInstance._biasProcessing.process_signals(filtered_data)
    predicted_command = biasAI.predict_command(eeg_data=eeg_signals)
    print(f"Predicted Command: {predicted_command}")


class AIBias:
    def __init__(self, n, fs, channels):
        self._n = n
        self._fs = fs
        self._number_of_channels = channels
        self._model = self.build_model()
        self._is_trained = False
        self._pca = PCA(n_components=0.95)  # Retain 95% of variance
        self._scaler = StandardScaler()


    # Define getter
    def ai_is_trained(self):
        return self._is_trained
    
    def collect_and_train(self, bias_instance, commands):
        """
        Collects EEG data, extracts features, and trains the model.
        """
        X = []
        y = []
        label_map = {"forward": 0, "backward": 1, "left": 2, "right": 3, "stop": 4, "rest": 5}

        for command in commands:
            # Get real data from the Bias instance
            signals = bias_instance._biasReception.get_real_data(channels=bias_instance._number_of_channels, n=bias_instance._n)
            filtered_data = bias_instance._biasFilter.filter_signals(signals)
            _, eeg_signals = bias_instance._biasProcessing.process_signals(filtered_data)

            # Extract features and append to X
            features = self.extract_features(eeg_signals)
            X.append(features)
            y.append(label_map[command])

        # Convert X and y to numpy arrays
        X = np.array(X)
        y = np.array(y)


        # Convert y to one-hot encoding
        lb = LabelBinarizer()
        y = lb.fit_transform(y)

        # Train the model with the collected data
        self.train_model(X, y)

    def build_model(self):
        model = Sequential([
            InputLayer(input_shape=(self._n, self._number_of_channels)),
            Conv1D(filters=64, kernel_size=3, activation='relu'),  # Adjust input_shape based on your data
            MaxPooling1D(pool_size=2),
            Dropout(0.5),
            Flatten(),
            Dense(100, activation='relu'),
            Dense(50, activation='relu'),
            Dense(6, activation='softmax')  # 6 output classes (forward, backward, etc.)
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def extract_features(self, eeg_data):
        for ch, signals in eeg_data.items():
            channel_features = []
            for band_name, sig in signals.items():
                sig = np.array(sig)

                # Statistical Features
                mean = np.mean(sig)
                variance = np.var(sig)
                skewness = skew(sig)
                kurt = kurtosis(sig)
                energy = np.sum(sig ** 2)

                # Frequency Domain Features (Power Spectral Density)
                freqs, psd = welch(sig, fs=self._fs)  # Assuming fs = 500 Hz

                # Band Power for specific frequency bands (e.g., alpha, beta, theta)
                alpha_power = np.sum(psd[(freqs >= 8) & (freqs <= 13)])
                beta_power = np.sum(psd[(freqs >= 13) & (freqs <= 30)])
                theta_power = np.sum(psd[(freqs >= 4) & (freqs <= 8)])
                delta_power = np.sum(psd[(freqs >= 0.5) & (freqs <= 4)])
                gamma_power = np.sum(psd[(freqs >= 30) & (freqs <= 100)])

                # Wavelet Transform (using the Morlet wavelet)
                coeffs, _ = pywt.cwt(sig, scales=np.arange(1, 31), wavelet='morl')
                wavelet_energy = np.sum(coeffs ** 2)

                # Append all features together
                channel_features.extend([mean, variance, skewness, kurt, energy,
                                 alpha_power, beta_power, theta_power, delta_power, gamma_power,
                                 wavelet_energy])
                
            features.append(channel_features)

        features = np.array(features)
        features = self._scaler.fit_transform(features)  # Normalize
        features = self._pca.fit_transform(features)  # Dimensionality Reduction
        features = features.reshape(self._n, self._number_of_channels, 1)

        features = np.array(features)
        return features

    def train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self._model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
        self._is_trained = True

    def predict_command(self, eeg_data):
        if not self._is_trained:
            raise Exception("Model has not been trained yet.")
        features = self.extract_features(eeg_data)
        features = features.reshape(1, -1)  # Reshape for the model input
        prediction = self._model.predict(features)
        return np.argmax(prediction, axis=1)[0]

def generate_synthetic_eeg(n_samples, n_channels, duration, fs):
    """
    Generate synthetic EEG data.
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    data = []

    for _ in range(n_channels):
        # Base random noise
        signal = np.random.normal(0, 0.5, size=t.shape)
        
        # Add sinusoidal components to simulate EEG bands
        alpha = np.sin(2 * np.pi * 10 * t)  # Alpha band (8-13 Hz)
        beta = np.sin(2 * np.pi * 20 * t)   # Beta band (13-30 Hz)
        theta = np.sin(2 * np.pi * 6 * t)   # Theta band (4-8 Hz)
        delta = np.sin(2 * np.pi * 2 * t)   # Delta band (0.5-4 Hz)
        gamma = np.sin(2 * np.pi * 40 * t)  # Gamma band (30-100 Hz)
        
        # Create different patterns for each "command"
        if np.random.rand() > 0.5:
            signal += alpha + beta
        else:
            signal += theta + delta + gamma
        
        data.append(signal)
    
    data = np.array(data)
    return data

if __name__ == "__main__":
    main()
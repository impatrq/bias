from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, InputLayer # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from bias_reception import ReceptionBias
from bias_dsp import FilterBias, ProcessingBias
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from scipy.signal import cwt, morlet
import numpy as np

def main():
    n = 1000
    fs = 500
    online = True
    number_of_channels = 4
    port = '/dev/serial0'
    baudrate = 115200
    timeout = 1
    biasReception = ReceptionBias(port, baudrate, timeout)
    biasFilter = FilterBias(n=n, fs=fs, notch=True, bandpass=True, fir=False, iir=False)
    biasProcessing = ProcessingBias(n=n, fs=fs)
    commands = ["forward", "backward", "left", "right", "stop", "rest"]
    biasAI = AIBias(n=n, fs=fs, channels=number_of_channels)
    biasAI.collect_and_train(reception_instance=biasReception, filter_instance=biasFilter, processing_instance=biasProcessing, 
                             commands=commands, real_data=False)
    # Generate synthetic data
    signals = generate_synthetic_eeg(n_samples=n, n_channels=number_of_channels, fs=fs)
    #signals = biasReception.get_real_data(channels=number_of_channels, n=n)
    
    filtered_data = biasFilter.filter_signals(signals)
    # Process data
    times, eeg_signals = biasProcessing.process_signals(filtered_data)
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
    
    def collect_and_train(self, reception_instance, filter_instance, processing_instance, commands, real_data=True):
        """
        Collects EEG data, extracts features, and trains the model.
        """
        X = []
        y = []
        label_map = {"forward": 0, "backward": 1, "left": 2, "right": 3, "stop": 4, "rest": 5}

        for command in commands:
            # Get real data from the Bias instance
            if real_data:
                signals = reception_instance.get_real_data(channels=self._number_of_channels, n=self._n)
            else:
                signals = generate_synthetic_eeg(n_samples=self._n, n_channels=self._number_of_channels, fs=self._fs)
            filtered_data = filter_instance.filter_signals(signals)
            _, eeg_signals = processing_instance.process_signals(filtered_data)

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
            InputLayer(shape=(self._number_of_channels, 55)),  # Adjusted input shape to match the feature count
            Conv1D(filters=64, kernel_size=3, activation='relu'),
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
        features = []
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

                # Use scipy.signal.cwt instead of pywt
                scales = np.arange(1, 31)
                coeffs = cwt(sig, morlet, scales)
                wavelet_energy = np.sum(coeffs ** 2)

                # Append all features together
                channel_features.extend([mean, variance, skewness, kurt, energy,
                                 alpha_power, beta_power, theta_power, delta_power, gamma_power,
                                 wavelet_energy])
                
            features.append(channel_features)

        features = np.abs(np.array(features))
        features = self._scaler.fit_transform(features)  # Normalize
        # Perform PCA if needed, currently commented out
        # features = self._pca.fit_transform(features)  # Dimensionality Reduction

        # Adjust reshaping based on actual size
        # Get the total number of features per channel
        num_features_per_channel = features.shape[1]

        # Reshape based on the number of samples, channels, and features
        expected_shape = (self._number_of_channels, num_features_per_channel, 1)
        features = features.reshape(expected_shape)
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

def generate_synthetic_eeg(n_samples, n_channels, fs):
    """
    Generate synthetic raw EEG data for multiple channels. 
    The output is a dictionary where each channel has 1000 raw samples.
    """
    t = np.linspace(0, n_samples/fs, n_samples, endpoint=False)
    data = {}

    for ch in range(n_channels):
        # Create a raw EEG signal by summing several sine waves to simulate brain activity
        signal = (
            np.sin(2 * np.pi * 10 * t) +  # Simulate alpha wave (8-13 Hz)
            np.sin(2 * np.pi * 20 * t) +  # Simulate beta wave (13-30 Hz)
            np.sin(2 * np.pi * 6 * t) +   # Simulate theta wave (4-8 Hz)
            np.sin(2 * np.pi * 2 * t) +   # Simulate delta wave (0.5-4 Hz)
            np.sin(2 * np.pi * 40 * t)    # Simulate gamma wave (30-100 Hz)
        )

        # Add random noise to simulate realistic EEG signals
        noise = np.random.normal(0, 0.5, size=t.shape)
        signal += noise

        # Store the raw signal in the dictionary
        data[ch] = signal

    return data

if __name__ == "__main__":
    main()
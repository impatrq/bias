from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, InputLayer, BatchNormalization # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from bias_reception import ReceptionBias
from bias_dsp import FilterBias, ProcessingBias
from scipy.signal import welch
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import cwt, morlet
import numpy as np
import random
import time

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
    commands = ["forward", "backwards", "left", "right", "stop", "rest"]
    biasAI = AIBias(n=n, fs=fs, channels=number_of_channels, commands=commands)
    train = input("Do you want to train model? (y/n): ")
    if train.lower() == "y":
        saved_dataset_path = None
        save_path = None
        loading_dataset = input("Do you want to load a existent dataset? (y/n): ")
        if loading_dataset.lower() == "y":
            saved_dataset_path = input("Write the name of the file where dataset was saved: ")
        else:
            save_new_dataset = input("Do you want to save the new dataset? (y/n): ")
            if save_new_dataset == "y":
                save_path = input("Write the path where you want to save the dataset: ")
        biasAI.collect_and_train(reception_instance=biasReception, filter_instance=biasFilter, processing_instance=biasProcessing, 
                                 trials_per_command=1, save_path=save_path, saved_dataset_path=saved_dataset_path, real_data=False)
    # Generate synthetic data
    signals = generate_synthetic_eeg(n_samples=n, n_channels=number_of_channels, fs=fs, command="left")
    #signals = biasReception.get_real_data(channels=number_of_channels, n=n)
    
    filtered_data = biasFilter.filter_signals(signals)
    # Process data
    times, eeg_signals = biasProcessing.process_signals(filtered_data)
    predicted_command = biasAI.predict_command(eeg_data=eeg_signals)
    print(f"Predicted Command: {predicted_command}")


class AIBias:
    def __init__(self, n, fs, channels, commands):
        self._n = n
        self._fs = fs
        self._number_of_channels = channels
        self._features_length = len(["mean", "variance", "skewness", "kurt", "energy",
                                 "band_power", "wavelet_energy", "entropy"])
        self._number_of_waves_per_channel = len(["alpha", "beta", "gamma", "delta", "theta"])
        self._num_features_per_channel = self._features_length * self._number_of_waves_per_channel
        self._commands = commands
        self._model = self.build_model(output_dimension=len(self._commands))
        self._is_trained = False
        self._pca = PCA(n_components=0.95)  # Retain 95% of variance
        self._scaler = StandardScaler()

        # Create a dynamic label map based on the provided commands
        self._label_map = {command: idx for idx, command in enumerate(self._commands)}
        self._reverse_label_map = {idx: command for command, idx in self._label_map.items()}

    # Define getter
    def ai_is_trained(self):
        return self._is_trained
    
    def collect_and_train(self, reception_instance, filter_instance, processing_instance, trials_per_command, 
                          save_path=None, saved_dataset_path=None, real_data=True):
        """
        Collects EEG data, extracts features, and trains the model.
        """
        X = []
        y = []

        if saved_dataset_path is None:
            for trial in range(trials_per_command):
                for command in self._commands:
                    # Get real data or generate synthetic data
                    if real_data:
                        print(f"Think about {command}. Trial: {trial}")
                        signals = reception_instance.get_real_data(channels=self._number_of_channels, n=self._n)
                    else:
                        print(f"Trial: {trial}")
                        signals = generate_synthetic_eeg(n_samples=self._n, n_channels=self._number_of_channels, fs=self._fs, command=command)
                    
                    filtered_data = filter_instance.filter_signals(signals)
                    _, eeg_signals = processing_instance.process_signals(filtered_data)

                    # Extract features and append to X
                    features = self.extract_features(eeg_signals)
                    
                    X.append(features)
                    y.append(self._label_map[command])

                    if real_data:
                        time.sleep(1)
                
                if real_data:
                    print("Changing command. Be ready")
                    time.sleep(20)

            # Convert X and y to numpy arrays
            X = np.array(X)
            y = np.array(y)

            if save_path:
                # Save the dataset as a compressed NumPy file
                np.savez_compressed(f"{save_path}.npz", X=X, y=y)
                print(f"Dataset saved to {save_path}.npz")
        
        else:
            data = np.load(f"{saved_dataset_path}.npz")
            X, y = data['X'], data['y']

        # Convert y to one-hot encoding
        lb = LabelBinarizer()
        y = lb.fit_transform(y)

        # Train the model with the collected data
        self.train_model(X, y)

    def build_model(self, output_dimension):
        model = Sequential([
            InputLayer(shape=(self._number_of_channels, self._num_features_per_channel)),  # Adjusted input shape to match the feature count
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            #BatchNormalization(),
            MaxPooling1D(pool_size=1),
            Dropout(0.5),
            Flatten(),
            Dense(100, activation='relu'), #, kernel_regularizer=l2(0.01)),
            #BatchNormalization(),
            Dropout(0.5),
            Dense(50, activation='relu'), #, kernel_regularizer=l2(0.01)),
            #BatchNormalization(),
            Dropout(0.5),
            Dense(output_dimension, activation='softmax')  # 6 output classes (forward, backward, etc.)
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def extract_features(self, eeg_data):
        features = []
        # Iterate over each channel in eeg_data
        for ch, signals_per_channel in eeg_data.items():
            channel_features = []
            assert(len(signals_per_channel) == self._number_of_waves_per_channel)
            # Iterate over the signals of each channel
            for band_name, signal_wave in signals_per_channel.items():
                signal_wave = np.array(signal_wave)

                # Statistical Features
                mean = np.mean(signal_wave)
                variance = np.var(signal_wave)
                skewness = skew(signal_wave)
                kurt = kurtosis(signal_wave)
                energy = np.sum(signal_wave ** 2)

                # Frequency Domain Features (Power Spectral Density)
                freqs, psd = welch(signal_wave, fs=self._fs)  # Assuming fs = 500 Hz

                # Band Power
                band_power = np.sum(psd)  # Total power within this band

                # Use scipy.signal.cwt instead of pywt
                scales = np.arange(1, 31)
                coeffs = cwt(signal_wave, morlet, scales)
                wavelet_energy = np.sum(coeffs ** 2)

                # Entropy
                signal_entropy = entropy(np.histogram(signal_wave, bins=10)[0])
                list_of_features = [mean, variance, skewness, kurt, energy, band_power, wavelet_energy, signal_entropy]

                # Append all features together
                channel_features.extend(list_of_features)
                
                assert(len(list_of_features) == self._features_length)
                
            features.append(channel_features)

        features = np.abs(np.array(features))
        features = self._scaler.fit_transform(features)  # Normalize
        # Perform PCA if needed, currently commented out
        # features = self._pca.fit_transform(features)  # Dimensionality Reduction

        # Adjust reshaping based on actual size
        # Get the total number of features per channel
        num_features_per_channel = features.shape[1]
        assert(self._num_features_per_channel == num_features_per_channel)
        # Reshape based on the number of samples, channels, and features
        expected_shape = (self._number_of_channels, self._num_features_per_channel)
        features = features.reshape(expected_shape)
        return features

    def train_model(self, X, y):
        # X_processed = np.array([self.extract_features(epoch) for epoch in X])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self._model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
        self._is_trained = True

    def predict_command(self, eeg_data):
        if not self._is_trained:
            raise Exception("Model has not been trained yet.")
        
        # Extract features from the EEG data
        features = self.extract_features(eeg_data)
        
        # Ensure the features have the correct shape (1, number_of_channels, number_of_features)
        features = features.reshape(1, self._number_of_channels, self._num_features_per_channel)
        
        # Make prediction
        prediction = self._model.predict(features)
        
        # Get the predicted label index
        predicted_label_index = np.argmax(prediction, axis=1)[0]
        
        # Convert the numerical prediction to the text label
        predicted_command = self._reverse_label_map[predicted_label_index]
        
        return predicted_command

def generate_synthetic_eeg(n_samples, n_channels, fs, command=None):
    """
    Generate synthetic raw EEG data for multiple channels.
    The output is a dictionary where each channel has 1000 raw samples.
    Simulate different tasks by altering the signal patterns.
    """
    t = np.linspace(0, n_samples/fs, n_samples, endpoint=False)
    data = {}

    for ch in range(n_channels):
        # Simulate different frequency bands with some basic correlations
        base_alpha = np.sin(2 * np.pi * 10 * t)  # Alpha signal_wave (10 Hz)
        base_beta = np.sin(2 * np.pi * 20 * t)   # Beta signal_wave (20 Hz)
        base_theta = np.sin(2 * np.pi * 6 * t)   # Theta signal_wave (6 Hz)
        base_delta = np.sin(2 * np.pi * 2 * t)   # Delta signal_wave (2 Hz)
        base_gamma = np.sin(2 * np.pi * 40 * t)  # Gamma signal_wave (40 Hz)

        alpha_power = 1.0
        beta_power = 1.0
        theta_power = 1.0
        delta_power = 1.0
        gamma_power = 1.0 # Adjust signal based on the command

        if command == "forward":
            alpha_power = 1.5
            beta_power = 0.5
        elif command == "backward":
            alpha_power = 0.5
            beta_power = 1.5
        elif command == "left":
            theta_power = 1.5
            delta_power = 0.5
        elif command == "right":
            theta_power = 0.5
            delta_power = 1.5
        elif command == "stop":
            alpha_power = 0.2
            beta_power = 0.2
            gamma_power = 0.2
        else:  # rest
            alpha_power = 1.0
            beta_power = 1.0
            theta_power = 1.0
            delta_power = 1.0
            gamma_power = 1.0        
        
        # Generate signal with some added randomness and correlations
        signal = (
            alpha_power * base_alpha +
            beta_power * base_beta +
            theta_power * base_theta +
            delta_power * base_delta +
            gamma_power * base_gamma
        )

        # Add channel correlation (e.g., 10% of the previous channel’s signal)
        if ch > 0:
            signal += 0.1 * data[ch-1]

        # Add random noise to simulate realistic EEG signals
        noise = np.random.normal(0, 0.1, size=t.shape)
        signal += noise

        # Store the raw signal in the dictionary
        data[ch] = signal

    return data

'''
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
            random.randrange(0, 10) * np.sin(2 * np.pi * random.randrange(8, 13) * t) +  # Simulate alpha signal_wave (8-13 Hz)
            random.randrange(0, 10) * np.sin(2 * np.pi * random.randrange(13, 30) * t) +  # Simulate beta signal_wave (13-30 Hz)
            random.randrange(0, 10) * np.sin(2 * np.pi * random.randrange(4, 8) * t) +   # Simulate theta signal_wave (4-8 Hz)
            random.randrange(0, 10) * np.sin(2 * np.pi * random.randrange(1, 4) * t) +   # Simulate delta signal_wave (0.5-4 Hz)
            random.randrange(0, 10) * np.sin(2 * np.pi * random.randrange(0, 50) * t)    # Simulate gamma signal_wave (30-100 Hz)
        )

        # Add random noise to simulate realistic EEG signals
        noise = np.random.normal(0, 0.5, size=t.shape)
        signal += noise

        # Store the raw signal in the dictionary
        data[ch] = signal

    return data
'''

if __name__ == "__main__":
    main()
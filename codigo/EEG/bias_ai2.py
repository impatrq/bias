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
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws

'''
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
                                 samples_per_command=1, save_path=save_path, saved_dataset_path=saved_dataset_path, real_data=False)
    # Generate synthetic data
    signals = generate_synthetic_eeg(n_samples=n, n_channels=number_of_channels, fs=fs, command="left")
    #signals = biasReception.get_real_data(channels=number_of_channels, n=n)
    
    filtered_data = biasFilter.filter_signals(signals)
    # Process data
    times, eeg_signals = biasProcessing.process_signals(filtered_data)
    predicted_command = biasAI.predict_command(eeg_data=eeg_signals)
    print(f"Predicted Command: {predicted_command}")
'''

def main():
    biasFilter = FilterBias(n=n, fs=fs, notch=True, bandpass=True, fir=False, iir=False)
    biasProcessing = ProcessingBias(n=n, fs=fs)

    # Load EEGBCI motor imagery dataset from PhysioNet
    subject = 1
    runs = [3, 4, 7, 8, 11, 12]  # Motor imagery tasks
    raw = load_and_preprocess_data(subject, runs)

    # Filter data for motor imagery analysis
    raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')

    # Extract events (motor imagery tasks)
    events, event_id = mne.events_from_annotations(raw)

    # Epoch the data
    tmin, tmax = -1., 4.
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, baseline=None, preload=True)

    # Get data and labels
    X = epochs.get_data()  # EEG data
    y = epochs.events[:, -1]  # Labels

    # Preprocess and train the model
    biasAI = AIBias(X.shape[2], epochs.info['sfreq'], X.shape[1], commands=list(event_id.keys()))
    Signals = {}

    # Iterate over each channel in eeg_data
    for ch in range(X.shape[0]):  # Assuming eeg_data is [channels, samples]
        signal_wave = X[ch, :]  # Select data for the channel
        Signals[ch] = signal_wave

    filtered_signals = biasFilter.filter_signals(Signals)
    processed_signals = biasProcessing.process_signals(filtered_signals)

    # Convert y to one-hot encoding
    lb = LabelBinarizer()
    y = lb.fit_transform(y)
    biasAI.train_model(processed_signals, y)

def load_and_preprocess_data(subject, runs):
    """
    Load and preprocess EEG data from PhysioNet BCI dataset.
    """
    # Download data if necessary
    raw_fnames = eegbci.load_data(subject, runs)
    raw = concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in raw_fnames])
    
    # Set the EEG channel types
    raw.pick_types(eeg=True, stim=False, exclude='bads')
    raw.set_eeg_reference('average', projection=True)
    return raw


class AIBias:
    def __init__(self, n, fs, channels, commands):
        self._n = n
        self._fs = fs
        self._number_of_channels = channels
        self._features_length = len(["mean", "variance", "skewness", "kurt", "energy",
                                 "band_power", "wavelet_energy", "entropy"])
        self._number_of_waves_per_channel = len(["alpha", "beta", "gamma", "delta", "theta"])
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
    
    def collect_and_train(self, reception_instance, filter_instance, processing_instance, samples_per_command, 
                          save_path=None, saved_dataset_path=None, real_data=True):
        """
        Collects EEG data, extracts features, and trains the model.
        """
        X = []
        y = []

        if saved_dataset_path is None:
            for command in self._commands:
                for sample in range(samples_per_command):
                    # Get real data or generate synthetic data
                    if real_data:
                        print(f"Think about {command}. Sample: {sample}")
                        signals = reception_instance.get_real_data(channels=self._number_of_channels, n=self._n)
                    else:
                        print(f"Sample: {sample}")
                        signals = generate_synthetic_eeg(n_samples=self._n, n_channels=self._number_of_channels, fs=self._fs) #, command=command)
                    
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
            InputLayer(shape=(self._number_of_channels, self._features_length * self._number_of_waves_per_channel)),  # Adjusted input shape to match the feature count
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            #BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.5),
            Flatten(),
            Dense(100, activation='relu'), #, kernel_regularizer=l2(0.01)),
            #BatchNormalization(),
            #Dropout(0.5),
            Dense(50, activation='relu'), #, kernel_regularizer=l2(0.01)),
            #BatchNormalization(),
            #Dropout(0.5),
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

        # Reshape based on the number of samples, channels, and features
        #expected_shape = (self._number_of_channels, num_features_per_channel, 1)
        #features = features.reshape(expected_shape)
        features = features.reshape(self._number_of_channels, -1, 1)  # Adjust as needed
        return features

    def train_model(self, X, y):
        X_processed = np.array([self.extract_features(epoch) for epoch in X])
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
        self._model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
        self._is_trained = True

    def predict_command(self, eeg_data):
        if not self._is_trained:
            raise Exception("Model has not been trained yet.")
        
        # Extract features from the EEG data
        features = self.extract_features(eeg_data)
        
        # Ensure the features have the correct shape (1, number_of_channels, number_of_features)
        features = features.reshape(1, self._number_of_channels, -1)
        
        # Make prediction
        prediction = self._model.predict(features)
        
        # Get the predicted label index
        predicted_label_index = np.argmax(prediction, axis=1)[0]
        
        # Convert the numerical prediction to the text label
        predicted_command = self._reverse_label_map[predicted_label_index]
        
        return predicted_command


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


if __name__ == "__main__":
    main()
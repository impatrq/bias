from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, InputLayer, BatchNormalization # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from bias_reception import ReceptionBias
from bias_dsp import FilterBias, ProcessingBias
from scipy.signal import welch
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import cwt, morlet
from sklearn.metrics import confusion_matrix
from mne.decoding import CSP
import numpy as np
import time
from signals import generate_synthetic_eeg, generate_synthetic_eeg_bandpower

def main():
    n = 1000
    fs = 500
    number_of_channels = 4
    port = '/dev/serial0'
    baudrate = 115200
    timeout = 1
    biasReception = ReceptionBias(port, baudrate, timeout)
    biasFilter = FilterBias(n=n, fs=fs, notch=True, bandpass=True, fir=False, iir=False)
    biasProcessing = ProcessingBias(n=n, fs=fs)
    commands = ["forward", "backwards", "left", "right"] #, "stop", "rest"]
    biasAI = AIBias(n=n, fs=fs, channels=number_of_channels, commands=commands)

    model_lt = input("Do you want to load or train a model (l/t): ")
    if model_lt.lower() == "t":
        # Get the user's input desires
        save_path = None
        saved_dataset_path = None

        training_real_data = False
        loading_dataset = input("Do you want to load an existent dataset? (y/n): ")
        if loading_dataset.lower() == "y":
            saved_dataset_path = input("Write the name of the file where dataset was saved: ")
        else:
            # Generate data
            want_real_data = input("Do you want to train it with real data? (y/n): ")

            if want_real_data.lower().strip() == "y":
                training_real_data = True
            else:
                training_real_data = False

            save_new_dataset = input("Do you want to save the new dataset? (y/n): ")
            if save_new_dataset == "y":
                save_path = input("Write the path where you want to save the dataset: ")

        biasAI.collect_and_train(reception_instance=biasReception, filter_instance=biasFilter, processing_instance=biasProcessing, 
                            trials_per_command=1, save_path=save_path, saved_dataset_path=saved_dataset_path, training_real_data=training_real_data)

    # Load an existent model
    elif model_lt.lower() == 'l':
        model_name = input("Write the filname where model is saved: ")
        print("Charging model")

    # Generate synthetic data
    signals = generate_synthetic_eeg_bandpower(n_samples=n, n_channels=number_of_channels, fs=fs, command="left")
    #signals = biasReception.get_real_data(n=n, channels=number_of_channels)

    filtered_data = biasFilter.filter_signals(eeg_signals=signals)
    # Process data
    times, eeg_signals = biasProcessing.process_signals(eeg_signals=filtered_data)
    predicted_command = biasAI.predict_command(eeg_data=eeg_signals)
    print(f"Predicted Command: {predicted_command}")


class AIBias:
    def __init__(self, n, fs, channels, commands):
        self._n = n
        self._fs = fs
        self._number_of_channels = channels
        self._features_length = len(["mean", "variance", "skewness", "kurt", "energy",
                                 "band_power", "wavelet_energy", "entropy"])
        self._number_of_waves_per_channel = len(["signal", "alpha", "beta", "gamma", "delta", "theta"])
        # Add features for the whole signal
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
                          save_path=None, saved_dataset_path=None, training_real_data=True):
        """
        Collects EEG data, extracts features, and trains the model.
        """
        X = []
        y = []

        if saved_dataset_path is None:
            for trial in range(trials_per_command):
                for command in self._commands:
                    # Get real data or generate synthetic data
                    if training_real_data:
                        print(f"Think about {command}. Trial: {trial}")
                        signals = reception_instance.get_real_data(n=self._n, channels=self._number_of_channels)
                    else:
                        print(f"Trial: {trial}")
                        #signals = generate_synthetic_eeg(n_samples=self._n, n_channels=self._number_of_channels, fs=self._fs)
                        signals = generate_synthetic_eeg_bandpower(n_samples=self._n, n_channels=self._number_of_channels, fs=self._fs, command=command)

                    filtered_data = filter_instance.filter_signals(signals)
                    _, eeg_signals = processing_instance.process_signals(filtered_data)

                    # Extract features and append to X
                    features = self.extract_features(eeg_signals)

                    X.append(features)
                    y.append(self._label_map[command])

                    if training_real_data:
                        time.sleep(1)

                if training_real_data:
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

        # Convert labels to one-hot encoded format using OneHotEncoder
        one_hot_encoder = OneHotEncoder(sparse_output=False)
        y_one_hot = one_hot_encoder.fit_transform(y.reshape(-1, 1))
        print(f"y_one_hot_shape: {y_one_hot.shape}")

        unique_classes, counts = np.unique(y_one_hot, return_counts=True)
        print(f"Classes in dataset: {unique_classes}, Counts: {counts}")
        
        # Train the model with the collected data
        self.train_model(X, y_one_hot)
        
        print("Training complete.")     

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

        # Get the total number of features per channel
        num_features_per_channel = features.shape[1]
        assert(self._num_features_per_channel == num_features_per_channel)
        # Reshape based on the number of samples, channels, and features
        expected_shape = (self._number_of_channels, self._num_features_per_channel)
        features = features.reshape(expected_shape)
        return features

    def train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self._model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
        self.model_evaluation(X_test, y_test)
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

    def model_evaluation(self, X_test, y_test):
        # Evaluate model performance on the test set
        loss, accuracy = self._model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {accuracy}")

        # Get model predictions on the test set
        y_pred = self._model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)  # Convert one-hot encoding to class labels

        # Confusion matrix
        cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred_classes)

        print(cm)

if __name__ == "__main__":
    main()

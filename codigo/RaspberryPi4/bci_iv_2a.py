from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, InputLayer
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.decomposition import PCA
from scipy.signal import welch, cwt, morlet
from scipy.stats import skew, kurtosis, entropy
import numpy as np
import random
import time

# Import MotorImageryDataset class from your dataset code.
class MotorImageryDataset:
    def __init__(self, dataset='A01T.npz'):
        if not dataset.endswith('.npz'):
            dataset += '.npz'
        self.data = np.load(dataset)
        self.Fs = 250  # 250Hz from original paper
        self.raw = self.data['s'].T
        self.events_type = self.data['etyp'].T
        self.events_position = self.data['epos'].T
        self.events_duration = self.data['edur'].T
        self.artifacts = self.data['artifacts'].T
        self.mi_types = {769: 'left', 770: 'right', 771: 'foot', 772: 'tongue', 783: 'unknown'}

    def get_trials_from_channel(self, channel=7):
        starttrial_code = 768
        starttrial_events = self.events_type == starttrial_code
        idxs = [i for i, x in enumerate(starttrial_events[0]) if x]
        trials, classes = [], []
        for index in idxs:
            try:
                type_e = self.events_type[0, index + 1]
                class_e = self.mi_types[type_e]
                classes.append(class_e)
                start = self.events_position[0, index]
                stop = start + self.events_duration[0, index]
                trial = self.raw[channel, start:stop].reshape((1, -1))
                trials.append(trial)
            except:
                continue
        return trials, classes

    def get_trials_from_channels(self, channels=[7, 9, 11]):
        trials_c, classes_c = [], []
        for c in channels:
            t, c = self.get_trials_from_channel(channel=c)
            tt = np.concatenate(t, axis=0)
            trials_c.append(tt)
            classes_c.append(c)
        return trials_c, classes_c


class AIBias:
    def __init__(self, n, fs, channels, commands):
        self._n = n
        self._fs = fs
        self._number_of_channels = channels
        self._features_length = len(["mean", "variance", "skewness", "kurt", "energy", "band_power", "wavelet_energy", "entropy"])
        self._number_of_waves_per_channel = len(["alpha", "beta", "gamma", "delta", "theta"])
        self._commands = commands
        self._model = self.build_model(output_dimension=len(self._commands))
        self._is_trained = False
        self._pca = PCA(n_components=0.95)
        self._scaler = StandardScaler()
        self._label_map = {command: idx for idx, command in enumerate(self._commands)}
        self._reverse_label_map = {idx: command for command, idx in self._label_map.items()}

    def ai_is_trained(self):
        return self._is_trained

    def build_model(self, output_dimension):
        model = Sequential([
            InputLayer(shape=(self._number_of_channels, self._features_length * self._number_of_waves_per_channel)),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.5),
            Flatten(),
            Dense(100, activation='relu'),
            Dropout(0.5),
            Dense(50, activation='relu'),
            Dropout(0.5),
            Dense(output_dimension, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def extract_features(self, eeg_data):
        features = []
        for ch, signals_per_channel in eeg_data.items():
            channel_features = []
            for band_name, signal_wave in signals_per_channel.items():
                signal_wave = np.array(signal_wave)
                mean = np.mean(signal_wave)
                variance = np.var(signal_wave)
                skewness = skew(signal_wave)
                kurt = kurtosis(signal_wave)
                energy = np.sum(signal_wave ** 2)
                freqs, psd = welch(signal_wave, fs=self._fs)
                band_power = np.sum(psd)
                scales = np.arange(1, 31)
                coeffs = cwt(signal_wave, morlet, scales)
                wavelet_energy = np.sum(coeffs ** 2)
                signal_entropy = entropy(np.histogram(signal_wave, bins=10)[0])
                list_of_features = [mean, variance, skewness, kurt, energy, band_power, wavelet_energy, signal_entropy]
                channel_features.extend(list_of_features)
            features.append(channel_features)
        features = np.abs(np.array(features))
        features = self._scaler.fit_transform(features)
        num_features_per_channel = features.shape[1]
        features = features.reshape((self._number_of_channels, num_features_per_channel, 1))
        return features

    def train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self._model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
        self._is_trained = True

    def predict_command(self, eeg_data):
        if not self._is_trained:
            raise Exception("Model has not been trained yet.")
        features = self.extract_features(eeg_data)
        features = features.reshape(1, self._number_of_channels, -1)
        prediction = self._model.predict(features)
        predicted_label_index = np.argmax(prediction, axis=1)[0]
        predicted_command = self._reverse_label_map[predicted_label_index]
        return predicted_command

def load_and_train_from_bci_dataset():
    # Load BCI dataset
    datasetA1 = MotorImageryDataset("A01T.npz")
    trials, classes = datasetA1.get_trials_from_channels([7, 9, 11])  # Example: C3, Cz, C4

    # Flatten the trials for each channel and convert classes into the desired format
    X, y = [], []
    command_map = {"left": "left", "right": "right", "foot": "forward", "tongue": "backwards"}

    ai_bias = AIBias(n=1000, fs=250, channels=3, commands=["left", "right", "forward", "backwards", "stop", "rest"])

    for i in range(len(trials)):
        trial_data = trials[i]
        class_label = classes[i][0]  # Assuming the first label for simplicity
        if class_label in command_map:
            # Process the EEG data and extract features
            eeg_data = {
                ch: {wave: trial_data for wave in ["alpha", "beta", "gamma", "delta", "theta"]}
                for ch in range(3)  # Assuming 3 channels
            }
            features = ai_bias.extract_features(eeg_data)
            X.append(features)
            y.append(ai_bias._label_map[command_map[class_label]])

    # Convert lists to arrays for training
    X = np.array(X)
    y = np.array(y)

    # Convert labels to one-hot encoded format
    lb = LabelBinarizer()
    y = lb.fit_transform(y)

    # Train the model
    ai_bias.train_model(X, y)
    print("Training complete.")

if __name__ == "__main__":
    load_and_train_from_bci_dataset()
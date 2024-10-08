from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, InputLayer, LSTM
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.decomposition import PCA
from scipy.signal import welch, cwt, morlet
from scipy.stats import skew, kurtosis, entropy
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from bias_dsp import ProcessingBias, FilterBias
import plotext as plt
import random
import time

def main():
    n = 1875
    fs = 250
    online = True
    number_of_channels = 4
    biasFilter = FilterBias(n=n, fs=fs, notch=True, bandpass=True, fir=False, iir=False)
    biasProcessing = ProcessingBias(n=n, fs=fs)
    commands = ["forward", "backwards", "left", "right"] #, "stop", "rest"]
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
        biasAI.collect_and_train_from_bci_dataset(filter_instance=biasFilter, processing_instance=biasProcessing, save_path=save_path, 
                                                  saved_dataset_path=saved_dataset_path)
    biasAI.make_predictions(filter_instance=biasFilter, processing_instance=biasProcessing)

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

    def get_trials_from_channels(self, channels=[0, 7, 9, 11]):
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
        self._number_of_waves_per_channel = len(["signal", "alpha", "beta", "gamma", "delta", "theta"])
        self._num_features_per_channel = self._features_length * self._number_of_waves_per_channel
        self._commands = commands
        self._model = self.build_model(output_dimension=len(self._commands))
        self._is_trained = False
        self._pca = PCA(n_components=0.95)
        self._scaler = StandardScaler()
        self._label_map = {command: idx for idx, command in enumerate(self._commands)}
        self._reverse_label_map = {idx: command for command, idx in self._label_map.items()}
        self._command_map = {"left": "left", "right": "right", "foot": "forward", "tongue": "backwards"}

    def ai_is_trained(self):
        return self._is_trained

    def build_model(self, output_dimension):
        model = Sequential([
            InputLayer(shape=(self._number_of_channels, self._num_features_per_channel)),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=1),
            Dropout(0.5),
            #Flatten(),
            LSTM(50, return_sequences=False),
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
        assert(self._num_features_per_channel == num_features_per_channel)
        features = features.reshape((self._number_of_channels, self._num_features_per_channel))
        return features

    def train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self._model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
        self.model_evaluation()
        self._is_trained = True

    def predict_command(self, eeg_data):
        if not self._is_trained:
            raise Exception("Model has not been trained yet.")
        features = self.extract_features(eeg_data)
        features = features.reshape(1, self._number_of_channels, self._num_features_per_channel)
        prediction = self._model.predict(features)
        predicted_label_index = np.argmax(prediction, axis=1)[0]
        predicted_command = self._reverse_label_map[predicted_label_index]
        return predicted_command
    
    def load_datasets(self, file_names):
        all_trials = []
        all_classes = []
        for file_name in file_names:
            dataset = MotorImageryDataset(file_name)
            trials, classes = dataset.get_trials_from_channels([0, 7, 9, 11])
            all_trials.extend(trials)
            all_classes.extend(classes)

        # Invert the dimensions of trials and classes using zip
        inverted_trials = list(map(list, zip(*all_trials)))
        inverted_classes = list(map(list, zip(*all_classes)))
        return inverted_trials, inverted_classes

    def make_predictions(self, filter_instance, processing_instance):
        file_list = [f"bcidatasetIV2a-master/A09T.npz"]
        trials, classes = self.load_datasets(file_list)

        for num_trial in range(len(trials)):
            label = classes[num_trial][0]
            print(f"Label: {label}")
            if label in self._command_map.keys():
                # Create a dictionary to hold the EEG signals for each channel
                eeg_signals = {f"ch{ch}": trials[num_trial][ch].tolist() for ch in range(self._number_of_channels)}  # Assuming 3 channels: C3, Cz, C4

                filtered_data = filter_instance.filter_signals(eeg_signals)
                # Process the raw EEG signals using ProcessingBias to extract frequency bands
                _, processed_signals = processing_instance.process_signals(filtered_data)

                predicted_command = self.predict_command(processed_signals)
                command = self._command_map[label]
                if predicted_command == command:
                    print(f"Prediction ok. Command: {command}")

                else:
                    print(f"Wrong prediction. Predicted {predicted_command}. Actual command: {command}")
            else:
                 print("Label not in command_map")

    def model_evaluation(self, X_test, y_test):
        # Evaluate model performance on the test set
        loss, accuracy = self._model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {accuracy}")

        # Get model predictions on the test set
        y_pred = self._model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)  # Convert one-hot encoding to class labels

        # Confusion matrix
        cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred_classes)

        # Plot the confusion matrix
        sns.heatmap(cm, annot=True, fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('True')

    def collect_and_train_from_bci_dataset(self, filter_instance, processing_instance, save_path, saved_dataset_path):
        # Initialize X and y as empty lists
        X = []
        y = []

        if saved_dataset_path is None:
            file_list = [f"bcidatasetIV2a-master/A0{i}T.npz" for i in range(1, 9)]
            trials, classes = self.load_datasets(file_list)
            for num_trial in range(len(trials)):
                label = classes[num_trial][0]
                if label in self._command_map.keys():
                    # Create a dictionary to hold the EEG signals for each channel
                    eeg_signals = {f"ch{ch}": trials[num_trial][ch].tolist() for ch in range(self._number_of_channels)}  # Assuming 4 channels: C3, Cz, C4

                    filtered_data = filter_instance.filter_signals(eeg_signals)
                    # Process the raw EEG signals using ProcessingBias to extract frequency bands
                    _, processed_signals = processing_instance.process_signals(filtered_data)

                    # Extract features from the processed signals (frequency bands)
                    features = self.extract_features(processed_signals)

                    # Append the extracted features and the corresponding command label
                    X.append(features)
                    y.append(self._label_map[self._command_map[label]])

            if save_path:
                # Save the dataset as a compressed NumPy file
                np.savez_compressed(f"{save_path}.npz", X=X, y=y)
                print(f"Dataset saved to {save_path}.npz")

        else:
            data = np.load(f"{saved_dataset_path}.npz")
            X, y = data['X'], data['y']

        # Convert lists to arrays for training
        X = np.array(X)
        y = np.array(y)

        # Convert labels to one-hot encoded format
        lb = LabelBinarizer()
        y = lb.fit_transform(y)

        # Train the model
        self.train_model(X, y)

        print("Training complete.")

if __name__ == "__main__":
    main()

'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, InputLayer, LSTM
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.decomposition import PCA
from scipy.signal import welch, cwt, morlet
from scipy.stats import skew, kurtosis, entropy
import numpy as np
from bias_dsp import ProcessingBias, FilterBias
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

    def get_trials_from_channels(self, channels=[0, 7, 9, 11]):
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
        self._num_features_per_channel = self._features_length * self._number_of_waves_per_channel
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
            InputLayer(shape=(self._number_of_channels, self._num_features_per_channel)),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=1),
            Dropout(0.5),
            # Flatten(),
            LSTM(50, return_sequences=False),
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
        assert(self._num_features_per_channel == num_features_per_channel)
        features = features.reshape((self._number_of_channels, self._num_features_per_channel))
        return features

    def train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self._model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
        self._is_trained = True

    def predict_command(self, eeg_data):
        if not self._is_trained:
            raise Exception("Model has not been trained yet.")
        features = self.extract_features(eeg_data)
        features = features.reshape(1, self._number_of_channels, self._num_features_per_channel)
        prediction = self._model.predict(features)
        predicted_label_index = np.argmax(prediction, axis=1)[0]
        predicted_command = self._reverse_label_map[predicted_label_index]
        return predicted_command

    def see_spectogram():
        bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 12),
            'Beta': (12, 30),
            'Gamma': (30, 64)
        }

        bandas_antes = { 'Delta': [], 'Theta': [], 'Alpha': [], 'Beta': [], 'Gamma': [] }
        bandas_durante = { 'Delta': [], 'Theta': [], 'Alpha': [], 'Beta': [], 'Gamma': [] }
        bandas_despues = { 'Delta': [], 'Theta': [], 'Alpha': [], 'Beta': [], 'Gamma': [] }


        lista_senales = [
            señales_filtradas03, señales_filtradas04, señales_filtradas15,
            señales_filtradas16, señales_filtradas18, señales_filtradas21, señales_filtradas26
        ]

    def compute_spectrogram(signal, fs, window, noverlap, nfft):
        f, t, Sxx = spectrogram(signal, fs, window=window, noverlap=noverlap, nfft=nfft)
        return f, t, Sxx

    def spectrogram_by_band(signal, fs, window, noverlap, nfft):
        
        spectrograms = { 'Delta': [], 'Theta': [], 'Alpha': [], 'Beta': [], 'Gamma': [] }
        
        for band, (low, high) in bands.items():
            filtered_signal = apply_band_pass_filter(signal, low, high, fs)
            
            f, t, Sxx = compute_spectrogram(filtered_signal, fs, window, noverlap, nfft)
            
            # Sumar la potencia en lugar de promediar
            band_power_sum = np.sum(Sxx, axis=1)
            
            spectrograms[band].extend(band_power_sum) 
        
        return spectrograms

def load_and_train_from_bci_dataset():
    # Initialize the ProcessingBias object
    fs = 250  # Sampling frequency from the dataset
    n = 1875  # Number of samples
    processing_bias = ProcessingBias(n=n, fs=fs)
    bias_filter = FilterBias(n=n, fs=fs, notch=True, bandpass=True, fir=False, iir=False)
    save_path = "bciiv2a"
    saved_dataset_path = None

    # Load BCI dataset
    datasetA1 = MotorImageryDataset("bcidatasetIV2a-master/A01T.npz")
    trials, classes = datasetA1.get_trials_from_channels([7, 9, 11])  # Example: C3, Cz, C4

    # Initialize X and y as empty lists|
    X = []
    y = []
    
    command_map = {"left": "left", "right": "right", "foot": "forward", "tongue": "backwards"}

    num_of_channels = 3

    # Initialize the AI model
    ai_bias = AIBias(n=n, fs=fs, channels=num_of_channels, commands=["left", "right", "forward", "backwards"]) # , "stop", "rest"])

    # Invert the dimensions of trials and classes using zip
    inverted_trials = list(map(list, zip(*trials)))
    inverted_classes = list(map(list, zip(*classes)))
    if saved_dataset_path is None:
        for num_trial in range(len(inverted_trials)):
            label = inverted_classes[num_trial][0]
            if label in command_map.keys():
                # Create a dictionary to hold the EEG signals for each channel
                eeg_signals = {f"ch{ch}": inverted_trials[num_trial][ch].tolist() for ch in range(num_of_channels)}  # Assuming 3 channels: C3, Cz, C4
                
                if num_trial == 2:
                    eeg_signal_to_save = eeg_signals
                    label_to_save = label

                filtered_data = bias_filter.filter_signals(eeg_signals)
                # Process the raw EEG signals using ProcessingBias to extract frequency bands
                _, processed_signals = processing_bias.process_signals(filtered_data)

                # Extract features from the processed signals (frequency bands)
                features = ai_bias.extract_features(processed_signals)

                # Append the extracted features and the corresponding command label
                X.append(features)
                y.append(ai_bias._label_map[command_map[label]])

        if save_path:
            # Save the dataset as a compressed NumPy file
            np.savez_compressed(f"{save_path}.npz", X=X, y=y)
            print(f"Dataset saved to {save_path}.npz")

    else:
        data = np.load(f"{saved_dataset_path}.npz")
        X, y = data['X'], data['y']

    # Convert lists to arrays for training
    X = np.array(X)
    y = np.array(y)

    # Convert labels to one-hot encoded format
    lb = LabelBinarizer()
    y = lb.fit_transform(y)

    # Train the model
    ai_bias.train_model(X, y)
    print("Training complete.")

    filtered_data_to_save = bias_filter.filter_signals(eeg_signal_to_save)
    # Process the raw EEG signals using ProcessingBias to extract frequency bands
    _, processed_signals_to_save = processing_bias.process_signals(filtered_data_to_save)
    predicted_command = ai_bias.predict_command(processed_signals_to_save)
    print(f"predicted: {predicted_command}")
    print(f"label_to_save: {label_to_save}")


if __name__ == "__main__":
    load_and_train_from_bci_dataset()
'''
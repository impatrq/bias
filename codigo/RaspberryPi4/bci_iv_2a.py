from tensorflow.keras.models import Sequential
from sklearn.model_selection import cross_val_score
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, InputLayer, LSTM, BatchNormalization
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from scipy.signal import welch, cwt, morlet
from scipy.stats import skew, kurtosis, entropy
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from bias_dsp import ProcessingBias, FilterBias
import plotext as plt
import random
import time

def main():
    n = 750
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
    #biasAI.make_predictions(filter_instance=biasFilter, processing_instance=biasProcessing)

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
        '''
        model = Sequential([
            InputLayer(shape=(self._number_of_channels, self._num_features_per_channel)),
            Conv1D(filters=128, kernel_size=3, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),

            LSTM(50, return_sequences=False),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(16, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        '''
        # CSP to extract spatial features + SVM classifier pipeline
        model = SVC(kernel='sigmoid', C=10, gamma='scale', class_weight='balanced')
        #model = SVC(kernel='linear', C=1)

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
        print(f"Unique classes in y_test: {np.unique(y_test)}")
        #self._model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
        #self.model_evaluation(X_test, y_test)

        self._model.fit(X_train, y_train)
        self.rendimiento_modelo_svm(self._model, X_test, y_test)
        '''
        # Train model
        # Build a pipeline
        # CSP to extract spatial features + SVM classifier pipeline
        # Initialize CSP (Common Spatial Patterns)

        csp = CSP(n_components=4, reg='ledoit_wolf', log=True)  # Choose `n_components` based on your experiment

        # Fit CSP to the training data (CSP will handle 3D shape internally)
        X_train_csp = csp.fit_transform(X_train, y_train)
        X_test_csp = csp.transform(X_test)

        # Now X_train_csp and X_test_csp are 2D arrays of shape (samples, components)

        # StandardScaler expects 2D data, so it's fine now
        scaler = StandardScaler()

        # Scale the CSP-transformed data
        X_train_scaled = scaler.fit_transform(X_train_csp)
        X_test_scaled = scaler.transform(X_test_csp)

        # Train SVM
        self._model.fit(X_train_scaled, y_train)
        #self.rendimiento_modelo_svm(self._model, X_train_scaled, y_train, X_test_scaled, y_test)
        '''

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
            # Invert the dimensions of trials and classes using zip
            inverted_trials = list(map(list, zip(*trials)))
            inverted_classes = list(map(list, zip(*classes)))
            print(f"trial length: {len(inverted_trials)}")
            all_trials.extend(inverted_trials)
            all_classes.extend(inverted_classes)
        print(f"trials length: {len(all_trials)}")

        return all_trials, all_classes
    
    def rendimiento_modelo_svm(self, svm, X_test, y_test):
        # Make predictions
        y_pred_test = svm.predict(X_test)

        # Calculate accuracy
        test_accuracy = accuracy_score(y_test, y_pred_test)
        print(f"Test Accuracy: {test_accuracy:.4f}")

        # Classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred_test, target_names=["forward", "backwards", "left", "right"]))  # Adjust class names as needed

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_test)
        print("Confusion Matrix:")
        print(cm)

    def make_predictions(self, filter_instance, processing_instance):
        file_list = [f"bcidatasetIV2a-master/A09T.npz"]
        trials, classes = self.load_datasets(file_list)

        for num_trial in range(len(trials)):
            label = classes[num_trial][0]
            print(f"Label: {label}")
            if label in self._command_map.keys():
                # Create a dictionary to hold the EEG signals for each channel
                eeg_signals = {f"ch{ch}": trials[num_trial][ch] for ch in range(self._number_of_channels)}  # Assuming 3 channels: C3, Cz, C4

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

    def segmentar_seniales(self, matrix, inicio, fin, fs=250):

        # Listas para almacenar los segmentos por canal
        segmentos_de_seniales = []  # Matriz de todos los bloques de las señales (ANTES, CRISIS, DESPUÉS)
        segmentos_de_seniales_completa = []  # Señales completas (Antes + Durante + Después)

        # Listas globales para almacenar todos los segmentos concatenados de todos los canales
        antes_total = []
        durante_total = []
        despues_total = []

        print(f"len matrix: {len(matrix)}, {len(matrix[0])}, {len(matrix[0][0])}")

        for trial in range(len(matrix)):
            matriz_trial = matrix[trial]
            antes_channel = []
            despues_channel = []
            durante_channel = []

            for ch in range(self._number_of_channels):
                antes_motor_imagery = matriz_trial[ch][(inicio -  3) * fs : inicio * fs].tolist()
                durante_motor_imagery = matriz_trial[ch][inicio * fs : fin * fs].tolist()
                despues_motor_imagery = matriz_trial[ch][fin * fs : (fin + 2) * fs].tolist()

                senial = [antes_motor_imagery, durante_motor_imagery, despues_motor_imagery]
                senial_completa = np.concatenate(senial)

                # Guardar en las listas por canal
                segmentos_de_seniales.append(senial)
                segmentos_de_seniales_completa.append(senial_completa)

                # Concatenar los segmentos a las listas globales
                antes_channel.append(antes_motor_imagery)
                durante_channel.append(durante_motor_imagery)
                despues_channel.append(despues_motor_imagery)

            antes_total.append(antes_channel)
            durante_total.append(durante_channel)
            despues_total.append(despues_channel)
            '''
            print(f"len antes_total: {len(antes_total)}")
            print(f"trials: {matrix.shape}")
            print(f"antes: {antes_total.shape}")
            print(f"durante: {durante_total.shape}")
            print(f"despues: {despues_total.shape}")
            '''
        n_samples_totales = len(segmentos_de_seniales_completa[0])  # Número total de muestras de la señal completa
        tiempo_inicial = inicio - 3  # En segundos, desde donde comenzamos el recorte
        time_total = tiempo_inicial + np.arange(n_samples_totales) / fs  # Vector de tiempo en segundos

        return segmentos_de_seniales, np.array(segmentos_de_seniales_completa), time_total, antes_total, durante_total, despues_total

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
            seniales, senial_completa, time_total, antes_total, durante_total, despues_total = self.segmentar_seniales(trials, 3, 6)
            for num_trial in range(len(trials)):
                label = classes[num_trial][0]
                if label in self._command_map.keys():
                    # Create a dictionary to hold the EEG signals for each channel
                    eeg_signals = {f"ch{ch}": antes_total[num_trial][ch] for ch in range(self._number_of_channels)}  # Assuming 4 channels: C3, Cz, C4

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

        unique_classes, counts = np.unique(y, return_counts=True)
        print(f"Classes in dataset: {unique_classes}, Counts: {counts}")

        print(f"x_Shape: {X.shape}")
        print(f"y_shape: {y.shape}")

        # Convert labels to one-hot encoded format using OneHotEncoder
        #one_hot_encoder = OneHotEncoder(sparse_output=False)
        #y_one_hot = one_hot_encoder.fit_transform(y.reshape(-1, 1))
        #print(f"y_one_hot_shape: {y_one_hot.shape}")

        #unique_classes, counts = np.unique(y_one_hot, return_counts=True)
        #print(f"Classes in dataset: {unique_classes}, Counts: {counts}")

        X_reshaped = X.reshape(X.shape[0], -1)

        # Train the model
        self.train_model(X_reshaped, y)

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
from sklearn.svm import SVC
from scipy.signal import welch, cwt, morlet
from scipy.stats import skew, kurtosis, entropy
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
from bias_dsp import ProcessingBias, FilterBias
import plotext as plt
import random
import time

def main():
    n = 750
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
    #biasAI.make_predictions(filter_instance=biasFilter, processing_instance=biasProcessing)

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
            MaxPooling1D(pool_size=2),
            Dropout(0.5),
            Flatten(),
            #LSTM(50, return_sequences=False),
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
        self.model_evaluation(X_test, y_test)
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
            # Invert the dimensions of trials and classes using zip
            inverted_trials = list(map(list, zip(*trials)))
            inverted_classes = list(map(list, zip(*classes)))
            print(f"trial length: {len(inverted_trials)}")
            all_trials.extend(inverted_trials)
            all_classes.extend(inverted_classes)
        print(f"trials length: {len(all_trials)}")

        return all_trials, all_classes

    def rendimiento_modelo_svm(self, X_train, y_train, X_test, y_test):
        modelo = SVC(kernel='rbf', C=10, gamma='scale')

        modelo.fit(X_train, y_train)

        y_pred_test = modelo.predict(X_test)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        precision_test = precision_score(y_test, y_pred_test)

        y_pred_train = modelo.predict(X_train)
        accuracy_train = accuracy_score(y_train, y_pred_train)
        precision_train = precision_score(y_train, y_pred_train)

        accuracy = accuracy_score(y_test, y_pred_test)
        print(f"Accuracy del modelo: {accuracy * 100:.2f}%")
        precision = precision_score(y_test, y_pred_test)
        print(f"Precisión del modelo: {precision * 100:.2f}%")

        print(f"Accuracy del modelo en entrenamiento: {accuracy_train * 100:.2f}%")
        print(f"Accuracy del modelo en prueba: {accuracy_test * 100:.2f}%")
        print(f"Precisión del modelo en entrenamiento: {precision_train * 100:.2f}%")
        print(f"Precisión del modelo en prueba: {precision_test * 100:.2f}%")

        conf_matrix = confusion_matrix(y_test, y_pred_test)
        TN = conf_matrix[0, 0]
        FP = conf_matrix[0, 1]
        FN = conf_matrix[1, 0]
        TP = conf_matrix[1, 1]
        TPR = TP / (TP + FN)
        TNR = TN / (TN + FP)

        print(f"TPR (True Positive Rate - Sensibilidad): {TPR:.2f}")
        print(f"TNR (True Negative Rate - Especificidad): {TNR:.2f}")
        print("Matriz de confusión:")
        print(conf_matrix)

    def make_predictions(self, filter_instance, processing_instance):
        file_list = [f"bcidatasetIV2a-master/A09T.npz"]
        trials, classes = self.load_datasets(file_list)

        for num_trial in range(len(trials)):
            label = classes[num_trial][0]
            print(f"Label: {label}")
            if label in self._command_map.keys():
                # Create a dictionary to hold the EEG signals for each channel
                eeg_signals = {f"ch{ch}": trials[num_trial][ch] for ch in range(self._number_of_channels)}  # Assuming 3 channels: C3, Cz, C4

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

    def segmentar_seniales(self, matrix, inicio, fin, fs=250):

        # Listas para almacenar los segmentos por canal
        segmentos_de_seniales = []  # Matriz de todos los bloques de las señales (ANTES, CRISIS, DESPUÉS)
        segmentos_de_seniales_completa = []  # Señales completas (Antes + Durante + Después)

        # Listas globales para almacenar todos los segmentos concatenados de todos los canales
        antes_total = []
        durante_total = []
        despues_total = []

        print(f"len matrix: {len(matrix)}, {len(matrix[0])}, {len(matrix[0][0])}")

        for trial in range(len(matrix)):
            matriz_trial = matrix[trial]
            antes_channel = []
            despues_channel = []
            durante_channel = []

            for ch in range(self._number_of_channels):
                antes_motor_imagery = matriz_trial[ch][(inicio -  3) * fs : inicio * fs].tolist()
                durante_motor_imagery = matriz_trial[ch][inicio * fs : fin * fs].tolist()
                despues_motor_imagery = matriz_trial[ch][fin * fs : (fin + 2) * fs].tolist()

                senial = [antes_motor_imagery, durante_motor_imagery, despues_motor_imagery]
                senial_completa = np.concatenate(senial)

                # Guardar en las listas por canal
                segmentos_de_seniales.append(senial)
                segmentos_de_seniales_completa.append(senial_completa)

                # Concatenar los segmentos a las listas globales
                antes_channel.append(antes_motor_imagery)
                durante_channel.append(durante_motor_imagery)
                despues_channel.append(despues_motor_imagery)

            antes_total.append(antes_channel)
            durante_total.append(durante_channel)
            despues_total.append(despues_channel)
            '''
            print(f"len antes_total: {len(antes_total)}")
            print(f"trials: {matrix.shape}")
            print(f"antes: {antes_total.shape}")
            print(f"durante: {durante_total.shape}")
            print(f"despues: {despues_total.shape}")
            '''
        n_samples_totales = len(segmentos_de_seniales_completa[0])  # Número total de muestras de la señal completa
        tiempo_inicial = inicio - 3  # En segundos, desde donde comenzamos el recorte
        time_total = tiempo_inicial + np.arange(n_samples_totales) / fs  # Vector de tiempo en segundos

        return segmentos_de_seniales, np.array(segmentos_de_seniales_completa), time_total, antes_total, durante_total, despues_total

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
            seniales, senial_completa, time_total, antes_total, durante_total, despues_total = self.segmentar_seniales(trials, 3, 6)
            for num_trial in range(len(trials)):
                label = classes[num_trial][0]
                if label in self._command_map.keys():
                    # Create a dictionary to hold the EEG signals for each channel
                    eeg_signals = {f"ch{ch}": antes_total[num_trial][ch] for ch in range(self._number_of_channels)}  # Assuming 4 channels: C3, Cz, C4

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
        #self.train_model(X, y) # Train in with CNN
        self.rendimiento_modelo_svm(X, y) # Train it with SVM model
        print("Training complete.")

if __name__ == "__main__":
    main()
'''
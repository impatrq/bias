from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, InputLayer, LSTM, BatchNormalization, AveragePooling1D, Activation, GlobalAveragePooling1D, Flatten, Conv2D, DepthwiseConv2D, SeparableConv2D, BatchNormalization, AveragePooling2D, Input, MultiHeadAttention, LayerNormalization, Reshape, SpatialDropout2D
from tensorflow.keras.regularizers import l2
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from keras import activations
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.layers import Add, Concatenate, Lambda, Input, Permute
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from scipy.signal import welch, cwt, morlet
from scipy.stats import skew, kurtosis, entropy
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from mne.decoding import CSP
from tensorflow.keras import backend as K
from sklearn.model_selection import GridSearchCV
from bias_dsp import ProcessingBias, FilterBias
from bias_reception import ReceptionBias
from signals import generate_synthetic_eeg, generate_synthetic_eeg_bandpower
from sklearn.base import BaseEstimator, ClassifierMixin
from bias_graphing import GraphingBias
from sklearn.utils.class_weight import compute_class_weight
CNN = False
SVM = False
CSPT = False
ALL = False

def main():
    n = 1000
    fs = 250
    number_of_channels = 4
    port = '/dev/serial0'
    baudrate = 115200
    timeout = 1 
    
    biasReception = ReceptionBias(port=port, baudrate=baudrate, timeout=timeout)
    biasFilter = FilterBias(n=n, fs=fs, notch=True, bandpass=True, fir=False, iir=False)
    biasProcessing = ProcessingBias(n=n, fs=fs)
    biasGraphing = GraphingBias(graph_in_terminal=True)
    commands = ["forward", "backwards", "left", "right"]

    global CNN, SVM, CSPT, ALL

    while True:
        algorithm = input("Choose an algorithm: (cnn/svm/csp/all): ")
        if algorithm.lower().strip() == "cnn":
            CNN = True
            print("CNN model selected")
            break
        elif algorithm.lower().strip() == "svm":
            SVM = True
            print("SVM model selected")
            break
        elif algorithm.lower().strip() == "csp":
            CSPT = True
            print("CSP model selected")
            break
        elif algorithm.lower().strip() == "all":
            ALL = True
            print("All models selected")
            break
        else:
            print("Error selecting type of model, try again.")

    biasAI = None 

    while True:
        model_lt = input("Do you want to load or train a model? (l/t): ")

        saved_dataset_path = None
        save_new_dataset_path = None
        model_path = None
        
        if model_lt.lower() == "t" or model_lt.lower() == "train":
            biasAI = AIBias(n=n, fs=fs, channels=number_of_channels, commands=commands, model=None)
            loading_dataset = input("Do you want to load an existent dataset? (y/n): ")
            if loading_dataset.lower().strip() == "y" or loading_dataset.lower().strip() == "yes":
                saved_dataset_path = input("Write the name of the file where dataset was saved (extension .npz): ")
            else:
                save_new_dataset = input("Do you want to save the new dataset? (y/n): ")
                if save_new_dataset.lower().strip() == "y" or save_new_dataset.lower().strip() == "yes":
                    save_new_dataset_path = input("Write the path where you want to save the dataset (extensio .npz): ")
            save_model = input("Do you want to save the model (y/n): ")
            if save_model.lower().strip() == "y" or save_model.lower().strip() == "yes":
                model_path = input("Write the filename where model will be saved (exttension .keras): ")

            biasAI.collect_and_train_from_bci_dataset(filter_instance=biasFilter, processing_instance=biasProcessing, save_new_dataset_path=save_new_dataset_path, saved_dataset_path=saved_dataset_path, model_path=model_path)
            break
        elif model_lt.lower().strip() == "l" or model_lt.lower().strip() == "load":
            model_name = input("Write the filname where model is saved (extension .keras): ")
            model = load_model(f"{model_name}.keras")
            biasAI = AIBias(n=n, fs=fs, channels=number_of_channels, commands=commands, model=model)
            break
        else:
            print("Mode invalid. Choose between loading and training.")

    real_data = input("Do you want to get real data? (y/n): ")
    for i in range(10):
        if real_data.lower().strip() == 'y' or real_data.lower().strip() == "yes":
            signals = biasReception.get_real_data(channels=number_of_channels, n=n)
        else:
            signals = generate_synthetic_eeg(n_samples=n, n_channels=number_of_channels, fs=fs)

        filtered_data = biasFilter.filter_signals(eeg_signals=signals) 

        
        # Calculate the time vector
        t = np.arange(n) / fs
        # Graph signals
        for ch, signal in filtered_data.items():
           # Graph filtered signal
            biasGraphing.graph_signal_voltage_time(t=t, signal=np.array(signal), title="Filtered Signal {}".format(ch))

        # Process data
        #times, eeg_signals = biasProcessing.process_signals(eeg_signals=filtered_data)

        predicted_command = biasAI.predict_command(eeg_data=filtered_data)
        print(f"Predicted Command: {predicted_command}")

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

    def get_trials_from_channels(self, channels=[[6, 7], [9, 10], [11, 12], [19, 20]]):
        trials_c, classes_c = [], []
    
        for set_of_channels in channels:
            ch0, ch1 = set_of_channels[0], set_of_channels[1]
        
            # Get trials for each channel in the pair
            t0, c0 = self.get_trials_from_channel(channel=ch0)
            t1, c1 = self.get_trials_from_channel(channel=ch1)
        
            # Ensure both channels have the same classes
            assert(len(t0) == len(t1))
        
            # Initialize `t` to store differential data for this channel pair
            t = []
        
            # Loop through each trial and perform sample-wise subtraction
            for trial in range(len(t0)):
                assert(len(t0[trial]) == len(t1[trial]))  # Ensure sample lengths match
                differential_trial = [t0[trial][sample] - t1[trial][sample] for sample in range(len(t0[trial]))]
                t.append(differential_trial)
        
                # Confirm class labels match and append data
                assert(c0[trial] == c1[trial])
            tt = np.concatenate(t, axis=0)

            trials_c.append(tt)  # Convert to array for each channel pair
            classes_c.append(c0)  # Use the class labels from `c0` (or `c1`, as they should be identical)
    
        return trials_c, classes_c

class AIBias:
    def __init__(self, n, fs, channels, commands, model=None):
        self._n = n
        self._fs = fs
        self._number_of_channels = channels
        self._features_length = len(["mean", "variance", "skewness", "kurt", "energy", "band_power", "wavelet_energy", "entropy"])
        self._number_of_waves_per_channel = len(["signal", "alpha", "beta", "gamma", "delta", "theta"])
        self._num_features_per_channel = self._features_length * self._number_of_waves_per_channel
        self._commands = commands
        if model is None:
            self._model = self.build_model(output_dimension=len(self._commands))
            self._is_trained = False
        else:
            self._model = model
            self._is_trained = True
        self._label_map = {command: idx for idx, command in enumerate(self._commands)}
        self._reverse_label_map = {idx: command for command, idx in self._label_map.items()}
        self._command_map = {"left": "left", "right": "right", "foot": "forward", "tongue": "backwards"}
        self._montage = {'Fz': 0, 'FC3': 1, 'FC1': 2, 'FCz': 3, 'FC2': 4, 'FC4': 5, 'C5': 6, 'C3': 7, 'C1': 8,
                         'Cz': 9, 'C2': 10, 'C4': 11, 'C6': 12, 'CP3': 13, 'CP1': 14, 'CPz': 15, 'CP2': 16,
                         'CP4': 17, 'P3': 18, 'Pz': 19, 'P4': 20, 'Oz':21}

    # Augmentation function
    def augment_data(self, data):
        # Gaussian noise
        noise = np.random.normal(0, 0.01, data.shape)
        data_with_noise = data + noise
    
        # Amplitude scaling
        scale = np.random.uniform(0.9, 1.1)
        data_scaled = data_with_noise * scale
    
        # Baseline shift
        shift = np.random.uniform(-0.05, 0.05)
        data_shifted = data_scaled + shift
    
        return data_shifted


    def standardize_data(self, X_train, X_test): 
        # X_train & X_test :[Trials, MI-tasks, Channels, Time points]
        for j in range(self._number_of_channels):
            scaler = StandardScaler()
            scaler.fit(X_train[:, 0, j, :])
            X_train[:, 0, j, :] = scaler.transform(X_train[:, 0, j, :])
            X_test[:, 0, j, :] = scaler.transform(X_test[:, 0, j, :])

        return X_train, X_test

    def bandpass_filter(self, data, low_freq, high_freq, fs=250, order=4):
        sos = butter(order, [low_freq, high_freq], btype='band', fs=fs, output='sos')
        return sosfilt(sos, data)

    def ShallowConvNet(self, nb_classes=4, dropoutRate = 0.7, l2_rate=0.01):
        # start the model
        # input_main   = Input((Chans, Samples, 1))
        input_main   = Input((1, self._number_of_channels, self._n))
        input_2 = Permute((2,3,1))(input_main) 
        block1       = Conv2D(40, (1, 25), 
                              input_shape=(self._number_of_channels, self._n, 1),
                              kernel_constraint = max_norm(2., axis=(0,1,2)), kernel_regularizer=l2(l2_rate))(input_2)
        block1       = Conv2D(40, (self._number_of_channels, 1), use_bias=False, 
                          kernel_constraint = max_norm(2., axis=(0,1,2)), kernel_regularizer=l2(l2_rate))(block1)
        block1       = BatchNormalization(epsilon=1e-05, momentum=0.8)(block1)
        block1       = Activation(self.square)(block1)
        block1       = AveragePooling2D(pool_size=(1, 75), strides=(1, 15))(block1)
        block1       = Activation(self.log)(block1)
        block1 = Dropout(dropoutRate)(block1)
        #block1 = SpatialDropout2D(dropoutRate)(block1)
        flatten      = Flatten()(block1)
        dense        = Dense(nb_classes, kernel_constraint = max_norm(0.5))(flatten)
        softmax      = Activation('softmax')(dense)
    
        model =  Model(inputs=input_main, outputs=softmax)

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model.summary()
        
        return model

    #%% need these for ShallowConvNet
    def square(self, x):
        return K.square(x)

    def log(self, x):
        return K.log(K.clip(x, min_value = 1e-7, max_value = 10000))
    
    def EEGNet(self):
        F1=8
        D=2
        kernLength=64
        dropout=0.25

        input1 = Input(shape = (1, self._number_of_channels, self._n))   
        input2 = Permute((3,2,1))(input1) 
        regRate=.25

        F2 = F1*D
        block1 = Conv2D(F1, (kernLength, 1), padding = 'same',data_format='channels_last',use_bias = False)(input2)
        block1 = BatchNormalization(axis = -1)(block1)
        block2 = DepthwiseConv2D((1, self._number_of_channels), use_bias = False, 
        depth_multiplier = D,
        data_format='channels_last',
        depthwise_constraint = max_norm(1.))(block1)

        block2 = BatchNormalization(axis = -1)(block2)
        block2 = Activation('elu')(block2)
        block2 = AveragePooling2D((8,1),data_format='channels_last')(block2)
        block2 = Dropout(dropout)(block2)
        block3 = SeparableConv2D(F2, (16, 1),
        data_format='channels_last',
        use_bias = False, padding = 'same')(block2)

        block3 = BatchNormalization(axis = -1)(block3)
        block3 = Activation('elu')(block3)
        block3 = AveragePooling2D((8,1),data_format='channels_last')(block3)
        block3 = Dropout(dropout)(block3)
            
        eegnet = Flatten()(block3)
        dense = Dense(4, name = 'dense',kernel_constraint = max_norm(regRate))(eegnet)
        softmax = Activation('softmax', name = 'softmax')(dense)

        model = Model(inputs=input1, outputs=softmax) 

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model.summary()

        return model


    def build_model(self, output_dimension):
        global CNN, SVM, CSPT, ALL
        if CNN or CSP:
            model = self.ShallowConvNet()

        if SVM or ALL:
            # CSP to extract spatial features + SVM classifier pipeline
            model = SVC(kernel='linear', C=0.1, gamma='scale', class_weight='balanced')
            #model = SVC(kernel='linear', C=1)
            #model = SVC(class_weight='balanced')

        return model

    def extract_features(self, eeg_data):
        features = []
        for ch, signals_per_channel in eeg_data.items():
            channel_features = []
            assert(len(signals_per_channel) == self._number_of_waves_per_channel)
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
                assert(len(list_of_features) == self._features_length)
            features.append(channel_features)
        features = np.abs(np.array(features))
        features = self._scaler.fit_transform(features)
        num_features_per_channel = features.shape[1]
        assert(self._num_features_per_channel == num_features_per_channel)
        features = features.reshape((self._number_of_channels, self._num_features_per_channel))
        return features
    
    def plot_confusion_matrix(y_true, y_pred_classes, labels):
        cm = confusion_matrix(y_true, y_pred_classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.show()

    def train_model(self, X, y, model_path=None):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        #X_train = shuffle(X_train)
        #X_test = shuffle(X_test)
        X_train, X_test = self.standardize_data(X_train, X_test)
        print(f"Unique classes in y_test: {np.unique(y_test)}")
        
        global CNN, SVM, ALL, CSPT

        if CNN:
            # Define all class labels explicitly
            all_classes = np.arange(y_train.shape[1])  # This will be [0, 1, 2, 3] for 4 classes
            class_weights = dict(enumerate(compute_class_weight('balanced', classes=all_classes, y=y_train.argmax(axis=1))))
            print("Training")

            # Define EarlyStopping to monitor validation loss with a patience of 20 epochs
            early_stopping_callback = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, mode='min')
            lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=10, min_lr=1e-5)
            if model_path is not None:
                checkpoint_callback = ModelCheckpoint(f'{model_path}.keras', monitor='val_accuracy', save_best_only=True, mode='max')
        
            callbacks = [early_stopping_callback, checkpoint_callback, lr_scheduler]
            history = self._model.fit(
                    X_train, y_train, 
                    epochs=300, batch_size=32, 
                    validation_data=(X_test, y_test), 
                    class_weight=class_weights, 
                    callbacks=callbacks
                    )
            self._is_trained = True
            self.model_evaluation(X_test, y_test)

        if SVM:
            # StandardScaler expects 2D data, so it's fine now
            scaler = StandardScaler()

            # Scale the CSP-transformed data
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            self._model.fit(X_train_scaled, y_train)
            self._is_trained = True
            self.rendimiento_modelo_svm(self._model, X_test_scaled, y_test)
       
        if ALL:
            print("hola")
            # Step 1: Set up a pipeline to combine PCA and CSP
            pca_csp_pipeline = Pipeline([
                #('csp', CSP(n_components=4, reg='ledoit_wolf', log=True)),  # CSP to extract spatial patterns
                ('pca', PCA(n_components=0.95)),      # PCA to reduce feature dimensionality 
                ('scaler', StandardScaler()),         # Standardize the features
                ('svm', SVC(kernel='rbf', C=1, gammea='scale', class_weight='balanced'))  # SVM classifier
            ])

            # Step 2: Fit the pipeline to the training data
            pca_csp_pipeline.fit(X_train, y_train)  # X_train should be raw EEG data (samples x channels x time)
            self._is_trained = True
            # Step 3: Evaluate the pipeline on test data
            y_pred = pca_csp_pipeline.predict(X_test)
            print(f"Test Accuracy: {accuracy_score(y_test, y_pred)}")
            print(f"Classification Report:\n {classification_report(y_test, y_pred)}")

        if CSPT:
            # Train model
            # Build a pipeline
            # CSP to extract spatial features + SVM classifier pipeline
            # Initialize CSP (Common Spatial Patterns)
            csp = CSP(n_components=4, reg='ledoit_wolf', log=True)  # Choose `n_components` based on your experiment

            X_train_csp = csp.fit_transform(X_train, y_train)
            X_test_csp = csp.transform(X_test)
            '''
            # Define parameter grid
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto'],
                'kernel': ['linear', 'rbf', 'sigmoid', 'poly']
            }

            # Initialize SVM with balanced class weight
            svm = SVC(class_weight='balanced')

            # Perform grid search
            grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
            grid_search.fit(X_train_csp, y_train)

            # Best parameters and model
            best_params = grid_search.be            # Compile the model
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
st_params_
            best_model = grid_search.best_estimator_
            print(f"Best SVM parameters: {best_params}")
            '''
            svm = SVC(C=10, gamma='scale', kernel='rbf', class_weight='balanced')
            svm.fit(X_train_csp, y_train)
            accuracy = svm.score(X_test_csp, y_test)
            print(f'SVM Accuracy with CSP: {accuracy}')

            '''
            # Train SVM
            self._model.fit(X_train_scaled, y_train)
            self._is_trained = True
            self.rendimiento_modelo_svm(self._modele, X_test_scaled, y_test)
            
            
            # Convert labels to one-hot encoded format using OneHotEncoder
            one_hot_encoder = OneHotEncoder(sparse_output=False)
            y_one_hot_test = one_hot_encoder.fit_transform(y_test.reshape(-1, 1))
            y_one_hot_train = one_hot_encoder.fit_transform(y_train.reshape(-1, 1))

            print(f"y_one_hot_shape: {y_one_hot_test.shape}")

            self._model.fit(X_train, y_one_hot_train, epochs=10, batch_size=32, validation_data=(X_test, y_one_hot_test))
            self._is_trained = True
            self.model_evaluation(X_test, y_one_hot_test)
            '''

    def predict_command(self, eeg_data):
        if not self._is_trained:
            raise Exception("Model has not been trained yet.")
        # Stack the lists from each channel along the first axis (rows)
        # Extract the values from the dictionary and stack them as rows
        print(eeg_data)
        eeg_data = np.stack([eeg_data[key] for key in sorted(eeg_data.keys())])
        for ch in range(self._number_of_channels):
            for sample_number in range(self._n):
                eeg_data[ch][sample_number] = eeg_data[ch][sample_number] / 29.47045
        print(eeg_data.shape)
        eeg_reshaped = eeg_data.reshape(1, 1, self._number_of_channels, self._n)
        if SVM:
            features_reshaped = features_reshaped.reshape(features_reshaped.shape[0], -1)
        prediction = self._model.predict(eeg_reshaped)
        predicted_label_index = np.argmax(prediction, axis=1)[0]
        predicted_command = self._reverse_label_map[predicted_label_index]
        return predicted_command

    def load_datasets(self, file_names):
        montage = {'Fz': 0, 'FC3': 1, 'FC1': 2, 'FCz': 3, 'FC2': 4, 'FC4': 5, 'C5': 6, 'C3': 7, 'C1': 8,
                         'Cz': 9, 'C2': 10, 'C4': 11, 'C6': 12, 'CP3': 13, 'CP1': 14, 'CPz': 15, 'CP2': 16,
                         'CP4': 17, 'P3': 18, 'Pz': 19, 'P4': 20, 'Oz': 21}

        all_trials = []
        all_classes = []
        for file_name in file_names:
            dataset = MotorImageryDataset(file_name)
            trials, classes = dataset.get_trials_from_channels([[montage['C3'], montage['FC1']], 
                                                                [montage['C4'], montage['FC2']],
                                                                [montage['Cz'], montage['Fz']], 
                                                                [montage['FC3'], montage['FC4']]])

            # Invert the dimensions of trials and classes using zip
            inverted_trials = list(map(list, zip(*trials)))
            inverted_classes = list(map(list, zip(*classes)))
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
                antes_motor_imagery = matriz_trial[ch][0 : int(inicio * fs)].tolist()
                durante_motor_imagery = matriz_trial[ch][int(inicio * fs) : fin * fs].tolist()
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

        n_samples_totales = len(segmentos_de_seniales_completa[0])  # Número total de muestras de la señal completa
        tiempo_inicial = 0  # En segundos, desde donde comenzamos el recorte
        time_total = tiempo_inicial + np.arange(n_samples_totales) / fs  # Vector de tiempo en segundos

        return segmentos_de_seniales, np.array(segmentos_de_seniales_completa), time_total, np.array(antes_total), np.array(durante_total), np.array(despues_total)

    def model_evaluation(self, X_test, y_test):
        # Evaluate model performance on the test set
        loss, accuracy = self._model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {accuracy}")

        # Get model predictions on the test set
        y_pred = self._model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)  # Convert one-hot encoding to class labels

        # Print confusion matrix and classification report
        y_true = np.argmax(y_test, axis=1)
        cm = confusion_matrix(y_true, y_pred_classes)
        print(cm)
    
        print(classification_report(y_true, y_pred_classes, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3']))
        self.plot_confusion_matrix(y_true, y_pred_classes)
    
    def collect_and_train_from_bci_dataset(self, filter_instance, processing_instance, save_new_dataset_path, saved_dataset_path, model_path):
        # Initialize X and y as empty lists
        X = []
        y = []

        if saved_dataset_path is None:
            file_list = [f"bcidatasetIV2a-master/A0{i}T.npz" for i in range(1, 9)]
            trials, classes = self.load_datasets(file_list)
            seniales, senial_completa, time_total, antes_total, durante_total, despues_total = self.segmentar_seniales(trials, 2, 6)
            #durante_total = self.normalize_eeg_data_zscore(durante_total)
            for num_trial in range(len(trials)):
                label = classes[num_trial][0]
                if label in self._command_map.keys():
                    # Create a dictionary to hold the EEG signals for each channel
                    eeg_signals = {f"ch{ch}": durante_total[num_trial][ch] for ch in range(self._number_of_channels)}  # Assuming 4 channels: C3, Cz, C4, FPz
                    
                    for ch, signal in eeg_signals.items():
                        signal = self.augment_data(signal)

                    filtered_data = filter_instance.filter_signals(eeg_signals)
                    # Process the raw EEG signals using ProcessingBias to extract frequency bands
                    #_, processed_signals = processing_instance.process_signals(filtered_data)
                    
                    # Extract features from the processed signals (frequency bands)
                    #features = self.extract_features(processed_signals)

                    # Append the extracted features and the corresponding command label
                    #X.append(features)
                    eeg_data = np.stack([filtered_data[key] for key in sorted(filtered_data.keys())])
                    X.append(durante_total[num_trial])
                    y.append(self._label_map[self._command_map[label]])

            if save_new_dataset_path:
                # Save the dataset as a compressed NumPy file
                np.savez_compressed(f"{save_new_dataset_path}.npz", X=X, y=y)
                print(f"Dataset saved to {save_new_dataset_path}.npz")

        else:
            data = np.load(f"{saved_dataset_path}.npz")
            X, y = data['X'], data['y']

        # Convert lists to arrays for training
        X = np.array(X)
        print(X.shape)
        y = np.array(y)
        
        num_trials = len(X)
        if CNN:
            X = X.reshape(num_trials, 1, self._number_of_channels, self._n)
            y_one_hot = to_categorical(y)
            print(f"y_one_hot_shape: {y_one_hot.shape}")

            unique_classes, counts = np.unique(y_one_hot, return_counts=True)
            print(f"Classes in dataset: {unique_classes}, Counts: {counts}")
            self.train_model(X, y_one_hot, model_path)

        if SVM:
            X_reshaped = X.reshape(X.shape[0], -1)
            self.train_model(X_reshaped, y)

        if ALL or CSPT:
            #X.reshape(6273000, self._number_of_channels, 750)
            self.train_model(X, y)

        print("Training complete.")

if __name__ == "__main__":
    main()

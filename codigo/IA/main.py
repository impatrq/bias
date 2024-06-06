import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# Paso 1: Adquisición de Datos
# Supongamos que tienes un DataFrame df con las señales EEG y las etiquetas correspondientes

# Paso 2: Preprocesamiento de Datos
def preprocess_data(df):
    # Normalizar las características
    scaler = StandardScaler()
    features = df.drop('label', axis=1).values
    labels = df['label'].values
    features_scaled = scaler.fit_transform(features)
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)
    
    # Convertir etiquetas a one-hot encoding
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    return X_train, X_test, y_train, y_test

# Paso 3: Extracción de Características
# En este ejemplo, asumimos que las características ya están extraídas y presentes en el DataFrame df

# Paso 4: Entrenamiento del Modelo
def build_model(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(5, activation='softmax'))  # 5 clases: avanzar, retroceder, girar derecha, girar izquierda, frenar
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Supongamos que df es el DataFrame que contiene las características y las etiquetas
df = pd.read_csv('eeg_data.csv')  # Cargar datos (modificar según tu fuente de datos)

# Preprocesar datos
X_train, X_test, y_train, y_test = preprocess_data(df)

# Construir y entrenar el modelo
input_dim = X_train.shape[1]
model = build_model(input_dim)
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Paso 5: Evaluación y Validación
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Precisión del modelo: {accuracy*100:.2f}%')

# Guardar el modelo entrenado
model.save('eeg_model.h5')

# Cargar el modelo y realizar predicciones (si es necesario)
# from keras.models import load_model
# model = load_model('eeg_model.h5')
# predictions = model.predict(X_test)
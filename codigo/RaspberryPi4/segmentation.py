import numpy as np
from bci_iv_2a import MotorImageryDataset

def segmentar_seniales(matriz, inicio, fin, fs=250):
    matriz = np.array(matriz)

    # Listas para almacenar los segmentos por canal
    segmentos_de_seniales = []  # Matriz de todos los bloques de las señales (ANTES, CRISIS, DESPUÉS)
    segmentos_de_seniales_completa = []  # Señales completas (Antes + Durante + Después)

    # Listas globales para almacenar todos los segmentos concatenados de todos los canales
    antes_total = []
    durante_total = []
    despues_total = []

    for i in range(len(matriz)):
        antes_motor_imagery = matriz[i, (inicio -  3) * fs : inicio * fs]
        durante_motor_imagery = matriz[i, inicio * fs : fin * fs]
        despues_motor_imagery = matriz[i, fin * fs : (fin + 2) * fs]

        senial = [antes_motor_imagery, durante_motor_imagery, despues_motor_imagery]
        senial_completa = np.concatenate(senial)
        
        # Guardar en las listas por canal
        segmentos_de_seniales.append(senial)
        segmentos_de_seniales_completa.append(senial_completa)

        # Concatenar los segmentos a las listas globales
        antes_total.append(antes_motor_imagery)
        durante_total.append(durante_motor_imagery)
        despues_total.append(despues_motor_imagery)

    # Convertir las listas globales en arrays para facilitar su uso
    antes_total = np.concatenate(antes_total)
    durante_total = np.concatenate(durante_total)
    despues_total = np.concatenate(despues_total)

    n_samples_totales = len(segmentos_de_seniales_completa[0])  # Número total de muestras de la señal completa
    tiempo_inicial = inicio - 3  # En segundos, desde donde comenzamos el recorte
    time_total = tiempo_inicial + np.arange(n_samples_totales) / fs  # Vector de tiempo en segundos

    return segmentos_de_seniales, np.array(segmentos_de_seniales_completa), time_total, antes_total, durante_total, despues_total


dataset = MotorImageryDataset("bcidatasetIV2a-master/A01T.npz")
bandas = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 12),
    'Beta': (12, 30),
    'Gamma': (30, 64)
}
trials, classes = dataset.get_trials_from_channels([0, 7, 9, 11])
# Invert the dimensions of trials and classes using zip
trials = list(map(list, zip(*trials)))
classes = list(map(list, zip(*classes)))
seniales03, senial_completa03, time_total03, antes_total03, durante_total03, despues_total03 = segmentar_seniales(trials[0], 3, 6)
print(f"trials shape: {trials[0].shape}")
print(f"antes shape {antes_total03}")
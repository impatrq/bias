import numpy as np
from bci_iv_2a import MotorImageryDataset

def segmentar_seniales(matrix, inicio, fin, fs=250):
    number_of_channels = 4
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

        for ch in range(number_of_channels):
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
seniales, senial_completa, time_total, antes_total, durante_total, despues_total = segmentar_seniales(trials, 3, 6)

#print(f"trials shape: {trials}")
print(f"antes shape 1st: {len(antes_total)}")
print(f"antes shape 2nd: {len(antes_total[0])}")
print(f"despues shape 3rd: {len(antes_total[0][0])}")
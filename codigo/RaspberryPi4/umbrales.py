from bci_iv_2a import MotorImageryDataset
import numpy as np

def main():
    umbrales_instance = Umbrales()
    dataset = MotorImageryDataset()
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 12),
        'Beta': (12, 30),
        'Gamma': (30, 64)
    }
    trials, classes = dataset.get_trials_from_channel()
    # Invert the dimensions of trials and classes using zip
    trials = list(map(list, zip(*trials)))
    classes = list(map(list, zip(*classes)))
    umbrales_instance.calcular_umbrales(bands)
    umbrales_instance.see_umbrales()

class Umbrales:
    def segmentar_seniales(matriz, inicio, fin, fs=250):

        # Listas para almacenar los segmentos por canal
        segmentos_de_seniales = []  # Matriz de todos los bloques de las señales (ANTES, CRISIS, DESPUÉS)
        segmentos_de_seniales_completa = []  # Señales completas (Antes + Durante + Después)

        # Listas globales para almacenar todos los segmentos concatenados de todos los canales
        antes_total = []
        durante_total = []
        despues_total = []

        for i in range(len(matriz)):
            antes_motor_imagery = matriz[i, (inicio -  120) * fs : inicio * fs]
            durante_motor_imagery = matriz[i, inicio * fs : fin * fs]
            despues_motor_imagery = matriz[i, fin * fs : (fin + 120) * fs]

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
        tiempo_inicial = inicio - 120  # En segundos, desde donde comenzamos el recorte
        time_total = tiempo_inicial + np.arange(n_samples_totales) / fs  # Vector de tiempo en segundos

        return segmentos_de_seniales, np.array(segmentos_de_seniales_completa), time_total, antes_total, durante_total, despues_total

    def calcular_umbrales(self, bandas):

        delta = [float(min(bandas['Delta'])), float(max(bandas['Delta']))]
        theta = [float(min(bandas['Theta'])), float(max(bandas['Theta']))]
        alpha = [float(min(bandas['Alpha'])), float(max(bandas['Alpha']))]
        beta = [float(min(bandas['Beta'])), float(max(bandas['Beta']))]
        gamma = [float(min(bandas['Gamma'])), float(max(bandas['Gamma']))]
            
        umbrales = {'Delta': delta, 'Theta': theta, 'Alpha': alpha, 'Beta': beta, 'Gamma':gamma}

        return umbrales

    def see_umbrales(self, bandas_antes, bandas_durante, bandas_despues):
        # Calcular los umbrales
        umbrales_bandas_antes = self.calcular_umbrales(bandas_antes)
        umbrales_bandas_durante = self.calcular_umbrales(bandas_durante)
        umbrales_bandas_despues = self.calcular_umbrales(bandas_despues)

        print(umbrales_bandas_antes)
        print(umbrales_bandas_durante)
        print(umbrales_bandas_despues)

if __name__ == "__main__":
    main()
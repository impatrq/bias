from bci_iv_2a import MotorImageryDataset
import numpy as np
from scipy.signal import butter, filtfilt, spectrogram

antes_total03 = 0
despues_total03 = 0
durante_total03 = 0

def main():
    umbrales_instance = Umbrales()
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
    seniales03, senial_completa03, time_total03, antes_total03, durante_total03, despues_total03 = umbrales_instance.segmentar_seniales(trials[0], 3, 6)
    umbrales_instance.see_umbrales(antes_total03, durante_total03, despues_total03)

class Umbrales:
    def __init__(self):
        self.bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 12),
            'Beta': (12, 30),
            'Gamma': (30, 64)
        }

    def compute_spectrogram(self, signal, fs, window, noverlap, nfft):
        f, t, Sxx = spectrogram(signal, fs, window=window, noverlap=noverlap, nfft=nfft)
        return f, t, Sxx
    
    def apply_band_pass_filter(dself, signal, lowcut, highcut, sampling_rate, order=4):
        nyquist = 0.5 * sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal

    def spectrogram_by_band(self, signal, fs, window, noverlap, nfft):
        
        spectrograms = { 'Delta': [], 'Theta': [], 'Alpha': [], 'Beta': [], 'Gamma': [] }
        
        for band, (low, high) in self.bands.items():
            filtered_signal = self.apply_band_pass_filter(signal, low, high, fs)
            
            f, t, Sxx = self.compute_spectrogram(filtered_signal, fs, window, noverlap, nfft)
            
            # Sumar la potencia en lugar de promediar
            band_power_sum = np.sum(Sxx, axis=1)
            
            spectrograms[band].extend(band_power_sum) 
        
        return spectrograms
    
    def estadisticos2(self, matriz): 
        fft = []
        psd = []
        señales_filtradas = []

        for canal in matriz:
            fft_f = []
            fft_v = []
            psd_f = []
            psd_v = []
            señal_filtrada_canal = []

            for j in range(3):
                canal[j] -= np.mean(canal[j])

                filtered_signal = apply_low_pass_filter(canal[j], cutoff=65, sampling_rate=256)
                señal_filtrada_canal.append(filtered_signal)

                freqs_fft, fft_vals = compute_fft(filtered_signal, 256)
                fft_f.append(freqs_fft)
                fft_v.append(fft_vals)

                freqs_psd, psd_vals = compute_psd(filtered_signal, 256)
                psd_f.append(freqs_psd)
                psd_v.append(psd_vals)

            fft.append((fft_f, fft_v))  
            psd.append((psd_f, psd_v)) 
            señales_filtradas.append(señal_filtrada_canal)

        return fft, psd, señales_filtradas

    bandas_antes = { 'Delta': [], 'Theta': [], 'Alpha': [], 'Beta': [], 'Gamma': [] }
    bandas_durante = { 'Delta': [], 'Theta': [], 'Alpha': [], 'Beta': [], 'Gamma': [] }
    bandas_despues = { 'Delta': [], 'Theta': [], 'Alpha': [], 'Beta': [], 'Gamma': [] }


    lista_senales = [
        señales_filtradas03
    ]


    for señales_filtradas in lista_senales:
        for canal in range(len(señales_filtradas)): 

            spectrogram_resultsA = spectrogram_by_band(señales_filtradas[canal][0], fs=256, window='hann', noverlap=128, nfft=256)
            spectrogram_resultsC = spectrogram_by_band(señales_filtradas[canal][1], fs=256, window='hann', noverlap=128, nfft=256)
            spectrogram_resultsD = spectrogram_by_band(señales_filtradas[canal][2], fs=256, window='hann', noverlap=128, nfft=256)
            
            for band in bandas_antes:
                bandas_antes[band].extend(spectrogram_resultsA[band])  
            for band in bandas_durante:
                bandas_durante[band].extend(spectrogram_resultsC[band])
            for band in bandas_despues:
                bandas_despues[band].extend(spectrogram_resultsD[band])

    def segmentar_seniales(self, matriz, inicio, fin, fs=250):
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
import matplotlib.pyplot as plt

class graphingPythonBias:
    def graph_signal_voltage_time(self, t, signal, title):
        # Plot given signal in th time domain
        plt.figure(figsize=(12, 6))
        if signal.ndim == 1:
            plt.plot(t, signal)
        else:
            for i in range(signal.shape[0]):
                plt.plot(t, signal[i])
        plt.title(title)
        plt.xlabel("Time [s]")
        plt.ylabel("Magnitude")
        plt.grid()

    def graph_signal_voltage_frequency(self, frequencies, magnitudes, title):
        # Plot given signal in the frquency domain
        plt.figure(figsize=(12, 6))
        plt.plot(frequencies, magnitudes)
        plt.title(title)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude")
        plt.grid()

    def plot_now(self):
        plt.tight_layout()
        plt.show()

import plotext as plt

class graphingTerminalBias:
    def graph_signal_voltage_time(self, t, signal, title):
        plt.clear_data()
        plt.plot(t, signal)
        plt.title(title)
        plt.xlabel("Time [s]")
        plt.ylabel("Magnitude")
        plt.show()

    def graph_signal_voltage_frequency(self, frequencies, magnitudes, title):
        plt.clear_data()
        plt.plot(frequencies, magnitudes)
        plt.title(title)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude")
        plt.show()

def plot_now(self):
    pass
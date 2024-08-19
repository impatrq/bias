import plotext as plt

def graph_signal_voltage_time(t, signal, title):
    plt.clear_data()
    plt.plot(t, signal)
    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude")
    plt.show()

def graph_signal_voltage_frequency(frequencies, magnitudes, title):
    plt.clear_data()
    plt.plot(frequencies, magnitudes)
    plt.title(title)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.show()

def plot_now():
    pass
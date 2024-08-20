import matplotlib.pyplot as plt

def graph_signal_voltage_time(t, signal, title):
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

def graph_signal_voltage_frequency(frequencies, magnitudes, title):
    # Plot given signal in the frquency domain
    plt.figure(figsize=(12, 6))
    plt.plot(frequencies, magnitudes)
    plt.title(title)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.grid()

def plot_now():
    plt.tight_layout()
    plt.show()

'''
#import matplotlib

#matplotlib.use('Agg')  # Use the non-GUI backend

#import matplotlib.pyplot as plt
import plotext as plt

def graph_signal_voltage_time(t, signal, title, filename):
    # Plot given signal in th time domain
    #plt.figure(figsize=(12, 6))
    if signal.ndim == 1:
        plt.plot(t, signal)
    else:
        for i in range(signal.shape[0]):
            plt.plot(t, signal[i])
    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude")
    plt.grid()
    #plt.savefig(filename)  # Save the plot as an image
    #plt.close()  # Close the plot to free memory
    #print(f"Plot saved to {filename}")
    
def graph_signal_voltage_frequency(frequencies, magnitudes, title):
    # Plot given signal in the frquency domain
    #plt.figure(figsize=(12, 6))
    plt.plot(frequencies, magnitudes)
    plt.title(title)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.grid()

def plot_now():
    plt.tight_layout()
    plt.show()
'''
import reception
import extraction
import ai
import prediction
import preprocessing
import numpy as np
import graphingPython

def main():
    while True:
        show_menu()
        choice = input("Enter your choice: ")
        if choice == '1':
            n = int(input("Enter number of data points: "))
            fs = int(input("Enter sampling frequency: "))
            while True:
                number_of_channels = int(input("Enter number of channels"))
                signals = reception.get_real_combined_data(channels=number_of_channels, n=n, fs=fs, filter=True)

                for ch, signal in signals:
                    t = np.arange(len(signals[ch])) / fs
                    graphingPython.graph_signal_voltage_time(t=t, signal=np.array(signal), title="Channel {} Signal".format(ch))
                print("EEG Data Captured.")

def show_menu():
    print("EEG-based Wheelchair Control System")
    print("1. Capture EEG Data with graphing in Terminal")
    print("2. Preprocess EEG Data")
    print("3. Extract Features")
    print("4. Train Model")
    print("5. Predict Action")
    print("6. Exit")

if __name__ == "__main__":
    main()
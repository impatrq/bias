import reception
import extraction
#import ai
#import prediction
import preprocessing
import numpy as np
import graphingTerminal
from bias import Bias

def main():
    while True:
        show_menu()
        choice = input("Enter your choice: ")
        if choice == '1':
            n = int(input("Enter number of data points: "))
            fs = int(input("Enter sampling frequency: "))
            number_of_channels = int(input("Enter number of channels: "))
            while True:
                signals = reception.get_real_data(channels=number_of_channels, n=n)

                for ch, signal in signals.items():
                    t = np.arange(len(signals[ch])) / fs
                    graphingTerminal.graph_signal_voltage_time(t=t, signal=np.array(signal), title="Signal {}".format(ch))
                print("EEG data captured.")

        if choice == '2':
            n = int(input("Enter number of data points: "))
            fs = int(input("Enter sampling frequency: "))
            number_of_channels = int(input("Enter number of channels: "))
            while True:
                signal = reception.get_real_combined_data(channels=number_of_channels, n=n, fs=fs, filter=True)
                t = np.arange(len(signal)) / fs
                graphingTerminal.graph_signal_voltage_time(t=t, signal=np.array(signal), title="Signal")
                print("EEG data captured and comnbined.")

        if choice == '3':
            n = int(input("Enter number of data points: "))
            fs = int(input("Enter sampling frequency: "))
            duration = n / fs
            number_of_channels = int(input("Enter number of channels: "))
            while True:
                signals = reception.get_real_data(channels=number_of_channels, n=n)
                for ch, signal in signals.items():
                    t = np.arange(len(signals[ch])) / fs
                    graphingTerminal.graph_signal_voltage_time(t=t, signal=np.array(signal), title="Signal {}".format(ch))
                
                t0, alpha0, beta0, gamma0, delta0, theta0 = preprocessing.preprocess_signal(n=n, duration=duration, fs=fs, eeg_data=np.array(signals['ch0']))
                t1, alpha1, beta1, gamma1, delta1, theta1 = preprocessing.preprocess_signal(n=n, duration=duration, fs=fs, eeg_data=np.array(signals['ch1']))
                t2, alpha2, beta2, gamma2, delta2, theta2 = preprocessing.preprocess_signal(n=n, duration=duration, fs=fs, eeg_data=np.array(signals['ch2']))
                t3, alpha3, beta3, gamma3, delta3, theta3 = preprocessing.preprocess_signal(n=n, duration=duration, fs=fs, eeg_data=np.array(signals['ch3']))
                
                print("EEG data processed")
        
        if choice == '4':
            # Define propieties for Bias instance
            n = int(input("Enter number of data points: "))
            fs = int(input("Enter sampling frequency: "))
            number_of_channels = int(input("Enter number of channels: "))
            port = '/dev/serial0'
            baudrate = 115200
            timeout = 1
            # Create object and run the app
            biasInstance = Bias(n=n, fs=fs, channels=number_of_channels, port=port, baudrate=baudrate, timeout=timeout)
            biasInstance.app_run()

        if choice == '6':
            # Define propieties for Bias instance
            n = int(input("Enter number of data points: "))
            fs = int(input("Enter sampling frequency: "))
            number_of_channels = int(input("Enter number of channels: "))
            port = '/dev/serial0'
            baudrate = 115200
            timeout = 1
            # Create object and run the app
            biasInstance = Bias(n=n, fs=fs, channels=number_of_channels, port=port, baudrate=baudrate, timeout=timeout)
            biasInstance.train_ai_model()
         
def show_menu():
    print("EEG-based Wheelchair Control System")
    print("1. Capture EEG Data with graphing in Terminal")
    print("2. Capture and combine EEG")
    print("3. Preprocess EEG Data")
    print("4. App run")
    print("5. Extract Features")
    print("6. Train Model")
    print("7. Predict Action")
    print("8. Exit")

if __name__ == "__main__":
    main()

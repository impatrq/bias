import reception
import extraction
import ai
import prediction
import preprocessing

def main():
    while True:
        show_menu()
        choice = input("Enter your choice: ")

        if choice == '1':
            n = int(input("Enter number of data points: "))
            fs = int(input("Enter sampling frequency: "))
            number_of_channels = int(input("Enter number of channels"))
            real_eeg_signal = reception.get_real_combined_data(channels=number_of_channels, n=n, fs=fs, filter=False)
            print("EEG Data Captured.")
        elif choice == '2':
            if 'real_eeg_signal' in locals():
                duration = int(input("Enter duration in seconds: "))
                t, alpha, beta, gamma, delta, theta = preprocessing.preprocess_signal(n=n, duration=duration, fs=fs, eeg_data=real_eeg_signal)
                print("Preprocessing Complete.")
            else:
                print("No EEG data found. Capture data first.")
        elif choice == '3':
            if 'real_eeg_signal' in locals():
                labels = ['forward', 'backward', 'left', 'right', 'stop', 'rest']
                n = int(input("Enter number of data points: "))
                duration = int(input("Enter duration in seconds: "))
                fs = int(input("Enter sampling frequency: "))
                df = extraction.extract_data(real_eeg_signal, labels, n, duration, fs, online=True)
                df.to_csv('extracted_features.csv', index=False)
                print("Feature Extraction Complete.")
            else:
                print("No EEG data found. Capture data first.")
        elif choice == '4':
            n = int(input("Enter number of data points: "))
            duration = int(input("Enter duration in seconds: "))
            fs = int(input("Enter sampling frequency: "))
            ai.main(n, duration, fs, online=True) # Check this
        elif choice == '5':
            n = int(input("Enter number of data points: "))
            duration = int(input("Enter duration in seconds: "))
            fs = int(input("Enter sampling frequency: "))
            prediction.main(n, duration, fs) # Check this
        elif choice == '6':
            break
        else:
            print("Invalid choice. Please try again.")

def show_menu():
    print("EEG-based Wheelchair Control System")
    print("1. Capture EEG Data")
    print("2. Preprocess EEG Data")
    print("3. Extract Features")
    print("4. Train Model")
    print("5. Predict Action")
    print("6. Exit")

if __name__ == "__main__":
    main()

'''
import ai
import prediction

FS = 500  # Sampling frequency
N = 1000  # Number of samples
DURATION = N / FS

def main():
    while True:
        print("1. Train Model")
        print("2. Make Single Prediction")
        print("3. Real-Time Prediction")
        print("4. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            ai.main(N, DURATION, FS)
        elif choice == '2':
            prediction.make_prediction(N, DURATION, FS)
        elif choice == '3':
            prediction.main(N, DURATION, FS)
        elif choice == '4':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
'''
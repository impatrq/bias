import reception
import preprocessing
import extraction
import ia
import prediction
import time

def train_model():
    real_eeg_signal = reception.get_real_data()
    X, y = ia.prepare_data(real_eeg_signal)
    X_train, X_test, y_train, y_test = ia.split_data(X, y)
    X_train, X_test, scaler = ia.standardize_data(X_train, X_test)
    model = ia.build_model(X_train.shape[1], y_train.shape[1])
    model = ia.train_model(model, X_train, y_train, X_test, y_test)
    ia.evaluate_model(model, X_test, y_test)
    ia.save_model_and_scaler(model, scaler)

def make_prediction():
    real_eeg_signal = reception.get_real_data()
    model, scaler = prediction.load_model_and_scaler()
    prediction_result = prediction.classify_eeg(model, scaler, real_eeg_signal)
    print(f'Predicted class: {prediction_result}')

def real_time_prediction():
    model, scaler = prediction.load_model_and_scaler()
    
    while True:
        real_eeg_signal = reception.get_real_data()
        prediction_result = prediction.classify_eeg(model, scaler, real_eeg_signal)
        print(f'Predicted class: {prediction_result}')
        
        # Add a small delay to control the prediction rate
        time.sleep(1)  # Adjust this value as needed


def main():
    while True:
        print("1. Train Model")
        print("2. Make Single Prediction")
        print("3. Real-Time Prediction")
        print("4. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            train_model()
        elif choice == '2':
            make_prediction()
        elif choice == '3':
            real_time_prediction()
        elif choice == '4':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
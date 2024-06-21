import extraction
import preprocessing
from keras.models import load_model # type: ignore
import joblib
import reception
import pandas as pd

def load_model_and_scaler():
    try:
        model = load_model('wheelchair_model.h5')
        scaler = joblib.load('scaler.save')
        return model, scaler
    except Exception as e:
        print(f"Error loading model and scaler: {e}")
        raise

def classify_eeg(model, scaler, eeg_data, n, duration, fs):
    try:
        t, alpha, beta, gamma, delta, theta = preprocessing.preprocess_signal(eeg_data, n, duration, fs)
        features = extraction.extract_features(alpha, beta, gamma, delta, theta)
        features_df = pd.DataFrame([features])
        features_scaled = scaler.transform(features_df)
        prediction = model.predict(features_scaled)
        predicted_class = prediction.argmax(axis=1)
        return predicted_class
    except Exception as e:
            print(f"Error classifying EEG data: {e}")
            raise

def make_prediction(n, duration, fs):
    try:
        real_eeg_signal = reception.get_real_combined_data(n, fs, filter=False)
        model, scaler = load_model_and_scaler()
        prediction_result = classify_eeg(model, scaler, real_eeg_signal, n, duration, fs)
        print(f'Predicted class: {prediction_result}')

    except Exception as e:
        print(f"Error making prediction: {e}")
        raise


def main(n=1000, duration=2, fs=500):
    try:
        model, scaler = load_model_and_scaler()
        while True:
            real_eeg_signal = reception.get_real_combined_data(n, fs, filter=False)
            prediction = classify_eeg(model, scaler, real_eeg_signal, n, duration, fs)
            print(f'Predicted class: {prediction}')
    
    except KeyboardInterrupt:
        print("Prediction loop terminated by user.")

    except Exception as e:
        print(f"Error in prediction main: {e}")
        raise

if __name__ == "__main__":
    main()
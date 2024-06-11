import extraction
import preprocessing
from keras.models import load_model
import joblib
import reception
import pandas as pd

def load_model_and_scaler():
    model = load_model('wheelchair_model.h5')
    scaler = joblib.load('scaler.save')
    return model, scaler

def classify_eeg(model, scaler, eeg_data, n, duration, fs):
    t, alpha, beta, gamma, delta, theta = preprocessing.preprocess_signal(eeg_data, n, duration, fs)
    features = extraction.extract_features(alpha, beta, gamma, delta, theta)
    features_df = pd.DataFrame([features])
    features_scaled = scaler.transform(features_df)
    prediction = model.predict(features_scaled)
    predicted_class = prediction.argmax(axis=1)
    return predicted_class

def make_prediction(n, duration, fs):
    real_eeg_signal = reception.get_real_data(n, fs)
    model, scaler = load_model_and_scaler()
    prediction_result = classify_eeg(model, scaler, real_eeg_signal, n, duration, fs)
    print(f'Predicted class: {prediction_result}')


def main(n=1000, duration=2, fs=500):
    model, scaler = load_model_and_scaler()
    while True:
        real_eeg_signal = reception.get_real_data(n, fs)
        prediction = classify_eeg(model, scaler, real_eeg_signal, n, duration, fs)
        print(f'Predicted class: {prediction}')

if __name__ == "__main__":
    main()
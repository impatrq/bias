import extraction
import ia
import preprocessing
from keras.models import load_model
import joblib
import reception
import pandas as pd

def load_model_and_scaler():
    model = load_model('wheelchair_model.h5')
    scaler = joblib.load('scaler.save')
    return model, scaler

def classify_eeg(model, scaler, eeg_data):
    t, alpha, beta, gamma, delta, theta = preprocessing.preprocess_signal(eeg_data)
    features = extraction.extract_features(alpha, beta, gamma, delta, theta)
    features_df = pd.DataFrame([features])
    features_scaled = scaler.transform(features_df)
    prediction = model.predict(features_scaled)
    predicted_class = prediction.argmax(axis=1)
    return predicted_class

def main():
    model, scaler = load_model_and_scaler()
    real_eeg_signal = reception.main()
    prediction = classify_eeg(model, scaler, real_eeg_signal)
    print(f'Predicted class: {prediction}')

if __name__ == "__main__":
    main()
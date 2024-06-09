import extraction
import ia
import preprocessing
from keras.models import load_model
import joblib

model = load_model('wheelchair_model.h5')
scaler = joblib.load('scaler.save')

def classify_signal(model, scaler, features):
    features_scaled = scaler.transform([features])
    return model.predict(features_scaled)[0]

# Simulate real-time signal processing
def real_time_prediction(eeg_data):
    t, alpha, beta, gamma, delta, theta = eeg_data
    features = extraction.extract_features(alpha, beta, gamma, delta, theta)
    movement_command = classify_signal(model, scaler, features)
    send_command_to_wheelchair(movement_command)

# Example usage
eeg_data = preprocessing.preprocess_signal()  # Replace with actual real-time data
real_time_prediction(eeg_data)

def send_command_to_wheelchair(command):
    # Send the command to the wheelchair
    print(f'Sending command to wheelchair: {command}')

if __name__ == "__main__":
    eeg_data = preprocessing.preprocess_signal()
    real_time_prediction(eeg_data)

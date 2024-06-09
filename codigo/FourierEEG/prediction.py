import extraction
import ia
import preprocessing
from keras.models import load_model
import joblib

def load_model_and_scaler():
    model = load_model('wheelchair_model.h5')
    scaler = joblib.load('scaler.save')
    return model, scaler

def classify_signal(model, scaler, features):
    features_scaled = scaler.transform([features])
    return model.predict(features_scaled)[0]

# Simulate real-time signal processing
def real_time_prediction(eeg_data, model, scaler):
    t, alpha, beta, gamma, delta, theta = eeg_data
    features = extraction.extract_features(alpha, beta, gamma, delta, theta)
    movement_command = classify_signal(model, scaler, features)
    send_command_to_wheelchair(movement_command)

def send_command_to_wheelchair(command):
    movements = ['forward', 'backward', 'left', 'right', 'stop']
    command_index = command.argmax()
    print(f'Sending command to wheelchair: {movements[command_index]}')

def main():
    model, scaler = load_model_and_scaler()
    eeg_data = preprocessing.preprocess_signal()  # Replace with actual real-time data
    real_time_prediction(eeg_data, model, scaler)

if __name__ == "__main__":
    main()
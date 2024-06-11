import time
import reception
import preprocessing
import extraction
import pickle
import pandas as pd

# Duration to capture the EEG signal (in seconds)
CAPTURE_DURATION = 5  # Adjust this based on your requirements

def main():
    print("Starting EEG signal capture...")
    start_time = time.time()

    # Capture the EEG signal for the specified duration
    real_eeg_signal = reception.capture_signal(duration=CAPTURE_DURATION)
    print("EEG signal capture completed.")

    # Preprocess the captured EEG signal
    t, alpha, beta, gamma, delta, theta = preprocessing.preprocess_signal(real_eeg_signal)

    # Extract features from the preprocessed signals
    features = extraction.extract_features(alpha, beta, gamma, delta, theta)

    # Load the trained model
    with open('trained_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Convert the features to a DataFrame for prediction
    features_df = pd.DataFrame([features])

    # Predict the movement using the trained model
    prediction = model.predict(features_df)
    print(f"Predicted movement: {prediction[0]}")

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
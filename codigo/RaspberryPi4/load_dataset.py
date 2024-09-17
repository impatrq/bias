import mne
import os

def load_bci_iv_2a_data(subject=1, run=1):
    dataset_path = "BCI_IV_2a"  # Change this to your unzipped dataset folder
    file_name = f"A0{subject}T.gdf"  # For training data, or use A0{subject}E.gdf for evaluation

    file_path = os.path.join(dataset_path, file_name)
    # file_path = f"{dataset_path}/{file_name}"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist. Make sure the dataset is downloaded and extracted.")

    # Load the GDF file using MNE
    raw = mne.io.read_raw_gdf(file_path, preload=True)
    
    # Set EEG reference
    raw.set_eeg_reference('average')
    
    return raw

def main():
    # Load data for subject 1 and run 1
    raw = load_bci_iv_2a_data(subject=1, run=1)

    # Filter and preprocess the raw data
    raw.filter(7., 30., fir_design='firwin')

    # Load a standard montage (10-20 system for electrode placement)
    montage = mne.channels.make_standard_montage('standard_1020')

    # Create a mapping dictionary between your dataset's channels and standard ones
    mapping = {
        'EEG-Fz': 'Fz', 'EEG-0': 'FT7', 'EEG-1': 'FC3', 'EEG-2': 'FCz', 'EEG-3': 'FC4', 'EEG-4': 'FT8', 'EEG-5': 'T7', 
        'EEG-C3': 'C3', 'EEG-6': 'C1', 'EEG-Cz': 'Cz', 'EEG-7': 'C2', 'EEG-C4': 'C4', 'EEG-8': 'T8', 'EEG-9': 'TP7', 
        'EEG-10': 'CP3', 'EEG-11': 'CPz', 'EEG-12': 'CP4', 'EEG-13': 'TP8', 'EEG-14': 'P7', 'EEG-Pz': 'Pz', 'EEG-15': 'P8', 
        'EEG-16': 'Oz', 'EOG-left': 'A1', 'EOG-central': 'Fpz', 'EOG-right': 'A2'
        # Add other mappings if known
    }

    # Rename channels in your raw data
    raw.rename_channels(mapping)

    # Apply montage to your raw EEG data
    raw.set_montage(montage, on_missing='ignore')

    # New method to compute and plot PSD
    psd_fig = raw.compute_psd(fmin=7, fmax=30)
    psd_fig.plot()

    # Plot sensor locations to verify montage correctness
    sensor_fig = raw.plot_sensors(show_names=True)
    sensor_fig.show()  # Display the sensor locations plot

if __name__ == "__main__":
    main()
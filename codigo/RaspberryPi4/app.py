import numpy as np
from bias import BiasClass
from bias_reception import ReceptionBias
from bias_dsp import FilterBias, ProcessingBias
from bias_motors import MotorBias
from bias_ai import AIBias
from bias_graphing import GraphingBias
from signals import generate_synthetic_eeg, generate_synthetic_eeg_bandpower

def main():
    while True:
        # Show the menu
        show_menu()
        choice = input("Enter your choice: ")

        if choice == '1':
            # Set parameters
            n = int(input("Enter number of data points: "))
            fs = int(input("Enter sampling frequency: "))
            number_of_channels = int(input("Enter number of channels: "))
            port = '/dev/serial0'
            baudrate = 115200
            timeout = 1 

            # Receive eeg data
            biasReception = ReceptionBias(port=port, baudrate=baudrate, timeout=timeout)
            biasGraphing = GraphingBias(graph_in_terminal=True)

            real_data = input("Do you want to get real data? (y/n): ")
            if real_data.lower().strip() == 'y':
                signals = biasReception.get_real_data(channels=number_of_channels, n=n)
            else:
                signals = generate_synthetic_eeg(n_samples=n, n_channels=number_of_channels, fs=fs)

            # Graph signals
            for ch, signal in signals.items():
                t = np.arange(len(signals[ch])) / fs
                biasGraphing.graph_signal_voltage_time(t=t, signal=np.array(signal), title="Signal {}".format(ch))

            # Plot
            biasGraphing.plot_now()

        if choice == '2':
            # Set parameters
            n = int(input("Enter number of data points: "))
            fs = int(input("Enter sampling frequency: "))
            number_of_channels = int(input("Enter number of channels: "))
            port = '/dev/serial0'
            baudrate = 115200
            timeout = 1
            
            # Define objects that will be used
            biasReception = ReceptionBias(port=port, baudrate=baudrate, timeout=timeout)
            biasGraphing = GraphingBias(graph_in_terminal=True)
            biasFilter = FilterBias(n=n, fs=fs, notch=True, bandpass=True, fir=False, iir=False)

            # Generate data
            real_data = input("Do you want to get real data? (y/n): ")

            if real_data.lower().strip() == "y":
                signals = biasReception.get_real_data(channels=number_of_channels, n=n)
            else:
                signals = generate_synthetic_eeg(n_samples=n, n_channels=number_of_channels, fs=fs)

            # Graph signals
            for ch, signal in signals.items():
                t = np.arange(len(signals[ch])) / fs
                biasGraphing.graph_signal_voltage_time(t=t, signal=np.array(signal), title="Signal {}".format(ch))

            # Apply digital filtering
            filtered_data = biasFilter.filter_signals(signals)

            # Calculate the time vector
            t = np.linspace(0, n/fs, n, endpoint=False)
            
            
            # Graph signals
            for ch, signal in filtered_data.items():
                # Graph filtered signal
                biasGraphing.graph_signal_voltage_time(t=t, signal=signal, title="Filtered Signal {}".format(ch))

            # Plot
            biasGraphing.plot_now()

        if choice == '3':
            # Set parameters
            n = int(input("Enter number of data points: "))
            fs = int(input("Enter sampling frequency: "))
            number_of_channels = int(input("Enter number of channels: "))
            port = '/dev/serial0'
            baudrate = 115200
            timeout = 1

            # Define objects that will be used
            biasReception = ReceptionBias(port=port, baudrate=baudrate, timeout=timeout)
            biasGraphing = GraphingBias(graph_in_terminal=True)
            biasFilter = FilterBias(n=n, fs=fs, notch=True, bandpass=True, fir=False, iir=False)
            biasProcessing = ProcessingBias(n=n, fs=fs)

            # Generate data
            real_data = input("Do you want to get real data? (y/n): ")

            if real_data.lower().strip() == "y":
                signals = biasReception.get_real_data(channels=number_of_channels, n=n)
            else:
                signals = generate_synthetic_eeg(n_samples=n, n_channels=number_of_channels, fs=fs)

            # Graph signals
            for ch, signal in signals.items():
                t = np.arange(len(signals[ch])) / fs
                biasGraphing.graph_signal_voltage_time(t=t, signal=np.array(signal), title="Signal {}".format(ch))

            # Apply digital filtering
            filtered_data = biasFilter.filter_signals(eeg_signals=signals)

            # Calculate the time vector
            t = np.linspace(0, n/fs, n, endpoint=False)
            
            # Graph signals
            for ch, signal in filtered_data.items():
                # Graph filtered signal
                biasGraphing.graph_signal_voltage_time(t=t, signal=signal, title="Filtered Signal {}".format(ch))

            # Process data
            times, eeg_signals = biasProcessing.process_signals(eeg_signals=filtered_data)

            # Plot 4 signals with its resepctive bands
            for ch, signals in eeg_signals.items():
                # Plot the interpolated signals
                for band_name, wave in signals.items():
                    biasGraphing.graph_signal_voltage_time(t=times[ch], signal=wave, title=f"{band_name.capitalize()} interpolated. {ch}")
            
            # Plot
            biasGraphing.plot_now()
            
        if choice == '4':
            # Set parameters
            n = int(input("Enter number of data points: "))
            fs = int(input("Enter sampling frequency: "))
            number_of_channels = int(input("Enter number of channels: "))
            port = '/dev/serial0'
            baudrate = 115200
            timeout = 1

            commands = input("Write commands (separated by commas): ")

            # Split the input string by commas and convert to a list
            command_list = [cmd.strip() for cmd in commands.split(",")]

            saved_dataset_path = None
            save_path = None
            
            # Get the user's input desires
            loading_dataset = input("Do you want to load an existing dataset? (y/n): ")
            if loading_dataset.lower() == "y":
                saved_dataset_path = input("Enter the name of the file where dataset was saved (without extension): ")
            else:
                save_new_dataset = input("Do you want to save the new dataset? (y/n): ")
                if save_new_dataset.lower() == "y":
                    save_path = input("Enter the path where you want to save the dataset (without extension): ")

            # Create BiasClass instance
            biasInstance = BiasClass(n=n, fs=fs, channels=number_of_channels, port=port, baudrate=baudrate, timeout=timeout, save_path=save_path, saved_dataset_path=saved_dataset_path, model_name=None, commands=command_list)

            biasInstance.train_ai_model()

        if choice == '5':
            # Set parameters
            n = int(input("Enter number of data points: "))
            fs = int(input("Enter sampling frequency: "))
            number_of_channels = int(input("Enter number of channels: "))
            port = '/dev/serial0'
            baudrate = 115200
            timeout = 1

            save_path = None
            saved_dataset_path = None
            model_name = None
            commands = input("Write commands (separated by commas): ")

            # Split the input string by commas and convert to a list
            command_list = [cmd.strip() for cmd in commands.split(",")]

            # Define objects to use
            biasReception = ReceptionBias(port=port, baudrate=baudrate, timeout=timeout)
            biasFilter = FilterBias(n=n, fs=fs, notch=True, bandpass=True, fir=False, iir=False)
            biasProcessing = ProcessingBias(n=n, fs=fs)
            biasAI = AIBias(n=n, fs=fs, channels=number_of_channels, commands=command_list)

            # Get the user's input desires
            model_lt = input("Do you want to load or train a model (l/t): ")
            if model_lt.lower() == "t":
                training_real_data = False
                loading_dataset = input("Do you want to load a existent dataset? (y/n): ")
                if loading_dataset.lower() == "y":
                    saved_dataset_path = input("Write the name of the file where dataset was saved: ")
                else:
                    # Generate data
                    want_real_data = input("Do you want to get real data? (y/n): ")

                    if want_real_data.lower().strip() == "y":
                        training_real_data = True
                    else:
                        training_real_data = False

                    save_new_dataset = input("Do you want to save the new dataset? (y/n): ")
                    if save_new_dataset == "y":
                        save_path = input("Write the path where you want to save the dataset: ")

                biasAI.collect_and_train(reception_instance=biasReception, filter_instance=biasFilter, processing_instance=biasProcessing, 
                                 trials_per_command=1, save_path=save_path, saved_dataset_path=saved_dataset_path, training_real_data=training_real_data)

            # Load an existent model
            elif model_lt.lower() == 'l':
                model_name = input("Write the filname where model is saved: ")
                print("Charging model")

            # Generate data
            real_data = input("Do you want to get real data? (y/n): ")

            if real_data.lower().strip() == "y":
                signals = biasReception.get_real_data(channels=number_of_channels, n=n)
            else:
                signals = generate_synthetic_eeg(n_samples=n, n_channels=number_of_channels, fs=fs)

            # Filter data
            filtered_data = biasFilter.filter_signals(signals)
            # Process data
            times, eeg_signals = biasProcessing.process_signals(eeg_signals=filtered_data)
            # Predict command
            predicted_command = biasAI.predict_command(eeg_data=eeg_signals)
            print(f"Command: {predicted_command}")

        if choice == '6':
            # Set parameters
            run_motor_control()

        if choice == '7':
            # Set parameters
            n = int(input("Enter number of data points: "))
            fs = int(input("Enter sampling frequency: "))
            number_of_channels = int(input("Enter number of channels: "))
            port = '/dev/serial0'
            baudrate = 115200
            timeout = 1

            # Initialize everything as None
            save_path = None
            saved_dataset_path = None
            model_name = None

            # Generate data
            want_real_data = input("Do you want to get real data? (y/n): ")

            if want_real_data.lower().strip() == "y":
                real_data = True
            else:
                real_data = False

            # See the user's commands
            commands = input("Write commands (separated by commas): ")

            # Split the input string by commas and convert to a list
            command_list = [cmd.strip() for cmd in commands.split(",")]

            # Get the user's desire
            model_lt = input("Do you want to load or train a model (l/t): ")
            if model_lt.lower() == "t":
                loading_dataset = input("Do you want to load a existent dataset? (y/n): ")
                if loading_dataset.lower() == "y":
                    saved_dataset_path = input("Write the name of the file where dataset was saved: ")
                else:
                    save_new_dataset = input("Do you want to save the new dataset? (y/n): ")
                    if save_new_dataset == "y":
                        save_path = input("Write the path where you want to save the dataset: ")

            # Load an existent model
            elif model_lt.lower() == 'l':
                model_name = input("Write the filname where model is saved: ")
                print("Charging model")

            biasInstance = BiasClass(n=n, fs=fs, channels=number_of_channels, port=port, baudrate=baudrate, timeout=timeout, save_path=save_path, saved_dataset_path=saved_dataset_path, model_name=model_name, commands=command_list)
            biasInstance.app_run(real_data=real_data)
         
def show_menu():
    print("EEG-based Wheelchair Control System")
    print("1. Capture EEG Data with graphing in Terminal")
    print("2. Capture and filter signals")
    print("3. Capture, filter and process signals") 
    print("4. Train Model")
    print("5. Predict Action")
    print("6. Motor movement")
    print("7. App run")
    print("8. Exit")

def run_motor_control():
    # Define MotorBias class
    biasMotor = MotorBias(
        echo_forward=18, trigger_forward=17,
        echo_backwards=23, trigger_backwards=22,
        echo_right=5, trigger_right=6,
        echo_left=25, trigger_left=24,
        led_forward=16, led_backwards=20,
        led_left=21, led_right=26,
        buzzer=12, motor1_in1=13,
        motor1_in2=19, motor2_in1=7,
        motor2_in2=8
    )

    while True:
        # Move if command is possible
        command = input("Enter command (forward/left/backwards/right/stop): ").strip()
        biasMotor.move_if_possible(command)

if __name__ == "__main__":
    main()

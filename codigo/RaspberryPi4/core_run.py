import time
from bias_graphing import GraphingBias
from bias_motors import MotorBias
from bias_reception import ReceptionBias
from bias_dsp import FilterBias, ProcessingBias
from signals import generate_synthetic_eeg, generate_synthetic_eeg_bandpower

def task_motors():
    biasMotor = MotorBias(
        echo_forward=18, trigger_forward=17, echo_backwards=23, trigger_backwards=22,
        echo_right=5, trigger_right=6, echo_left=25, trigger_left=24,
        led_forward=16, led_backwards=20, led_left=21, led_right=26,
        buzzer=12, motor1_in1=13, motor1_in2=19, motor2_in1=7, motor2_in2=8
    )
    biasMotor.brake()

    command = input("Enter command (forward/left/backwards/right/stop): ").strip()
    biasMotor.move_if_possible(command)
    print("Motor task executed")

def task_reception():
    print("Executing reception task")

    n = 1000
    fs = 500
    number_of_channels = 4
    port = '/dev/serial0'
    baudrate = 115200
    timeout = 1

    biasReception = ReceptionBias(port=port, baudrate=baudrate, timeout=timeout)

    real_data = input("Do you want to get real data? (y/n): ")

    if real_data.lower().strip() == "y":
        signals = biasReception.get_real_data(n=n, channels=number_of_channels)
    else:
        signals = generate_synthetic_eeg(n_samples=n, n_channels=number_of_channels, fs=fs)

    biasGraphing = GraphingBias(graph_in_terminal=True)
    for ch, signal in signals.items():
        t = np.arange(len(signals[ch])) / fs
        biasGraphing.graph_signal_voltage_time(t=t, signal=np.array(signal), title="Signal {}".format(ch))
    
    print("Reception task executed")

def task_dsp():
    n = 1000
    fs = 500
    number_of_channels = 4
    duration = n / fs

    number_of_channels = 4
    port = '/dev/serial0'
    baudrate = 115200
    timeout = 1

    # Receive data
    biasReception = ReceptionBias(port=port, baudrate=baudrate, timeout=timeout)

    # Generate data
    real_data = input("Do you want to get real data? (y/n): ")

    if real_data.lower().strip() == "y":
        signals = biasReception.get_real_data(n=n, channels=number_of_channels)
    else:
        signals = generate_synthetic_eeg(n_samples=n, n_channels=number_of_channels, fs=fs)

    # Graph signals
    biasGraphing = GraphingBias(graph_in_terminal=True)
    for ch, signal in signals.items():
        t = np.arange(len(signals[ch])) / fs
        biasGraphing.graph_signal_voltage_time(t=t, signal=np.array(signal), title="Signal {}".format(ch))

    # Apply digital filtering
    biasFilter = FilterBias(n=n, fs=fs, notch=True, bandpass=True, fir=False, iir=False)
    filtered_data = biasFilter.filter_signals(eeg_signals=signals)

    # Calculate the time vector
    t = np.linspace(0, duration, n, endpoint=False)

    for ch, signal in filtered_data.items():
        # Graph filtered signal
        biasGraphing.graph_signal_voltage_time(t=t, signal=signal, title="Filtered Signal {}".format(ch))

    # Process data
    biasProcessing = ProcessingBias(n=n, fs=fs)
    signals = biasProcessing.process_signals(eeg_signals=filtered_data)

    # Plot
    biasGraphing.plot_now()

def task_AI():
    print("Executing AI task")

    n = 1000
    fs = 500
    number_of_channels = 4
    port = '/dev/serial0'
    baudrate = 115200
    timeout = 1
    biasReception = ReceptionBias(port, baudrate, timeout)
    biasFilter = FilterBias(n=n, fs=fs, notch=True, bandpass=True, fir=False, iir=False)
    biasProcessing = ProcessingBias(n=n, fs=fs)
    commands = ["forward", "backwards", "left", "right"] #, "stop", "rest"]
    biasAI = AIBias(n=n, fs=fs, channels=number_of_channels, commands=commands)

    model_lt = input("Do you want to load or train a model (l/t): ")
    if model_lt.lower() == "t":
        save_path = None
        saved_dataset_path = None

        training_real_data = False
        loading_dataset = input("Do you want to load an existent dataset? (y/n): ")
        if loading_dataset.lower() == "y":
            saved_dataset_path = input("Write the name of the file where dataset was saved: ")
        else:
            want_real_data = input("Do you want to train it with real data? (y/n): ")

            if want_real_data.lower().strip() == "y":
                training_real_data = True
            else:
                training_real_data = False

            save_new_dataset = input("Do you want to save the new dataset? (y/n): ")
            if save_new_dataset == "y":
                save_path = input("Write the path where you want to save the dataset: ")

        biasAI.collect_and_train(reception_instance=biasReception, filter_instance=biasFilter, processing_instance=biasProcessing, 
                            trials_per_command=1, save_path=save_path, saved_dataset_path=saved_dataset_path, training_real_data=training_real_data)

    elif model_lt.lower() == 'l':
        model_name = input("Write the filname where model is saved: ")
        print("Charging model")

    signals = generate_synthetic_eeg_bandpower(n_samples=n, n_channels=number_of_channels, fs=fs, command="left")

    filtered_data = biasFilter.filter_signals(eeg_signals=signals)

    times, eeg_signals = biasProcessing.process_signals(eeg_signals=filtered_data)
    predicted_command = biasAI.predict_command(eeg_data=eeg_signals)
    print(f"Predicted Command: {predicted_command}")

    print("AI task executed")

if __name__ == "__main__":
    while True:
        # Ejecuta la tarea de recepción
        task_reception()
        time.sleep(1)  # Espera 1 segundo antes de la siguiente tarea

        task_dsp()
        time.sleep(1)  # Espera 1 segundo antes de la siguiente tarea

        # Ejecuta la tarea de IA
        task_AI()
        time.sleep(1)  # Espera 1 segundo antes de la siguiente tarea

        # Ejecuta la tarea de motores
        task_motors()
        time.sleep(1)  # Espera 1 segundo antes de la siguiente iteración

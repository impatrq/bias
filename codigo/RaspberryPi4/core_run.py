import multiprocessing
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
    biasMotor.brake()  # Inicializa el motor

    while True:
        # Puedes reemplazar esto con otra lógica para recibir comandos, o usar un bucle infinito
        command = input("Enter command (forward/left/backwards/right/stop): ").strip()
        biasMotor.move_if_possible(command)
    # print("Executing motor task")

def task_reception():

    print("Executing reception task")

def task_AI():
    
    print("Executing AI task")

if __name__ == "__main__":
    # Crear procesos para cada tarea
    process_motors = multiprocessing.Process(target=task_motors)
    process_reception = multiprocessing.Process(target=task_reception)
    process_AI = multiprocessing.Process(target=task_AI)

    # Iniciar los procesos
    process_motors.start()
    process_reception.start()
    process_AI.start()

    # Esperar a que los procesos terminen
    process_motors.join()
    process_reception.join()
    process_AI.join()

'''
def app_run(self, real_data):
        if self._model_name is None:
            self.train_ai_model()

        self._biasMotor.brake()

        while True:
            # Receive eeg data
            if real_data:
                signals = self._biasReception.get_real_data(n=self._n, channels=self._number_of_channels)
            else:
                signals = generate_synthetic_eeg(n_samples=self._n, n_channels=self._number_of_channels, fs=self._fs)

            # Graph signals
            for ch, signal in signals.items():
                t = np.arange(len(signals[ch])) / self._fs
                self._biasGraphing.graph_signal_voltage_time(t=t, signal=np.array(signal), title="Signal {}".format(ch))

            # Apply digital filtering
            filtered_data = self._biasFilter.filter_signals(eeg_signals=signals)

            # Calculate the time vector
            t = np.linspace(0, self._duration, self._n, endpoint=False)
            
            # Graph signals
            for ch, signal in filtered_data.items():
                # Graph filtered signal
                self._biasGraphing.graph_signal_voltage_time(t=t, signal=signal, title="Filtered Signal {}".format(ch))

            # Process data
            times, eeg_signals = self._biasProcessing.process_signals(eeg_signals=filtered_data)

            
            # Plot 4 signals with its resepctive bands
            for ch, signals in eeg_signals.items():
                # Plot the interpolated signals
                for band_name, wave in signals.items():
                    self._biasGraphing.graph_signal_voltage_time(t=times[ch], signal=wave, title=f"{band_name.capitalize()} interpolated. {ch}")
            
            # Plot
            self._biasGraphing.plot_now()
            

            command = self._biasAI.predict_command(eeg_data=eeg_signals)
            print(f"command: {command}")

            self._biasMotor.move_if_possible(command)
'''
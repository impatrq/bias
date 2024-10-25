# En este programa se alojan el DSP y el Motors
import multiprocessing
import time
import random
import numpy as np
import scipy.interpolate
from scipy.signal import butter, filtfilt, firwin, lfilter, iirfilter
from gpiozero import DistanceSensor, PWMLED, Buzzer, PWMOutputDevice
from gpiozero.pins.pigpio import PiGPIOFactory

class DSPBias:
    def __init__(self, n, fs):
        self._n = n
        self._fs = fs
        self._duration = self._n / self._fs

# Process signals
class ProcessingBias(DSPBias):
    # Constructor
    def __init__(self, n, fs):
        super().__init__(n, fs)
        self._biasGraphing = GraphingBias(graph_in_terminal=True)

    # Process all the data
    def process_signals(self, eeg_signals):
        processed_signals = {}
        times = {}

        # Process each signal in each channel for processing
        for ch, signal in eeg_signals.items():
            t, processed_signal = self.preprocess_signal(np.array(signal))
            processed_signals[ch] = processed_signal
            times[ch] = t

        return times, processed_signals

    # Process one signal in particular
    def preprocess_signal(self, eeg_signal):
        # Time vector
        t = np.linspace(0, self._duration, self._n, endpoint=False)

        # Check that eeg_signal is a numpy array
        if isinstance(eeg_signal, np.ndarray):
            # Injection of real data
            signal = eeg_signal

        else:
            raise ValueError("Unsupported data format")

        # Make Fourier transform
        signal_fft, frequencies, signal_fft_magnitude = self.do_fft(signal)

        # Eliminate the range of negative frequencies for original signal
        signal_fft_reduced = signal_fft[:self._n//2]
        frequencies_reduced = frequencies[:self._n//2]
        signal_fft_magnitude_reduced = signal_fft_magnitude[:self._n//2]

        # Graph signal in frequency and in time domain
        #self._biasGraphing.graph_signal_voltage_time(t=t, signal=signal, title=f"Input signal {channel_number}")
        #self._biasGraphing.graph_signal_voltage_frequency(frequencies=frequencies_reduced, magnitudes=signal_fft_magnitude_reduced, title=f'Frequency spectrum of signal of {channel_number}')

        # EEG bands
        bands = {
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 100),
            "delta": (0.5, 4),
            "theta": (4, 8)
        }

        filtered_signals = {}

        # Reconstruct the negative part of signals
        for band_name, band_range in bands.items():
            # Reconstruct and then apply Fourier in order to get the five signals over time
            filtered_signals[band_name] = self.filter_and_reconstruct(signal_fft, frequencies, band_range)

        # Add signal dimention
        filtered_signals["signal"] = signal

        # New sampling rate for interpolation
        new_fs = self._fs * 10
        new_t = np.linspace(0, self._duration, int(self._duration * new_fs), endpoint=True)

        # Interpolate each wave
        interpolated_signals = {band_name: self.interpolate_signal(t, sig.real, new_t) for band_name, sig in filtered_signals.items()}

        # Return time vector and the signals already processed
        return new_t, interpolated_signals

    def do_fft(self, signal):
        # Apply Fourier Transform for the original signal
        signal_fft = np.fft.fft(signal)
        frequencies = np.fft.fftfreq(self._n, d=1/self._fs)
        signal_fft_magnitude = np.abs(signal_fft) / self._n

        return signal_fft, frequencies, signal_fft_magnitude

    def filter_and_reconstruct(self, signal_fft, frequencies, band):
        # Filter each band with the corresponding wave
        filtered_fft = np.zeros_like(signal_fft)
        band_indices = np.where((frequencies >= band[0]) & (frequencies <= band[1]))
        filtered_fft[band_indices] = signal_fft[band_indices] 
        # Reconstruct signal in the negative side
        filtered_fft[-band_indices[0]] = signal_fft[-band_indices[0]]
        # Apply inverse Fourier transform in order to obtain a signal in the time domain
        filtered_signal = np.fft.ifft(filtered_fft)
        return filtered_signal.real

    def interpolate_signal(self, t, signal, new_t):
        # Clip new_t to the range of t to avoid out-of-bounds values
        new_t_clipped = np.clip(new_t, t.min(), t.max())

        # Interpolate the signal to new time points
        interpolated_signal = scipy.interpolate.interp1d(t, signal, kind='cubic')(new_t_clipped)
        return interpolated_signal

class FilterBias(DSPBias):
    # Constructor
    def __init__(self, n, fs, notch, bandpass, fir, iir):
        self._notch = notch
        self._bandpass = bandpass
        self._fir = fir
        self._iir = iir
        super().__init__(n=n, fs=fs)

    # Filter all the signals
    def filter_signals(self, eeg_signals):
        filtered_signals = {}
        # Filter by signals in each channel
        for ch, signal in eeg_signals.items():
            filtered_signal = self.digital_filtering(eeg_data=signal)
            filtered_signals[ch] = filtered_signal
        return filtered_signals

    def digital_filtering(self, eeg_data):
        try:
            # Handle NaN and infinite values
            eeg_data = self.preprocess_data(data=eeg_data)

            # Print data shape
            print(f"Original data shape: {eeg_data.shape}")

            # Check the dimensions of the eeg_data
            if eeg_data.ndim == 1:
                eeg_data = eeg_data.reshape(1, -1)
            
            if self._notch:
                # Remove power line noise
                eeg_data = self.butter_notch_filter(eeg_data, notch_freq=50)
                print(f"Data shape after notch filter: {eeg_data.shape}")
            
            if self._bandpass:
                # Apply high-pass and low-pass filters (bandpass)
                eeg_data = self.butter_bandpass_filter(eeg_data, lowcut=0.5, highcut=50)
                print(f"Data shape after bandpass filter: {eeg_data.shape}")
            
            if self._fir:
                # Apply FIR filter
                eeg_data = self.fir_filter(eeg_data, cutoff=30, numtaps=101)
                print(f"Data shape after FIR filter: {eeg_data.shape}")
            
            if self._iir:
                # Apply IIR filter
                eeg_data = self.iir_filter(eeg_data, cutoff=30)
                print(f"Data shape after IIR filter: {eeg_data.shape}")
            
            if eeg_data is not None:
                # Ensure the filtered data has the same length as t
                if eeg_data.shape[0] == 1:
                    eeg_data = eeg_data.flatten()

                return eeg_data

        # Handle errors in the digital filtering
        except Exception as e:
            print(f"An error occurred during filtering: {e}")
            return None

    # Preprocessing function to handle inf and NaN values
    def preprocess_data(self, data):
        # Replace inf with NaN
        data = np.where(np.isinf(data), np.nan, data)
        # Remove NaN values
        data = np.nan_to_num(data)
        return data

    def butter_bandpass_filter(self, data, lowcut, highcut, order=5):
        # Bandpass filter which allows a specific range of frequencies to pass
        nyquist = 0.5 * self._fs
        low = lowcut / nyquist
        high = highcut / nyquist

        # Range of the bandpass filter
        b, a = butter(order, [low, high], btype='band')

        # Check the padding length
        padlen = 3 * max(len(b), len(a)) 
        if data.shape[1] <= padlen:
            raise ValueError(f"The length of the input vector must be greater than padlen, which is {padlen}. Data length is {data.shape[1]}.")
        
        # Apply the bandpass filter
        y = filtfilt(b, a, data, axis=1)
        return y

    def butter_notch_filter(self, data, notch_freq, quality_factor=30):
        # Filter used to remove a specific frequency
        nyquist = 0.5 * self._fs
        notch = notch_freq / nyquist

        # Calculate the specific small band which will be filtered
        b, a = butter(2, [notch - notch / quality_factor, notch + notch / quality_factor], btype='bandstop')

        # Calculate the padding length
        padlen = 3 * max(len(b), len(a))
        if data.shape[1] <= padlen:
            raise ValueError(f"The length of the input vector must be greater than padlen, which is {padlen}. Data length is {data.shape[1]}.")
        
        # Apply the notch filter
        y = filtfilt(b, a, data, axis=1)
        return y

    def fir_filter(self, data, cutoff, numtaps):
        # Design FIR filter using firwin
        fir_coefficients = firwin(numtaps, cutoff, fs=self._fs, pass_zero=True)  # Low-pass FIR filter
        # Apply the FIR filter using lfilter
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered_data[i, :] = lfilter(fir_coefficients, 1.0, data[i, :])
        
        return filtered_data

    def iir_filter(self, data, cutoff):
        # Design IIR filter using iirfilter
        b, a = iirfilter(4, cutoff, fs=self._fs, btype='low', ftype='butter')  # Low-pass IIR filter
        # Apply the IIR filter using filtfilt for zero-phase filtering
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered_data[i, :] = filtfilt(b, a, data[i, :])
        
        return filtered_data

class MotorBias:
    def __init__(self, echo_forward, trigger_forward, echo_backwards, trigger_backwards, echo_right, trigger_right,
                 echo_left, trigger_left, led_forward, led_backwards, led_left, led_right, buzzer, motor1_in1,
                 motor1_in2, motor2_in1, motor2_in2):

        # Set up pin factory in order to use pigpio
        factory = PiGPIOFactory()

        # Configure ultrasonic sensors and LEDs
        self._ultrasonic_forward = DistanceSensor(echo=echo_forward, trigger=trigger_forward, pin_factory=factory)
        self._ultrasonic_backwards = DistanceSensor(echo=echo_backwards, trigger=trigger_backwards, pin_factory=factory)
        self._ultrasonic_right = DistanceSensor(echo=echo_right, trigger=trigger_right, pin_factory=factory)
        self._ultrasonic_left = DistanceSensor(echo=echo_left, trigger=trigger_left, pin_factory=factory)

        # Configure LEDs
        self._led_forward = PWMLED(led_forward, pin_factory=factory)
        self._led_backwards = PWMLED(led_backwards, pin_factory=factory)
        self._led_left = PWMLED(led_left, pin_factory=factory)
        self._led_right = PWMLED(led_right, pin_factory=factory)

        # Configure buzzer
        self._buzzer = Buzzer(buzzer)

        # GPIO Pin setup for Motor 1
        self._motor1_in1 = PWMOutputDevice(motor1_in1, initial_value=0,  frequency=50, pin_factory=factory)
        self._motor1_in2 = PWMOutputDevice(motor1_in2, initial_value=0, frequency=50, pin_factory=factory)

        # GPIO Pin setup for Motor 2
        self._motor2_in1 = PWMOutputDevice(motor2_in1, initial_value=0, frequency=50, pin_factory=factory)
        self._motor2_in2 = PWMOutputDevice(motor2_in2, initial_value=0, frequency=50, pin_factory=factory)

    def move_if_possible(self, command):
        try:
            # Move forward
            if command == "forward":
                distance = self._ultrasonic_forward.distance * 100
                # Maximum distance of 20 cm
                if distance < 20:
                    # Forward is blocked
                    self._led_forward.on()
                    self._buzzer.on()
                    print(f"Obastacle forward: {distance:.1f} cm. Blocked movement.")
                else:
                    print("Going forward")
                    # Do the movement
                    self._led_forward.off()
                    self._buzzer.off()
                    self.move_forward(25)
            # Move backwards
            elif command == "backwards":
                distance = self._ultrasonic_backwards.distance * 100
                # Maximum distance of 20 cm
                if distance < 20:
                    # Backwards is blocked
                    self._led_backwards.on()
                    self._buzzer.on()
                    print(f"Obstacle backwards: {distance:.1f} cm. Blocked movement.")
                else:
                    print("Going backwards")
                    # Do the movement
                    self._led_backwards.off()
                    self._buzzer.off()
                    self.move_backward(25)
            # Turn left
            elif command == "left":
                distance = self._ultrasonic_left.distance * 100
                # Maximum distance of 20 cm
                if distance < 20:
                    # Left is blocked
                    self._led_left.on()
                    self._buzzer.on()
                    print(f"Obstacle on the left: {distance:.1f} cm. Blocked movement")
                else:
                    print("Turning left")
                    # Do the movement
                    self._led_left.off()
                    self._buzzer.off()
                    self.turn_left(25)
            # Turn right
            elif command == "right":
                distance = self._ultrasonic_right.distance * 100
                # Maximum distance of 20 cm
                if distance < 20:
                    # Right is blocked
                    self._led_right.on()
                    self._buzzer.on()
                    print(f"Obstacle on the right: {distance:.1f} cm. Blocked movement.")
                else:
                    print("Turning right")
                    # Do the movement
                    self._led_right.off()
                    self._buzzer.off()
                    self.turn_right(25)
            # Brake
            elif command == "stop":
                print("Stopping")
                # Make all parameters off
                self.brake()
                self._led_forward.off()
                self._led_backwards.off()
                self._led_left.off()
                self._led_right.off()
                self._buzzer.off()
            else:
                print("Invalid command")

            time.sleep(1)
            self.brake() # Stop after each command
            self._led_forward.off()
            self._led_backwards.off()
            self._led_left.off()
            self._led_right.off()
            self._buzzer.off()


        except KeyboardInterrupt:
            self.brake()
            print("Program stopped by user")

    # Configure speed of motor depending on PWM
    def set_motor_speed(self, motor_in1, motor_in2, speed):
        # Define positive speed
        if speed > 0:
            motor_in2.value = 0
            if motor_in2.value == 0:
                motor_in1.value = speed / 100.0
            print(motor_in1.value)
            print(motor_in2.value)

        # Define negative speed
        elif speed < 0:
            motor_in1.value = 0
            if motor_in1.value == 0:
                motor_in2.value = abs(speed) / 100.0
            print(motor_in1.value)
            print(motor_in2.value)
        # If it's zero brake
        else:
            motor_in1.value = 0
            motor_in2.value = 0
            print(motor_in1.value)
            print(motor_in2.value)

    # Move wheelchair forward
    def move_forward(self, speed):
        self.set_motor_speed(self._motor1_in1, self._motor1_in2, speed)
        self.set_motor_speed(self._motor2_in1, self._motor2_in2, speed)

    # Move wheelchair backwards
    def move_backward(self, speed):
        self.set_motor_speed(self._motor1_in1, self._motor1_in2, -speed)
        self.set_motor_speed(self._motor2_in1, self._motor2_in2, -speed)

    # Turn wheelchair left
    def turn_left(self, speed):
        self.set_motor_speed(self._motor1_in1, self._motor1_in2, -speed)
        self.set_motor_speed(self._motor2_in1, self._motor2_in2, speed)

    # Turn wheelchair right
    def turn_right(self, speed):
        self.set_motor_speed(self._motor1_in1, self._motor1_in2, speed)
        self.set_motor_speed(self._motor2_in1, self._motor2_in2, -speed)

    # Brake wheelchair
    def brake(self):
        self.set_motor_speed(self._motor1_in1, self._motor1_in2, 0)
        self.set_motor_speed(self._motor2_in1, self._motor2_in2, 0)

def procesar_senales(cola_datos):
    n = 1000
    fs = 500
    number_of_channels = 4

    # ... (Lógica de adquisición de señales - biasReception.get_real_data o generate_synthetic_eeg) ...

    biasFilter = FilterBias(n=n, fs=fs, notch=True, bandpass=True, fir=False, iir=False)
    filtered_data = biasFilter.filter_signals(eeg_signals=signals)


    biasProcessing = ProcessingBias(n=n, fs=fs)
    times, processed_signals = biasProcessing.process_signals(eeg_signals=filtered_data)


    cola_datos.put((times, processed_signals))


def controlar_motor(cola_datos):
    # Define motor instance
    biasMotor = MotorBias(echo_forward=18, trigger_forward=17, echo_backwards=23, trigger_backwards=22, echo_right=5, trigger_right=6,
                          echo_left=25, trigger_left=24, led_forward=16, led_backwards=20, led_left=21, led_right=26, buzzer=12, motor1_in1=13,
                          motor1_in2=19, motor2_in1=7, motor2_in2=8)

    biasMotor.brake()

    while True:
        try:
            times, processed_signals = cola_datos.get()

            #  Lógica de predicción y control del motor basada en processed_signals

        except KeyboardInterrupt:
            biasMotor.brake()
            print("Program stopped by user")
            break



if __name__ == "__main__":
    cola_datos = multiprocessing.Queue()

    proceso_senales = multiprocessing.Process(target=procesar_senales, args=(cola_datos,))
    proceso_motor = multiprocessing.Process(target=controlar_motor, args=(cola_datos,))

    proceso_senales.start()
    proceso_motor.start()

    proceso_senales.join()
    proceso_motor.join()
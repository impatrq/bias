#import smbus2
import time
import numpy as np
import graphing # type: ignore
import filtering

def get_real_combined_data(n, fs, filter=False):
    # Get real data of signals
    real_eeg_signals = get_real_data(n, fs)

    # Filter if necessary
    if filter:
        filtered_signals = filtering.filter_signals(real_eeg_signals, fs)
        combined_signal = combine_signals(filtered_signals)
    else:
        combined_signal = combine_signals(real_eeg_signals)

    return combined_signal

# Get the data from the Raspberry Pi Pico
def get_real_data(n=1000, fs=500):
    I2C_SLAVE_ADDRESS = 0x04
    I2C_BUS = 1

    # Set i2c bus
    bus = smbus2.SMBus(I2C_BUS)

    try:
        # Capture the signal from the i2c bus
        real_eeg_signals = capture_signals(bus, I2C_SLAVE_ADDRESS, n, fs)
        return real_eeg_signals

    # Except error in the bus
    except Exception as e:
        print(f"Error: {e}")
        raise

    # Close the bus after having measured
    finally:
        bus.close()

# Function to write data to the I2C slave
def write_data(bus, data, address):
    try:
        # Write data
        bus.write_byte(address, data)
        print(f"Sent data: {data}")
    except Exception as e:
        print("Error writing data: {e}")
        raise

# Function to read data from the I2C slave
def read_adc_value(bus, address):
    try:   
        # Read the data
        data = bus.read_i2c_block_data(address, 0, 2)
        # Sum the two bytes of the ADC to obtain the real measure
        adc_value = (data[0] << 8) | data[1]
        return adc_value
    except Exception as e:
        print(f"Error reading ADC: {e}")
        raise

def capture_signals(bus, address, channels=4, n=1000, fs=500):
    # Initialize variables
    signals = {ch: [] for ch in range(channels)}
    sampling_period = 1 / fs

    # Read ADC values alternately from each channel
    for _ in range(n // channels):
        # Start timer
        start_time = time.time()

        # Read all channels
        for adc_channel in range(channels):
            # Write the channel in order to get the reading
            write_data(bus, adc_channel, address)
            # Read ADC channel and combine the signal with the other threes
            adc_value = read_adc_value(bus, address)
            # Append the data obtained to each channel data
            signals[adc_channel].append(adc_value)

    # Check the time it takes to read
    elapsed_time = time.time() - start_time

    # Calculate resting time
    sleep_time = max(sampling_period - elapsed_time, 0)
    time.sleep(sleep_time)

    # Ensure all signals have the correct length
    for ch in range(channels):
        signals[ch] = signals[ch][:n // channels]

    return signals

def combine_signals(signals):
    combined_signal = np.mean([signals[ch] for ch in signals], axis=0)
    return combined_signal

def main():
    # Set constants
    n = 1000
    fs = 500
    # Obtain the signal in real time
    combined_signal = get_real_combined_data(n=n, fs=fs, filter=False)

    # Plotting or further processing of combined_signal
    t = np.arange(len(combined_signal)) / fs

    graphing.graph_signal_voltage_time(t=t, signal=combined_signal, title="Combined Filtered Signal")

if __name__ == "__main__":
    main()

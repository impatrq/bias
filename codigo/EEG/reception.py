import smbus2
import time
import numpy as np

# Get the data from the Raspberry Pi Pico
def get_real_data(n=1000, fs=500):
    I2C_SLAVE_ADDRESS = 0x04
    I2C_BUS = 1

    # Set i2c bus
    bus = smbus2.SMBus(I2C_BUS)

    try:
        # Capture the signal from the i2c bus
        real_eeg_signal = capture_signal(bus, I2C_SLAVE_ADDRESS, n, fs)
        return real_eeg_signal

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
        bus.write_byte(address, data)
        print(f"Sent data: {data}")
    except Exception as e:
        print("Error writing data: {e}")
        raise

# Function to read data from the I2C slave
def read_adc_value(bus, address):
    try:
        data = bus.read_i2c_block_data(address, 0, 2)
        # Sum the two bytes of the ADC
        adc_value = (data[0] << 8) | data[1]
        return adc_value
    except Exception as e:
        print(f"Error reading ADc: {e}")
        raise


def capture_signal(bus, address, channels=4, n=1000, fs=500):
    # Initialize variables
    combined_signal = []
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
            combined_signal.append(adc_value)

    elapsed_time = time.time() - start_time
    sleep_time = max(sampling_period - elapsed_time, 0)
    time.sleep(sleep_time)

    # Ensure the combined signal length matches the expected number of samples
    combined_signal = combined_signal[:n]

    return np.array(combined_signal)

if __name__ == "__main__":
    real_eeg_signal = get_real_data()
    print(real_eeg_signal)
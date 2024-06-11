import smbus2
import time
import numpy as np

def get_real_data(n=1000, fs=500):
    I2C_SLAVE_ADDRESS = 0x04
    I2C_BUS = 1

    # Set i2c bus
    bus = smbus2.SMBus(I2C_BUS)

    try:
        real_eeg_signal = capture_signal(bus, I2C_SLAVE_ADDRESS, n, fs)
        return real_eeg_signal

    except KeyboardInterrupt:
        print("Program stopped")
    finally:
        bus.close()

# Function to write data to the I2C slave
def write_data(bus, data, address):
    bus.write_byte(address, data)
    print(f"Sent data: {data}")

# Function to read data from the I2C slave
def read_adc_value(bus, address):
    data = bus.read_i2c_block_data(address, 0, 2)
    # Sum the two bytes of the ADC
    adc_value = (data[0] << 8) | data[1]
    return adc_value

def capture_signal(bus, address, channels=4, n=1000, fs=500):
    # Initialize variables
    combined_signal = []

    # Check frequency of readings

    # Read ADC values alternately from each channel
    for _ in range(n // channels):
        for adc_channel in range(channels):
            write_data(bus, adc_channel, address)
            time.sleep(0.01)
            adc_value = read_adc_value(bus, address)
            combined_signal.append(adc_value)
            time.sleep(0.01)

    # Ensure the combined signal length matches the expected number of samples
    combined_signal = combined_signal[:n]

    return np.array(combined_signal)

if __name__ == "__main__":
    real_eeg_signal = get_real_data()
    print(real_eeg_signal)
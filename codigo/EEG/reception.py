import serial
import time
import numpy as np
import graphing  # type: ignore
import json

# Global serial object
ser = None

def get_real_combined_data(n, fs, filter=False):
    # Get real data of signals
    real_eeg_signals = get_real_data(n=n, fs=fs)
    '''
    # Filter if necessary
    if filter:
        filtered_signals = filtering.filter_signals(real_eeg_signals, fs)
        combined_signal = combine_signals(filtered_signals)
    else:
    '''
    combined_signal = combine_signals(real_eeg_signals)

    return combined_signal

# Get the data from the RP2040 Zero
def get_real_data(n=1000, fs=500):
    global ser
    ser = init_serial()
    try:
        # Capture the signal from the UART
        real_eeg_signals = capture_signals(channels=4, n=n, fs=fs)
        return real_eeg_signals
    finally:
        ser.close()

def capture_signals(channels=4, n=1000, fs=500):
    # Initialize variables
    signals = {f'ch{ch}': [] for ch in range(channels)}
    start_time = time.time()

    while len(signals['ch0']) < n:  # Loop until we have enough samples
        if ser.in_waiting > 0:
            data = ser.readline().decode('utf-8').strip()
            eeg_data = process_data(data)
            if eeg_data:
                for ch in range(channels):
                    signals[f'ch{ch}'].extend(eeg_data[f'ch{ch}'])

    # Check the time it takes to read
    elapsed_time = time.time() - start_time
    print(f"elapsed time: {elapsed_time}")    # Ensure all signals have the correct length
    for ch in range(channels):
        signals[f'ch{ch}'] = signals[f'ch{ch}'][:n]

    return signals

def combine_signals(signals):
    combined_signal = np.mean([signals[f'ch{ch}'] for ch in range(len(signals))], axis=0)
    return combined_signal

def init_serial(port='/dev/serial0', baudrate=115200, timeout=1):
    return serial.Serial(port, baudrate, timeout=timeout)

def process_data(data):
    try:
        json_data = json.loads(data)
        print(json_data)
        return json_data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None

def main():
    # Set constants
    n = 1000
    fs = 500

    signals = get_real_data(n=n, fs=fs)
    print(signals['ch0'])
    t = np.arange(len(signals['ch0'])) / fs
    graphing.graph_signal_voltage_time(t=t, signal=np.array(signals['ch0']), title="Combined Filtered Signal")

    # Obtain the signal in real time
    #combined_signal = get_real_combined_data(n=n, fs=fs, filter=False)

    # Plotting or further processing of combined_signal
    #t = np.arange(len(combined_signal)) / fs

    #graphing.graph_signal_voltage_time(t=t, signal=combined_signal, title="Combined Filtered Signal")

if __name__ == "__main__":
    main()


'''
import smbus
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
    I2C_SLAVE_ADDRESS = 0x3E
    I2C_BUS = 1

    # Set i2c bus
    bus = smbus.SMBus(I2C_BUS)

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

    i = 1
    print ("Loop " + str(n))
    # start_time = time.time()
    # Read ADC values alternately from each channel
    for _ in range(n):
        try:
            print ("Writing data " + str(i))
            # Write out I2C command: address, cmd, msg[0]
            bus.write_i2c_block_data(address, i&0xff, [i>>8])
        except Exception as e:
            print ("Writing Error " + str(e))
            continue
        #sleep (0.1)
        read = 0
        while read == 0:
            try:
                print ("Reading data")
                rx_bytes = bus.read_i2c_block_data(address, 0, 8)
            except Exception as e:
                print ("Read Error "+str(e))
                continue
            read = 1
        print ("Read "+str(rx_bytes))

        value0 = rx_bytes[0] + (rx_bytes[1] << 8)
        value1 = rx_bytes[2] + (rx_bytes[3] << 8)
        value2 = rx_bytes[4] + (rx_bytes[5] << 8)
        value3 = rx_bytes[6] + (rx_bytes[7] << 8)

        signals[0].append(value0)
        signals[1].append(value1)
        signals[2].append(value2)
        signals[3].append(value3)

        print(f"Read value0: {value0}; value1: {value1}; value2: {value2}; value3: {value3}")
        i+=1
        
        # Start timer
        start_time = time.time()

        # Read all channels
        for adc_channel in range(channels):
            # Write the channel in order to get the reading
            write_data(bus, adc_channel, address)
            # Read ADC channel and combine the signal with the other threes
            adc_value = read_adc_value(bus, address)
            # Append the data obtained to each channel data
            signals[adc_channel].append(adc_value)}
        
    
    
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
'''
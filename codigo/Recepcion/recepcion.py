import smbus2
import time

def main():
    I2C_SLAVE_ADDRESS = 0x04
    I2C_BUS = 1

    # Set i2c bus
    bus = smbus2.SMBus(I2C_BUS)

    try:
        while True:
            # Set the ADC channel to read
            adc_channel = 0x00
            # Example of data to send
            write_data(bus, adc_channel, I2C_SLAVE_ADDRESS)
            time.sleep(0.1)
            # Read value from the ADC
            adc_value = read_adc_value(bus, I2C_SLAVE_ADDRESS)
            time.sleep(1)
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

if __name__ == "__main__":
    main()

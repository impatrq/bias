import smbus2
import time

def main():
    I2C_SLAVE_ADDRESS = 0x04
    I2C_BUS = 1
    bus = smbus2.SMBus(I2C_BUS)

    try:
        while True:
            adc_channel = 0x00
            write_data(bus, adc_channel, I2C_SLAVE_ADDRESS)  # Example data to send
            time.sleep(0.1)
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
    adc_value = (data[0] << 8) | data[1]
    return adc_value

if __name__ == "__main__":
    main()

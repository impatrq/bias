import smbus2
import time

def main():
    I2C_SLAVE_ADDRESS = 0x04
    I2C_BUS = 1
    bus = smbus2.SMBus(I2C_BUS)

    try:
        while True:
            write_data(0x10, I2C_BUS, I2C_SLAVE_ADDRESS)  # Example data to send
            time.sleep(1)
            read_data(bus, I2C_SLAVE_ADDRESS)
            time.sleep(1)
    except KeyboardInterrupt:
        print("Program stopped")
    finally:
        bus.close()

# Function to write data to the I2C slave
def write_data(data, bus, address):
    bus.write_byte(address, data)
    print(f"Sent data: {data}")

# Function to read data from the I2C slave
def read_data(bus, address):
    data = bus.read_byte(address)
    print(f"Received data: {data}")
    return data

if __name__ == "__main__":
    main()

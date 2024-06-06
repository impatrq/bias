import smbus2
import time

def main():
            # Dirección I2C de la Raspberry Pi Pico
    I2C_ADDR = 0x04

    # Inicializa el bus I2C
    bus = smbus2.SMBus(1)
    while True:
        for i in range(256):
            write_data(i)
            time.sleep(1)

def write_data(value, i2c_adress, bus):
    try:
        bus.write_byte(i2c_adress, value)
        print(f"Data {value} sent")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
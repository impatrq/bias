import smbus
import time
from gpiozero import Motor

# Inicializar I2C
bus = smbus.SMBus(1)
arduino_address = 0x08  # Dirección del Arduino

# Configuración de los motores usando gpiozero
motor_left = Motor(forward=13, backward=19)  # Pines GPIO para el motor izquierdo
motor_right = Motor(forward=7, backward=8) # Pines GPIO para el motor derecho

# Funciones para mover los motores
def move_forward():
    print("Moviendo hacia adelante")
    motor_left.forward()
    motor_right.forward()

def move_backward():
    print("Moviendo hacia atrás")
    motor_left.backward()
    motor_right.backward()

def move_left():
    print("Moviendo hacia la izquierda")
    motor_left.stop()  # Detener motor izquierdo
    motor_right.forward()  # Avanzar motor derecho

def move_right():
    print("Moviendo hacia la derecha")
    motor_left.forward()  # Avanzar motor izquierdo
    motor_right.stop()  # Detener motor derecho

def stop_motors():
    print("Deteniendo motores")
    motor_left.stop()
    motor_right.stop()

# Leer datos del Arduino
def read_from_arduino():
    try:
        data = bus.read_i2c_block_data(arduino_address, 0, 16)
        return ''.join([chr(b) for b in data if b != 0])
    except Exception as e:
        print(f"Error: {e}")
        return None

# Procesar comandos del joystick
while True:
    command = read_from_arduino()
    if command:
        print(f"Recibido: {command}")
        if command == "FORWARD":
            move_forward()
        elif command == "BACKWARD":
            move_backward()
        elif command == "LEFT":
            move_left()
        elif command == "RIGHT":
            move_right()
        else:
            stop_motors()
    time.sleep(0.2)

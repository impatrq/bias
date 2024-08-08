from gpiozero import PWMLED, PWMOutputDevice
from gpiozero.pins.pigpio import PiGPIOFactory
from time import sleep
import time

# Configurar la fábrica de pines para usar pigpio
factory = PiGPIOFactory()

# Configurar el pin que usaremos para PWM
pwm_led_pin = PWMLED(18, pin_factory=factory)

# GPIO Pin setup for Motor 1
motor1_in1 = PWMOutputDevice(13, pin_factory=factory)  # Use PWM-capable pin
motor1_in2 = PWMOutputDevice(19, pin_factory=factory)  # Use PWM-capable pin (si está en uso, cámbialo)

# GPIO Pin setup for Motor 2
motor2_in1 = PWMOutputDevice(17, pin_factory=factory)  # Use PWM-capable pin (debe estar disponible)
motor2_in2 = PWMOutputDevice(12, pin_factory=factory)  # Use PWM-capable pin

def set_motor_speed(motor_in1, motor_in2, speed):
    if speed > 0:
        motor_in1.value = speed / 100.0
        motor_in2.value = 0
    elif speed < 0:
        motor_in1.value = 0
        motor_in2.value = abs(speed) / 100.0
    else:
        motor_in1.value = 0
        motor_in2.value = 0

def move_forward(speed):
    set_motor_speed(motor1_in1, motor1_in2, speed)
    set_motor_speed(motor2_in1, motor2_in2, speed)

def move_backward(speed):
    set_motor_speed(motor1_in1, motor1_in2, -speed)
    set_motor_speed(motor2_in1, motor2_in2, -speed)

def turn_left(speed):
    set_motor_speed(motor1_in1, motor1_in2, -speed)
    set_motor_speed(motor2_in1, motor2_in2, speed)

def turn_right(speed):
    set_motor_speed(motor1_in1, motor1_in2, speed)
    set_motor_speed(motor2_in1, motor2_in2, -speed)

def brake():
    set_motor_speed(motor1_in1, motor1_in2, 0)
    set_motor_speed(motor2_in1, motor2_in2, 0)

# Simulate brain signal processing result
# Replace this with actual AI integration
def get_brain_signal_command():
    # Simulate commands for demonstration (w/a/s/d/b)
    return input("Enter command (w/a/s/d/b for forward/left/backward/right/brake): ").strip()

try:
    while True:
        command = get_brain_signal_command()
        if command == 'w':
            move_forward(50)
        elif command == 'a':
            turn_left(50)
        elif command == 's':
            move_backward(50)
        elif command == 'd':
            turn_right(50)
        elif command == 'b':
            brake()
        else:
            print("Invalid command")

        time.sleep(1)
        brake()  # Stop after each command

        # Controlar el LED PWM (opcional)
        for value in range(0, 101):
            pwm_led_pin.value = value / 100.0
            sleep(0.01)
        for value in range(100, -1, -1):
            pwm_led_pin.value = value / 100.0
            sleep(0.01)

except KeyboardInterrupt:
    pass
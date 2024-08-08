from gpiozero import PWMOutputDevice, DigitalOutputDevice
import time

# GPIO Pin setup for Motor 1
motor1_in1 = PWMOutputDevice(13)  # Use PWM-capable pin
motor1_in2 = PWMOutputDevice(19)  # Use PWM-capable pinx

# GPIO Pin setup for Motor 2
motor2_in1 = PWMOutputDevice(18)  # Use PWM-capable pin
motor2_in2 = PWMOutputDevice(12)  # Use PWM-capable pin

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
except KeyboardInterrupt:
    pass

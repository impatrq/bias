from gpiozero import PWMOutputDevice, DigitalOutputDevice
from time import sleep

class Motor:
    def __init__(self, pwm_pin, dir_pin1, dir_pin2):
        self.pwm = PWMOutputDevice(pwm_pin)
        self.dir1 = DigitalOutputDevice(dir_pin1)
        self.dir2 = DigitalOutputDevice(dir_pin2)

    def forward(self, speed=1.0):
        self.dir1.on()
        self.dir2.off()
        self.pwm.value = speed

    def backward(self, speed=1.0):
        self.dir1.off()
        self.dir2.on()
        self.pwm.value = speed

    def stop(self):
        self.dir1.off()
        self.dir2.off()
        self.pwm.value = 0

def main():
    # Configure motor pins
    right_motor = Motor(pwm_pin=17, dir_pin1=27, dir_pin2=22)
    left_motor = Motor(pwm_pin=18, dir_pin1=23, dir_pin2=24)

    # Dictionary of commands
    commands = {
        'forward': lambda: move_motors('forward', right_motor, left_motor),
        'backward': lambda: move_motors('backward', right_motor, left_motor),
        'left': lambda: move_motors('left', right_motor, left_motor),
        'right': lambda: move_motors('right', right_motor, left_motor),
        'stop': lambda: move_motors('stop', right_motor, left_motor)
    }

    try:
        while True:
            # Read user's command
            command = input("Write a command (forward, backward, left, right, stop): ").strip().lower()
            
            # Execute command only if it's valid
            if command in commands:
                commands[command]()
            else:
                print("Invalid command, try again.")

    except KeyboardInterrupt:
        # Handle user's interruption
        print("Program ended by the user")
    finally:
        right_motor.stop()
        left_motor.stop()

def move_motors(direction, right_motor, left_motor):
    if direction == 'forward':
        right_motor.forward()
        left_motor.forward()
        print("Moving forward")
    elif direction == 'backward':
        right_motor.backward()
        left_motor.backward()
        print("Moving backward")
    elif direction == 'left':
        right_motor.backward()
        left_motor.forward()
        print("Turning left")
    elif direction == 'right':
        right_motor.forward()
        left_motor.backward()
        print("Turning right")
    elif direction == 'stop':
        right_motor.stop()
        left_motor.stop()
        print("Stopping")

if __name__ == "__main__":
    main()

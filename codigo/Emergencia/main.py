from gpiozero import Motor
from time import sleep

def main():
    # Configure motor pins
    right_motor = Motor(forward=17, backward=18)
    left_motor = Motor(forward=22, backward=23) 

    # Dictionary of commands
    commands = {
    'forward': forward,
    'backward': backward,
    'left': left,
    'right': right,
    'stop': stop
    }

    try:
        while True:
            # Read user's command
            command = input("Write a command (forward, backward, left, right, stop): ").strip().lower()
            
            # Execute command only if it's valid
            if command in commands:
                commands[command](right_motor, left_motor)
            else:
                print("Invalid command, try again.")

    except KeyboardInterrupt:
        print("Programm ended by the user")
    finally:
        stop(right_motor, left_motor)

def forward(right_motor, left_motor):
    right_motor.forward()
    left_motor.forward()
    print("Going forward")
    sleep(3)
    right_motor.stop()
    left_motor.stop()

def backward(right_motor, left_motor):
    right_motor.backward()
    left_motor.backward()
    print("Going backwards")
    sleep(3)
    right_motor.stop()
    left_motor.stop()

def left(right_motor, left_motor):
    right_motor.forward()
    left_motor.backward()
    print("Turning left")
    sleep(3)
    right_motor.stop()
    left_motor.stop()

def right(right_motor, left_motor):
    right_motor.backward()
    left_motor.forward()
    print("Turning right")
    sleep(3)
    right_motor.stop()
    left_motor.stop()

def stop(right_motor, left_motor):
    right_motor.stop()
    left_motor.stop()
    print("Stopping")

if __name__ == "__main__":
    main()

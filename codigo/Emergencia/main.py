from gpiozero import Motor, DistanceSensor, LED, Buzzer
from time import sleep

def main():
    # Configure motor pins
    right_motor = Motor(forward=17, backward=18)
    left_motor = Motor(forward=22, backward=23)

    # Configure distance sensors
    sensor_forward = DistanceSensor(echo=15, trigger=16)
    sensor_back = DistanceSensor(echo=13, trigger=14)
    sensor_right = DistanceSensor(echo=11, trigger=12)
    sensor_left = DistanceSensor(echo=9, trigger=10)

    # Configure emergency system
    led_forward = LED(4)
    led_back = LED(5)
    led_left = LED(6)
    led_right = LED(7)
    buzzer = Buzzer(8)

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
                direction = commands[command]
                is_safe = check_safety(direction, sensor_forward, sensor_back, sensor_left, sensor_right)

                if is_safe:
                    move_motors(direction, right_motor, left_motor)
                else:
                    emergency_system(direction, led_forward, led_back, led_left, led_right, buzzer)
                    move_motors('stop', right_motor, left_motor)
            else:
                print("Invalid command, try again.")

    except KeyboardInterrupt:
        # Handle user's interruption
        print("Programm ended by the user")
    finally:
        stop(right_motor, left_motor)

def move_motors(direction, right_motor, left_motor):
    if direction == 'forward':
        forward(right_motor, left_motor)
    elif direction == 'backward':
        backward(right_motor, left_motor)
    elif direction == 'left':
        left(right_motor, left_motor)
    elif direction == 'right':
        right(right_motor, left_motor)
    elif direction == 'stop':
        stop(right_motor, left_motor)

def forward(right_motor, left_motor):
    # Go forward for 3 seconds
    right_motor.forward()
    left_motor.forward()
    print("Going forward")
    sleep(3)
    # Stop after having moved
    right_motor.stop()
    left_motor.stop()

def backward(right_motor, left_motor):
    # Go forward for 3 seconds
    right_motor.backward()
    left_motor.backward()
    print("Going backwards")
    sleep(3)
    # Stop after having moved
    right_motor.stop()
    left_motor.stop()

def left(right_motor, left_motor):
    # Turning left for three seconds
    right_motor.forward()
    left_motor.backward()
    print("Turning left")
    sleep(3)
    # Stop after having moved
    right_motor.stop()
    left_motor.stop()

def right(right_motor, left_motor):
    # Turning right for three seconds
    right_motor.backward()
    left_motor.forward()
    print("Turning right")
    sleep(3)
    # Stop after having moved
    right_motor.stop()
    left_motor.stop()

# Stopping
def stop(right_motor, left_motor):
    right_motor.stop()
    left_motor.stop()
    print("Stopping")

def detected_object(sensor):
    distance = sensor.distance * 100  # Convert to cm
    print(f'Distance: {distance:.1f} cm')
    return distance < 25

def check_safety(direction, sensor_forward, sensor_back, sensor_left, sensor_right):
    if direction == 'forward' and detected_object(sensor_forward):
        return False
    if direction == 'backward' and detected_object(sensor_back):
        return False
    if direction == 'left' and detected_object(sensor_left):
        return False
    if direction == 'right' and detected_object(sensor_right):
        return False
    return True

def emergency_system(direction, led_forward, led_back, led_left, led_right, buzzer):
    if direction == 'forward':
        led_forward.on()
    elif direction == 'backward':
        led_back.on()
    elif direction == 'left':
        led_left.on()
    elif direction == 'right':
        led_right.on()
    
    buzzer.on()
    sleep(1)
    led_forward.off()
    led_back.off()
    led_left.off()
    led_right.off()
    buzzer.off()


if __name__ == "__main__":
    main()

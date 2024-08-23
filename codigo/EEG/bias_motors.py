from gpiozero import DistanceSensor, PWMLED, Buzzer, PWMOutputDevice
from gpiozero.pins.pigpio import PiGPIOFactory
import time

def main():
    # Define motor instance
    biasMotor = MotorBias(echo_forward=18, trigger_forward=17, echo_backwards=23, trigger_backwards=22, echo_right=5, trigger_right=6,
                          echo_left=27, trigger_left=24, led_forward=16, led_backwards=20, led_left=21, led_right=26, buzzer=12, moto1_in1=13, 
                          motor1_in2=19, motor2_in1=7, motor_2_in2=8)
    while True:
        # Get command
        command = input("Enter command (w/a/s/d/b for forward/left/backward/right/brake): ").strip()
        biasMotor.move_if_possible(command)

class MotorBias:
    def __init__(self, echo_forward, trigger_forward, echo_backwards, trigger_backwards, echo_right, trigger_right,
                 echo_left, trigger_left, led_forward, led_backwards, led_left, led_right, buzzer, moto1_in1, 
                 motor1_in2, motor2_in1, motor_2_in2):
        
        # Configurar pin factory in order to use pigpio
        factory = PiGPIOFactory()

        # Configure ultrasonic sensors and LEDs
        self._ultrasonic_forward = DistanceSensor(echo=echo_forward, trigger=trigger_forward, pin_factory=factory)
        self._ultrasonic_backwards = DistanceSensor(echo=echo_backwards, trigger=trigger_backwards, pin_factory=factory)
        self._ultrasonic_right = DistanceSensor(echo=echo_right, trigger=trigger_right, pin_factory=factory)
        self._ultrasonic_left = DistanceSensor(echo=echo_left, trigger=trigger_left, pin_factory=factory)

        # Configure LEDs
        self._led_forward = PWMLED(led_forward, pin_factory=factory)
        self._led_backwards = PWMLED(led_backwards, pin_factory=factory)
        self._led_left = PWMLED(led_left, pin_factory=factory)
        self._led_right = PWMLED(led_right, pin_factory=factory)

        # Configure buzzer
        self._buzzer = Buzzer(buzzer)

        # GPIO Pin setup for Motor 1
        self._motor1_in1 = PWMOutputDevice(moto1_in1, pin_factory=factory)
        self._motor1_in2 = PWMOutputDevice(motor1_in2, pin_factory=factory)

        # GPIO Pin setup for Motor 2
        self._motor2_in1 = PWMOutputDevice(motor2_in1, pin_factory=factory)
        self._motor2_in2 = PWMOutputDevice(motor_2_in2, pin_factory=factory)

    def move_if_possible(self, command):
        try:
            # Move forward
            if command == 'w': 
                distance = self._ultrasonic_forward.distance * 100
                # Maximum distance of 20 cm
                if distance < 20:
                    # Forward is blocked
                    self._led_forward.on()
                    self._buzzer.on()
                    print(f"Obastacle forward: {distance:.1f} cm. Blocked movement.")
                else:
                    # Do the movement
                    self._led_forward.off()
                    self._buzzer.off()
                    self.move_forward(50)
            # Move backwards
            elif command == 's':
                distance = self._ultrasonic_backwards.distance * 100
                # Maximum distance of 20 cm
                if distance < 20:
                    # Backwards is blocked
                    self._led_backwards.on()
                    self._buzzer.on()
                    print(f"Obstacle backwards: {distance:.1f} cm. Blocked movement.")
                else:
                    # Do the movement
                    self._led_backwards.off()
                    self._buzzer.off()
                    self.move_backward(50)
            # Turn left
            elif command == 'a':
                distance = self._ultrasonic_left.distance * 100
                # Maximum distance of 20 cm
                if distance < 20:
                    # Left is blocked
                    self._led_left.on()
                    self._buzzer.on()
                    print(f"Obstacle on the left: {distance:.1f} cm. Blocked movement")
                else:
                    # Do the movement
                    self._led_left.off()
                    self._buzzer.off()
                    self.turn_left(50)
            # Turn right
            elif command == 'd':
                distance = self._ultrasonic_right.distance * 100
                # Maximum distance of 20 cm
                if distance < 20:
                    # Right is blocked
                    self._led_right.on()
                    self._buzzer.on()
                    print(f"Obstacle on the right: {distance:.1f} cm. Blocked movement.")
                else:
                    # Do the movement
                    self._led_right.off()
                    self._buzzer.off()
                    self.turn_right(50)
            # Brake
            elif command == 'b':
                # Make all parameters off
                self.brake()
                self._led_forward.off()
                self._led_backwards.off()
                self._led_left.off()
                self._led_right.off()
                self._buzzer.off()
            else:
                print("Invalid command")

            time.sleep(1)
            self.brake() # Stop after each command
        
        except KeyboardInterrupt:
            print("Program stopped by user")
    
    # Configure speed of motor depending on PWM
    def set_motor_speed(self, motor_in1, motor_in2, speed):
        # Define positive speed
        if speed > 0:
            motor_in1.value = speed / 100.0
            motor_in2.value = 0
        # Define negative speed
        elif speed < 0:
            motor_in1.value = 0
            motor_in2.value = abs(speed) / 100.0
        # If it's zero brake
        else:
            motor_in1.value = 0
            motor_in2.value = 0

    # Move wheelchair forward
    def move_forward(self, speed):
        self.set_motor_speed(self._motor1_in1, self._motor1_in2, speed)
        self.set_motor_speed(self._motor2_in1, self._motor2_in2, speed)

    # Move wheelchair backwards
    def move_backward(self, speed):
        self.set_motor_speed(self._motor1_in1, self._motor1_in2, -speed)
        self.set_motor_speed(self._motor2_in1, self._motor2_in2, -speed)

    # Turn wheelchair left
    def turn_left(self, speed):
        self.set_motor_speed(self._motor1_in1, self._motor1_in2, -speed)
        self.set_motor_speed(self._motor2_in1, self._motor2_in2, speed)

    # Turn wheelchair right
    def turn_right(self, speed):
        self.set_motor_speed(self._motor1_in1, self._motor1_in2, speed)
        self.set_motor_speed(self._motor2_in1, self._motor2_in2, -speed)

    # Brake wheelchair
    def brake(self):
        self.set_motor_speed(self._motor1_in1, self._motor1_in2, 0)
        self.set_motor_speed(self._motor2_in1, self._motor2_in2, 0)

if __name__ == "__main__":
    main()
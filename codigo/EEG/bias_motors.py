from gpiozero import DistanceSensor, PWMLED, Buzzer, PWMOutputDevice
from gpiozero.pins.pigpio import PiGPIOFactory
import time

class MotorBias:
    def __init__(self, echo_forward, trigger_forward, echo_backwards, trigger_backwards, echo_right, trigger_right,
                 echo_left, trigger_left, led_forward, led_backwards, led_left, led_right, buzzer, moto1_in1, 
                 motor1_in2, motor2_in1, motor_2_in2):
        
        # Configurar la fábrica de pines para usar pigpio
        factory = PiGPIOFactory()

        # Configuración de los sensores de ultrasonido y LEDs
        self._ultrasonido_forward = DistanceSensor(echo=echo_forward, trigger=trigger_forward, pin_factory=factory)
        self._ultrasonido_backwards = DistanceSensor(echo=echo_backwards, trigger=trigger_backwards, pin_factory=factory)
        self._ultrasonido_right = DistanceSensor(echo=echo_right, trigger=trigger_right, pin_factory=factory)
        self._ultrasonido_left = DistanceSensor(echo=echo_left, trigger=trigger_left, pin_factory=factory)

        self._led_forward = PWMLED(led_forward, pin_factory=factory)
        self._led_backwards = PWMLED(led_backwards, pin_factory=factory)
        self._led_left = PWMLED(led_left, pin_factory=factory)
        self._led_right = PWMLED(led_right, pin_factory=factory)

        # Configuración del buzzer
        self._buzzer = Buzzer(buzzer)

        # GPIO Pin setup for Motor 1
        self._motor1_in1 = PWMOutputDevice(moto1_in1, pin_factory=factory)
        self._motor1_in2 = PWMOutputDevice(motor1_in2, pin_factory=factory)

        # GPIO Pin setup for Motor 2
        self._motor2_in1 = PWMOutputDevice(motor2_in1, pin_factory=factory)
        self._motor2_in2 = PWMOutputDevice(motor_2_in2, pin_factory=factory)

    def get_data_and_move_motors(self):
        try:
            while True:
                # Procesar comandos de señales cerebrales (simuladas)
                command = self.get_brain_signal_command()

                if command == 'w':  # Mover hacia adelante
                    distancia = self._ultrasonido_forward.distance * 100
                    if distancia < 20:  # Umbral de 20 cm
                        self._led_forward.on()
                        self._buzzer.on()
                        print(f"Obastacle forward: {distancia:.1f} cm. Blocked movement.")
                    else:
                        self._led_forward.off()
                        self._buzzer.off()
                        self.move_forward(50)
                elif command == 's':  # Mover hacia atrás
                    distancia = self._ultrasonido_backwards.distance * 100
                    if distancia < 20:  # Umbral de 20 cm
                        self._led_backwards.on()
                        self._buzzer.on()
                        print(f"Obstacle backwards: {distancia:.1f} cm. Blocked movement.")
                    else:
                        self._led_backwards.off()
                        self._buzzer.off()
                        self.move_backward(50)
                elif command == 'a':  # Girar a la izquierda
                    distancia = self._ultrasonido_left.distance * 100
                    if distancia < 20:  # Umbral de 20 cm
                        self._led_left.on()
                        self._buzzer.on()
                        print(f"Obstacle on the left: {distancia:.1f} cm. Blocked movement")
                    else:
                        self._led_left.off()
                        self._buzzer.off()
                        self.turn_left(50)
                elif command == 'd':  # Girar a la derecha
                    distancia = self._ultrasonido_right.distance * 100
                    if distancia < 20:  # Umbral de 20 cm
                        self._led_right.on()
                        self._buzzer.on()
                        print(f"Obstacle on the right: {distancia:.1f} cm. Blocked movement.")
                    else:
                        self._led_right.off()
                        self._buzzer.off()
                        self.turn_right(50)
                elif command == 'b':  # Frenar
                    self.brake()
                    self._led_forward.off()
                    self._led_backwards.off()
                    self._led_left.off()
                    self._led_right.off()
                    self._buzzer.off()
                else:
                    print("Invalid command")

                time.sleep(1)
                self.brake() # Detener después de cada comando
        
        except KeyboardInterrupt:
            print("Program stopped by user")

    def set_motor_speed(self, motor_in1, motor_in2, speed):
        if speed > 0:
            motor_in1.value = speed / 100.0
            motor_in2.value = 0
        elif speed < 0:
            motor_in1.value = 0
            motor_in2.value = abs(speed) / 100.0
        else:
            motor_in1.value = 0
            motor_in2.value = 0

    def move_forward(self, speed):
        self.set_motor_speed(self._motor1_in1, self._motor1_in2, speed)
        self.set_motor_speed(self._motor2_in1, self._motor2_in2, speed)

    def move_backward(self, speed):
        self.set_motor_speed(self._motor1_in1, self._motor1_in2, -speed)
        self.set_motor_speed(self._motor2_in1, self._motor2_in2, -speed)

    def turn_left(self, speed):
        self.set_motor_speed(self._motor1_in1, self._motor1_in2, -speed)
        self.set_motor_speed(self._motor2_in1, self._motor2_in2, speed)

    def turn_right(self, speed):
        self.set_motor_speed(self._motor1_in1, self._motor1_in2, speed)
        self.set_motor_speed(self._motor2_in1, self._motor2_in2, -speed)

    def brake(self):
        self.set_motor_speed(self._motor1_in1, self._motor1_in2, 0)
        self.set_motor_speed(self._motor2_in1, self._motor2_in2, 0)

    def get_brain_signal_command():
        return input("Enter command (w/a/s/d/b for forward/left/backward/right/brake): ").strip()
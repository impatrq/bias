from gpiozero import DistanceSensor, PWMLED, Buzzer, PWMOutputDevice
from gpiozero.pins.pigpio import PiGPIOFactory
from time import sleep
import time

# Configurar la fábrica de pines para usar pigpio
factory = PiGPIOFactory()

# Configuración de los sensores de ultrasonido y LEDs
ultrasonido_adelante = DistanceSensor(echo=18, trigger=17, pin_factory=factory)
ultrasonido_atras = DistanceSensor(echo=23, trigger=22, pin_factory=factory)
ultrasonido_izquierda = DistanceSensor(echo=27, trigger=24, pin_factory=factory)
ultrasonido_derecha = DistanceSensor(echo=5, trigger=6, pin_factory=factory)

led_adelante = PWMLED(16, pin_factory=factory)
led_atras = PWMLED(20, pin_factory=factory)
led_izquierda = PWMLED(21, pin_factory=factory)
led_derecha = PWMLED(26, pin_factory=factory)

# Configuración del buzzer
buzzer = Buzzer(12)

# GPIO Pin setup for Motor 1
motor1_in1 = PWMOutputDevice(13, pin_factory=factory)
motor1_in2 = PWMOutputDevice(19, pin_factory=factory)

# GPIO Pin setup for Motor 2
motor2_in1 = PWMOutputDevice(7, pin_factory=factory)
motor2_in2 = PWMOutputDevice(8, pin_factory=factory)

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

def get_brain_signal_command():
    return input("Enter command (w/a/s/d/b for forward/left/backward/right/brake): ").strip()

try:
    while True:
        # Procesar comandos de señales cerebrales (simuladas)
        command = get_brain_signal_command()

        if command == 'w':  # Mover hacia adelante
            distancia = ultrasonido_adelante.distance * 100
            if distancia < 20:  # Umbral de 20 cm
                led_adelante.on()
                buzzer.on()
                print(f"Obstáculo adelante: {distancia:.1f} cm. Movimiento bloqueado.")
            else:
                led_adelante.off()
                buzzer.off()
                move_forward(50)
        elif command == 's':  # Mover hacia atrás
            distancia = ultrasonido_atras.distance * 100
            if distancia < 20:  # Umbral de 20 cm
                led_atras.on()
                buzzer.on()
                print(f"Obstáculo atrás: {distancia:.1f} cm. Movimiento bloqueado.")
            else:
                led_atras.off()
                buzzer.off()
                move_backward(50)
        elif command == 'a':  # Girar a la izquierda
            distancia = ultrasonido_izquierda.distance * 100
            if distancia < 20:  # Umbral de 20 cm
                led_izquierda.on()
                buzzer.on()
                print(f"Obstáculo a la izquierda: {distancia:.1f} cm. Movimiento bloqueado.")
            else:
                led_izquierda.off()
                buzzer.off()
                turn_left(50)
        elif command == 'd':  # Girar a la derecha
            distancia = ultrasonido_derecha.distance * 100
            if distancia < 20:  # Umbral de 20 cm
                led_derecha.on()
                buzzer.on()
                print(f"Obstáculo a la derecha: {distancia:.1f} cm. Movimiento bloqueado.")
            else:
                led_derecha.off()
                buzzer.off()
                turn_right(50)
        elif command == 'b':  # Frenar
            brake()
            led_adelante.off()
            led_atras.off()
            led_izquierda.off()
            led_derecha.off()
            buzzer.off()
        else:
            print("Comando inválido")

        time.sleep(1)
        brake()  # Detener después de cada comando

except KeyboardInterrupt:
    print("Programa detenido por el usuario")

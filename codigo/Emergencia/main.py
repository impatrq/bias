import pygame
from gpiozero import Motor
from time import sleep

# Configuración de los pines de los motores
motor1 = Motor(forward=17, backward=18)  # Motor derecho
motor2 = Motor(forward=22, backward=23)  # Motor izquierdo

# Inicialización de Pygame y el joystick
pygame.init()
pygame.joystick.init()

# Verificar el número de joysticks conectados
joystick_count = pygame.joystick.get_count()
if joystick_count == 0:
    print("No se encontró ningún joystick.")
    pygame.quit()
    exit()

joystick = pygame.joystick.Joystick(0)
joystick.init()

def get_axis_value(axis):
    value = joystick.get_axis(axis)
    if abs(value) < 0.1:
        value = 0
    return value

try:
    while True:
        pygame.event.pump()
        
        # Obtener los valores del joystick
        x = get_axis_value(0)  # Eje X
        y = get_axis_value(1)  # Eje Y
        
        # Control del movimiento
        if y < 0:
            motor1.forward(speed=-y)
            motor2.forward(speed=-y)
        elif y > 0:
            motor1.backward(speed=y)
            motor2.backward(speed=y)
        else:
            motor1.stop()
            motor2.stop()

        if x < 0:
            motor1.backward(speed=-x)
            motor2.forward(speed=-x)
        elif x > 0:
            motor1.forward(speed=x)
            motor2.backward(speed=x)
        else:
            motor1.stop()
            motor2.stop()
        
        sleep(0.1)
finally:
    motor1.stop()
    motor2.stop()
    pygame.quit()
from gpiozero import Motor
from time import sleep

# Configuración de los pines de los motores
motor1 = Motor(forward=17, backward=18)  # Motor derecho
motor2 = Motor(forward=22, backward=23)  # Motor izquierdo

def adelante():
    motor1.forward()
    motor2.forward()
    print("Avanzando hacia adelante")
    sleep(3)
    motor1.stop()
    motor2.stop()

def atras():
    motor1.backward()
    motor2.backward()
    print("Retrocediendo")
    sleep(3)
    motor1.stop()
    motor2.stop()

def izquierda():
    motor1.backward()
    motor2.forward()
    print("Girando a la izquierda")
    sleep(3)
    motor1.stop()
    motor2.stop()

def derecha():
    motor1.forward()
    motor2.backward()
    print("Girando a la derecha")
    sleep(3)
    motor1.stop()
    motor2.stop()

def parar():
    motor1.stop()
    motor2.stop()
    print("Parando")

# Diccionario de comandos
commands = {
    'adelante': adelante,
    'atrás': atras,
    'izquierda': izquierda,
    'derecha': derecha,
    'parar': parar
}

try:
    while True:
        # Leer el comando del usuario
        command = input("Introduce un comando (adelante, atrás, izquierda, derecha, parar): ").strip().lower()
        
        # Ejecutar el comando si es válido
        if command in commands:
            commands[command]()
        else:
            print("Comando no reconocido, intenta de nuevo.")

except KeyboardInterrupt:
    print("Programa terminado por el usuario.")
finally:
    motor1.stop()
    motor2.stop()
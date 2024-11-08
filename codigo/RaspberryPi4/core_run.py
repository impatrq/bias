import time
from bias_graphing import GraphingBias
from bias_motors import MotorBias
from bias_reception import ReceptionBias
from bias_dsp import FilterBias, ProcessingBias
from signals import generate_synthetic_eeg, generate_synthetic_eeg_bandpower

def task_motors():
    biasMotor = MotorBias(
        echo_forward=18, trigger_forward=17, echo_backwards=23, trigger_backwards=22,
        echo_right=5, trigger_right=6, echo_left=25, trigger_left=24,
        led_forward=16, led_backwards=20, led_left=21, led_right=26,
        buzzer=12, motor1_in1=13, motor1_in2=19, motor2_in1=7, motor2_in2=8
    )
    biasMotor.brake()

    command = input("Enter command (forward/left/backwards/right/stop): ").strip()
    biasMotor.move_if_possible(command)
    print("Motor task executed")

def task_reception():
    print("Executing reception task")
    
    print("Reception task executed")

def task_AI():
    print("Executing AI task")

    print("AI task executed")

if __name__ == "__main__":
    while True:
        # Ejecuta la tarea de recepción
        task_reception()
        time.sleep(1)  # Espera 1 segundo antes de la siguiente tarea

        # Ejecuta la tarea de IA
        task_AI()
        time.sleep(1)  # Espera 1 segundo antes de la siguiente tarea

        # Ejecuta la tarea de motores
        task_motors()
        time.sleep(1)  # Espera 1 segundo antes de la siguiente iteración

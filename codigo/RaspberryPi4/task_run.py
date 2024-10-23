import subprocess
import threading

# Función para ejecutar un script en un hilo
def run_script(script_name):
    subprocess.run(['python3', script_name])

# Crear hilos para cada script
thread_dsp = threading.Thread(target=run_script, args=("task_bias_dsp.py",))
thread_motors = threading.Thread(target=run_script, args=("task_motor.py",))
thread_reception = threading.Thread(target=run_script, args=("task_bias_reception.py",))

# Iniciar los hilos
thread_dsp.start()
thread_motors.start()
thread_reception.start()

# Mantener el programa corriendo hasta que todos los hilos terminen
thread_dsp.join()
thread_motors.join()
thread_reception.join()


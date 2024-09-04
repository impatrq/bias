import threading
import time
import importlib

def run_motor_task():
    motor_module = importlib.import_module('motor_task')
    motor_module.motor_task()

def run_processing_task():
    processing_module = importlib.import_module('processing_task')
    
    n = 512
    fs = 256.0
    eeg_signals = {
        1: [0] * n,  
        2: [0] * n,  
        3: [0] * n,  
        4: [0] * n   
    }
    times, processed_signals = processing_module.processing_task(eeg_signals, n, fs)
    print("Processing task completed.")

if __name__ == "__main__":
    # Iniciar la tarea para el control de motores
    motor_thread = threading.Thread(target=run_motor_task)
    motor_thread.start()

    # Iniciar la tarea para el procesamiento de señales
    processing_thread = threading.Thread(target=run_processing_task)
    processing_thread.start()

    # Esperar a que ambas tareas terminen
    motor_thread.join()
    processing_thread.join()

    print("Tarea terminada")
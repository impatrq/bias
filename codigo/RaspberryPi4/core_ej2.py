import multiprocessing

def task_motors():
    
    print("Executing motors task")

def task_reception():

    print("Executing reception task")

def task_AI():
    
    print("Executing AI tash")

if __name__ == "__main__":
    # Crear procesos para cada tarea
    process_motors = multiprocessing.Process(target=task_motors)
    process_reception = multiprocessing.Process(target=task_reception)
    process_AI = multiprocessing.Process(target=task_AI)

    # Iniciar los procesos
    process_motors.start()
    process_reception.start()
    process_AI.start()

    # Esperar a que los procesos terminen
    process_motors.join()
    process_reception.join()
    process_AI.join()

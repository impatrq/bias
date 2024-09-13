import numpy as np
from bias import BiasClass
from bias_motors import MotorBias

def main():
    while True:
        show_menu()
        choice = input("Enter your choice: ")

        if choice == '1':
            pass

        if choice == '2':
            pass

        if choice == '3':
            run_motor_control()
            

        if choice == '4':
            # Define propiedades para la instancia de Bias
            n = int(input("Enter number of data points: "))
            fs = int(input("Enter sampling frequency: "))
            number_of_channels = int(input("Enter number of channels: "))
            port = '/dev/serial0'
            baudrate = 115200
            timeout = 1
            # Crear objeto y correr la aplicación
            biasInstance = BiasClass(n=n, fs=fs, channels=number_of_channels, port=port, baudrate=baudrate, timeout=timeout)
            biasInstance.app_run()

        if choice == '6':
            # Define propiedades para la instancia de Bias
            n = int(input("Enter number of data points: "))
            fs = int(input("Enter sampling frequency: "))
            number_of_channels = int(input("Enter number of channels: "))
            port = '/dev/serial0'
            baudrate = 115200
            timeout = 1
            # Crear objeto y entrenar el modelo AI
            biasInstance = BiasClass(n=n, fs=fs, channels=number_of_channels, port=port, baudrate=baudrate, timeout=timeout)
            saved_dataset_path = None
            save_path = None
            
            loading_dataset = input("Do you want to load an existing dataset? (y/n): ")
            if loading_dataset.lower() == "y":
                saved_dataset_path = input("Enter the name of the file where dataset was saved (without extension): ")
            else:
                save_new_dataset = input("Do you want to save the new dataset? (y/n): ")
                if save_new_dataset.lower() == "y":
                    save_path = input("Enter the path where you want to save the dataset (without extension): ")

            biasInstance.train_ai_model(save_path, saved_dataset_path)
         
def show_menu():
    print("EEG-based Wheelchair Control System")
    print("1. Capture EEG Data with graphing in Terminal")
    print("2. Capture and combine EEG")
    print("3. Motor movement")
    print("4. App run")
    print("5. Extract Features")
    print("6. Train Model")
    print("7. Predict Action")
    print("8. Exit")

def run_motor_control():
    # Crear instancia de MotorBias y correr el control de motores
    biasMotor = MotorBias(
        echo_forward=18, trigger_forward=17,
        echo_backwards=23, trigger_backwards=22,
        echo_right=5, trigger_right=6,
        echo_left=25, trigger_left=24,
        led_forward=16, led_backwards=20,
        led_left=21, led_right=26,
        buzzer=12, motor1_in1=13,
        motor1_in2=19, motor2_in1=7,
        motor2_in2=8
    )

    while True:
        command = input("Enter command (forward/left/backwards/right/stop): ").strip()
        biasMotor.move_if_possible(command)

if __name__ == "__main__":
    main()

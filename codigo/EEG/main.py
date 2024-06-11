import reception
import preprocessing
import extraction
import codigo.EEG.ai as ai
import prediction
import time

def main():
    while True:
        print("1. Train Model")
        print("2. Make Single Prediction")
        print("3. Real-Time Prediction")
        print("4. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            ai.main()
        elif choice == '2':
            prediction.make_prediction()
        elif choice == '3':
            prediction.main()
        elif choice == '4':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
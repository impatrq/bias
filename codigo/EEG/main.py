import ai
import prediction

FS = 500  # Sampling frequency
N = 1000  # Number of samples
DURATION = N / FS

def main():
    while True:
        print("1. Train Model")
        print("2. Make Single Prediction")
        print("3. Real-Time Prediction")
        print("4. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            ai.main(N, DURATION, FS)
        elif choice == '2':
            prediction.make_prediction(N, DURATION, FS)
        elif choice == '3':
            prediction.main(N, DURATION, FS)
        elif choice == '4':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
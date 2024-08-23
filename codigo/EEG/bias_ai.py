import extraction
import pandas as pd
from keras.models import Sequential # type: ignore
from keras.layers import Dense, Dropout # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore
import joblib
import reception


def main():
    n = 1000
    duration = 2
    fs = 500
    online = True
    number_of_channels = 4

    try:
        pass
    except Exception as e:
        print(f"Error in IA main: {e}")
        raise

class AIBias:
     pass

class Extraction(AIBias):
    pass
class Training(AIBias):
    pass
class Prediction(AIBias):
        pass
if __name__ == "__main__":
    main()
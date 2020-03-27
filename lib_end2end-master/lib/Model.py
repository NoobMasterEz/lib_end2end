from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten

class E2E(object):

    def __init__(self,**kwarge):
        """
        Batch_size :  
        """
        self._batch_size=kwarge.get("batch_size")
        self._epoch=kwarge.get("epoch_size")
    
    @propoty
    def model():
        model =Sequential
        model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
        model.add(Conv2D(24, 5, 5, activation='relu', subsample=(2, 2)))
        model.add(Conv2D(36, 5, 5, activation='relu', subsample=(2, 2)))
        model.add(Conv2D(48, 5, 5, activation='relu', subsample=(2, 2)))
        model.add(Conv2D(64, 3, 3, activation='relu'))
        model.add(Conv2D(64, 3, 3, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1))
        model.summary()
        return model


    def fit(self):

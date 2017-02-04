from keras.models import Sequential
from keras.layers import Dense, Activation


class DQNNet():

    """
    Multi Layer Perceptron with Experience Replay
    """

    def __init__(self, n_in, n_out):
        model = Sequential()
        model.add(Dense(16, input_shape=(n_in,)))
        model.add(Activation('relu'))
        model.add(Dense(n_out))
        self.model = model

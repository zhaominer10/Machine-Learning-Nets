from keras.layers import Dense, Flatten, Conv2D
from keras import Model


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, inputs, **kwargs):
        inputs = self.conv1(inputs)  # input[batch, 28, 28, 1] output[batch, 26, 26, 32]
        inputs = self.flatten(inputs)  # output[batch, 26*26*32]
        inputs = self.d1(inputs)  # output[batch, 128]
        return self.d2(inputs)  # output[batch, 10]

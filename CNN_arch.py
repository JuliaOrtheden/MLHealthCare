from tensorflow.keras import activations, models, optimizers
from tensorflow.keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D

class CNN():
    def __init__(self, name, loss, output_dim, last_activation='softmax'):
        self.name = name
        self.loss = loss
        self.output_dim = output_dim
        self.last_activation = last_activation
        self.model = self.get_model()
        self.compile_model()

    def get_model(self):
        inp = Input(shape=(187, 1))
        img_1 = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(inp)
        img_1 = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(img_1)
        img_1 = MaxPool1D(pool_size=2)(img_1)
        img_1 = Dropout(rate=0.1)(img_1)
        img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
        img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
        img_1 = MaxPool1D(pool_size=2)(img_1)
        img_1 = Dropout(rate=0.1)(img_1)
        img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
        img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
        img_1 = MaxPool1D(pool_size=2)(img_1)
        img_1 = Dropout(rate=0.1)(img_1)
        img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
        img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
        img_1 = GlobalMaxPool1D()(img_1)
        img_1 = Dropout(rate=0.2)(img_1)

        dense_1 = Dense(64, activation=activations.relu, name="dense_1")(img_1)
        dense_1 = Dense(64, activation=activations.relu, name="dense_2")(dense_1)
        dense_1 = Dense(self.output_dim, activation=self.last_activation, name="dense_3_mitbih")(dense_1)

        model = models.Model(inputs=inp, outputs=dense_1, name=self.name)

        return model

    def compile_model(self):
        self.model.compile(optimizer=optimizers.Adam(0.001), loss=self.loss, metrics=['acc'])
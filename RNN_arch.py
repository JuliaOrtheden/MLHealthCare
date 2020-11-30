from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Input, Dropout, CuDNNLSTM, Masking, Bidirectional

class RNN:
    def __init__(self, name, output_dim, loss='sparse_categorical_crossentropy', optimizer='RMSprop', dropout=0.2, neurons=200, lstm_size=150, last_activation='softmax'):
        self.name = name
        self.output_dim = output_dim
        self.loss = loss
        self.optimizer = optimizer
        self.dropout = dropout
        self.neurons = neurons
        self.lstm_size = lstm_size
        self.last_activation = last_activation
        self.model = self.get_model()
        self.compile_model()

    def get_model(self):
        x = Input(shape=(187, 1))
        layer = Masking(mask_value=0.0)
        lstm = Bidirectional(CuDNNLSTM(self.lstm_size), merge_mode='concat')(x)
        layer = Dense(self.neurons, activation='relu')(lstm)
        layer = Dropout(self.dropout)(layer)
        y = Dense(self.output_dim, name='out_layer', activation=self.last_activation)(layer)

        model = models.Model(inputs=x, outputs=y, name=self.name)
        return model

    def compile_model(self):
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
from tensorflow.keras import activations, models
from tensorflow.keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, CuDNNGRU, BatchNormalization, Bidirectional
		
class CRNN:
    def __init__(self, name, output_dim, loss='sparse_categorical_crossentropy', optimizer='RMSprop', dropout=.1, dense_size=200, hidden_size=150, last_activation='softmax'):
        self.name = name
        self.output_classes = output_classes
        self.loss = loss
        self.optimizer = optimizer
        self.dense_size = dense_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.last_activation = last_activation
        self.model = self.get_model()
        self.compile_model()
  
    def get_model(self):
        x = Input(shape=(187, 1))
        layer = Convolution1D(16, kernel_size=3, activation=activations.relu, padding="valid")(x)
        layer = Convolution1D(16, kernel_size=3, activation=activations.relu, padding="valid")(layer)
        layer = MaxPool1D(pool_size=2)(layer)
        layer = Dropout(rate=self.dropout)(layer)
        layer = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(layer)
        layer = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(layer)
        layer = MaxPool1D(pool_size=2)(layer)
        layer = Dropout(rate=self.dropout)(layer)
        layer = BatchNormalization()(layer)
        gru = Bidirectional(CuDNNGRU(self.hidden_size, name='rnn'), merge_mode='concat')(layer)
        layer = Dense(self.dense_size, activation=activations.relu, name='dense')(gru)
        y = Dense(self.output_classes, name='out_layer', activation=self.last_activation)(layer)

        model = models.Model(inputs=x, outputs=y, name=self.name)
        return model

    def compile_model(self):
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
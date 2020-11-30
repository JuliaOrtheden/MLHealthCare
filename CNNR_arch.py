from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Input, GlobalMaxPool1D, BatchNormalization, Conv1D, Add, ReLU

class CNNR:
    def __init__(self, name, output_dim, loss, last_activation):
        self.name = name
        self.output_dim = output_dim
        self.loss = loss
        self.last_activation = last_activation
        self.model = self.get_model()
        self.compile_model()

    def get_model(self):
        X_input = Input(shape=(187, 1))
        X = Conv1D(32, kernel_size=5, activation='relu')(X_input)
        
        X_after = Conv1D(32, kernel_size=5, activation='relu', padding='same')(X)
        X_shortcut = X
        X = self.residual_block_32(X)
        X = self.residual_block_32(X)
        X = self.residual_block_32(X)
        X = self.residual_block_32(X)

        X = Conv1D(32, kernel_size=5, activation='relu', padding='same')(X)
        X = Add()([X, X_after])
        X = BatchNormalization()(X)
        X = Conv1D(256, kernel_size=3, activation='relu')(X)
        transfer_layer = GlobalMaxPool1D()(X)
        X = Dense(64, activation='relu', name="dense_1")(transfer_layer)
        X = Dense(64, activation='relu', name="dense_2")(X)
        X_out = Dense(self.output_dim, activation=self.last_activation, name='output_layer')(X)

        model = models.Model(inputs=X_input, outputs=X_out)
        return model

    def compile_model(self):
        self.model.compile(optimizer=optimizers.Adam(0.001), loss=self.loss, metrics=['accuracy'])

    def residual_block_16(self, X):
        X_shortcut = X
        X = BatchNormalization()(X)
        X = ReLU()(X)
        X = Conv1D(16, kernel_size=3, padding = 'same')(X)
        X = BatchNormalization()(X)
        X = ReLU()(X)
        X = Conv1D(16, kernel_size=3, padding = 'same')(X)
        #X = GlobalMaxPool1D()(X)
        X = Add()([X, X_shortcut])
        return X

    def residual_block_32(self, X):
        X_shortcut = X
        X = BatchNormalization()(X)
        X = ReLU()(X)
        X = Conv1D(32, kernel_size=3, padding = 'same')(X)
        X = BatchNormalization()(X)
        X = ReLU()(X)
        X = Conv1D(32, kernel_size=3, padding = 'same')(X)
        #X = GlobalMaxPool1D()(X)
        X = Add()([X, X_shortcut])
        return X

    def residual_block_32_up(self, X):
        X_shortcut = Conv1D(32, kernel_size=1, padding = 'same')(X)
        X = BatchNormalization()(X)
        X = ReLU()(X)
        X = Conv1D(32, kernel_size=3, padding = 'same')(X)
        X = BatchNormalization()(X)
        X = ReLU()(X)
        X = Conv1D(32, kernel_size=3, padding = 'same')(X)
        #X = GlobalMaxPool1D()(X)
        X = Add()([X, X_shortcut])
        return X

    def residual_block_64(self, X):
        X_shortcut = X
        X = BatchNormalization()(X)
        X = ReLU()(X)
        X = Conv1D(64, kernel_size=3, padding = 'same')(X)
        X = BatchNormalization()(X)
        X = ReLU()(X)
        X = Conv1D(64, kernel_size=3, padding = 'same')(X)
        X = Add()([X, X_shortcut])
        return X

    def residual_block_64_up(self, X):
        X_shortcut = Conv1D(64, kernel_size=1, padding = 'same')(X)
        X = BatchNormalization()(X)
        X = ReLU()(X)
        X = Conv1D(64, kernel_size=3, padding = 'same')(X)
        X = BatchNormalization()(X)
        X = ReLU()(X)
        X = Conv1D(64, kernel_size=3, padding = 'same')(X)
        #X = GlobalMaxPool1D()(X)
        X = Add()([X, X_shortcut])
        return X
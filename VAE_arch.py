import numpy as np

from keras.layers import Input, Conv1D, Flatten, Dense, Lambda, Reshape, UpSampling1D, Dropout, MaxPool1D, GlobalMaxPool1D, Concatenate
from keras.models import Model
from keras.optimizers import RMSprop
from keras import backend as K
from keras import callbacks

CONV_FILTERS = [8, 16, 16, 32]
CONV_KERNEL_SIZES = [3,3,3,3]
CONV_STRIDES = [2,2,2,2]
CONV_ACTIVATIONS = ['relu','relu','relu','relu']

DENSE_SIZE = 64

CONV_T_FILTERS = [16,16,8,1]
CONV_T_KERNEL_SIZES = [4,5,2,2]
CONV_T_STRIDES = [1,1,1,1]
CONV_T_ACTIVATIONS = ['relu','relu','relu','sigmoid']

def sampling(args):
    z_mean, z_sigma = args
    epsilon = K.random_normal(shape=K.shape(z_sigma), mean=0.,stddev=1.)
    return z_mean + z_sigma * epsilon

def convert_to_sigma(z_log_var):
    return K.exp(z_log_var / 2)

class VAE():
    def __init__(self, input_dim, z_dim, batch_size = 100):
        self.encoder = self.encoder(input_dim, z_dim)
        self.decoder = self.decoder(z_dim)
        self.model = self.bmodel(input_dim, self.encoder, self.decoder)

    def encoder(self, input_dim, z_dim): 
        inp = Input(shape=input_dim, name="input")
        img_1 = Conv1D(16, kernel_size=5, activation="relu", padding="valid")(inp)
        img_1 = Conv1D(16, kernel_size=5, activation="relu", padding="valid")(img_1)
        img_1 = MaxPool1D(pool_size=2)(img_1)
        img_1 = Dropout(rate=0.1)(img_1)
        img_1 = Conv1D(32, kernel_size=3, activation="relu", padding="valid")(img_1)
        img_1 = Conv1D(32, kernel_size=3, activation="relu", padding="valid")(img_1)
        img_1 = MaxPool1D(pool_size=2)(img_1)
        img_1 = Dropout(rate=0.1)(img_1)
        img_1 = Conv1D(32, kernel_size=3, activation="relu", padding="valid")(img_1)
        img_1 = Conv1D(32, kernel_size=3, activation="relu", padding="valid")(img_1)
        img_1 = MaxPool1D(pool_size=2)(img_1)
        vae_z_flat = Flatten()(img_1)
        vae_z_in = Dense(z_dim)(vae_z_flat)
        
        vae_z_mean = Dense(z_dim, name='mu')(vae_z_in)
        vae_z_log_var = Dense(z_dim, name='log_var')(vae_z_in)
        vae_z_sigma = Lambda(convert_to_sigma, name='sigma')(vae_z_log_var)
        vae_z = Lambda(sampling, name='z')([vae_z_mean, vae_z_sigma])
        return Model(inputs=inp, outputs=vae_z)

    def decoder(self, z_dim):
        z = Input(shape=(z_dim+5,), name="latent_in")
        img_1 = Dense(608)(z)
        img_1 = Reshape((19, 32), name='unflatten')(img_1)
        img_1 = UpSampling1D(size=3)(img_1)
        img_1 = Conv1D(32, kernel_size=3, activation="relu", padding="valid")(img_1)
        img_1 = Conv1D(32, kernel_size=4, activation="relu", padding="valid")(img_1)
        img_1 = Dropout(rate=0.1)(img_1)
        img_1 = UpSampling1D(size=2)(img_1)
        img_1 = Conv1D(16, kernel_size=4, activation="relu", padding="valid")(img_1)
        img_1 = Conv1D(8, kernel_size=4, activation="relu", padding="valid")(img_1)
        img_1 = Dropout(rate=0.1)(img_1)
        img_1 = UpSampling1D(size=2)(img_1)
        img_1 = Conv1D(4, kernel_size=5, activation="relu", padding="valid")(img_1)
        img_1 = Conv1D(1, kernel_size=6, activation="relu", padding="valid")(img_1)
        return Model(inputs=z, outputs=img_1)
  
    def bmodel(self, input_dim, enc, dec):
        x = Input(shape=input_dim, name="input")
        vae_z = enc(x)
        
        y = Input(shape=(5,))
        vae_z = Concatenate()([vae_z, y])
        
        x_pred = dec(vae_z)
        ae = Model(inputs=[x,y], outputs=x_pred)
        opti = RMSprop()
        ae.compile(optimizer=opti, loss='mean_squared_error')
        return ae

    def set_weights(self, filepath):
        self.model.load_weights(filepath)

    def train(self, data_in, data_out, model_name, epochs=10, batch_size=100):
        tensorboard_callback = callbacks.TensorBoard(log_dir='./vae/'+model_name+'/log/')
        self.model.fit(data_in, data_out,
                shuffle=True,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[tensorboard_callback])
        self.save_weights('./vae/'+model_name+'/'+model_name+'_weights.h5')
        
    def save_weights(self, filepath):
        self.model.save_weights(filepath)

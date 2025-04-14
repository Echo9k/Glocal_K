from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class LocalKernelLayer(layers.Layer):
    def __init__(self, n_hid, n_dim, **kwargs):
        super(LocalKernelLayer, self).__init__(**kwargs)
        self.n_hid = n_hid
        self.n_dim = n_dim

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.n_hid), initializer='random_normal', trainable=True)
        self.u = self.add_weight(shape=(input_shape[-1], 1, self.n_dim), initializer='random_normal', trainable=True)
        self.v = self.add_weight(shape=(1, self.n_hid, self.n_dim), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(self.n_hid,), initializer='zeros', trainable=True)
        super(LocalKernelLayer, self).build(input_shape)

    def call(self, inputs):
        dist = keras.backend.sqrt(keras.backend.sum(keras.backend.square(inputs[:, None, :] - inputs[None, :, :]), axis=-1))
        w_hat = keras.backend.maximum(0., 1. - keras.backend.square(dist))
        W_eff = self.W * w_hat
        y = keras.backend.dot(inputs, W_eff) + self.b
        return keras.activations.sigmoid(y)

class GlobalKernelLayer(layers.Layer):
    def __init__(self, gk_size, dot_scale, **kwargs):
        super(GlobalKernelLayer, self).__init__(**kwargs)
        self.gk_size = gk_size
        self.dot_scale = dot_scale

    def build(self, input_shape):
        self.conv_kernel = self.add_weight(shape=(input_shape[-1], self.gk_size ** 2), initializer='random_normal', trainable=True)
        super(GlobalKernelLayer, self).build(input_shape)

    def call(self, inputs):
        avg_pooling = keras.backend.mean(inputs, axis=1, keepdims=True)
        gk = keras.backend.dot(avg_pooling, self.conv_kernel) * self.dot_scale
        return keras.backend.reshape(gk, (self.gk_size, self.gk_size, 1, 1))

def create_model(n_m, n_u, n_hid, n_dim, gk_size, dot_scale):
    inputs = keras.Input(shape=(n_m, n_u))
    x = LocalKernelLayer(n_hid, n_dim)(inputs)
    x = GlobalKernelLayer(gk_size, dot_scale)(x)
    outputs = LocalKernelLayer(n_u, n_dim)(x)
    
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, train_r, train_m, epochs):
    model.fit(train_r, train_m, epochs=epochs, batch_size=32, verbose=1)

def fine_tune_model(model, train_r, train_m, epochs):
    model.fit(train_r, train_m, epochs=epochs, batch_size=32, verbose=1)
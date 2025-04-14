from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import tensorflow as tf # Import tensorflow

class LocalKernelLayer(layers.Layer):
    def __init__(self, n_hid, n_dim, **kwargs):
        super(LocalKernelLayer, self).__init__(**kwargs)
        self.n_hid = n_hid
        self.n_dim = n_dim

    def build(self, input_shape):
        # input_shape is (batch, n_m, n_u) -> (None, 1682, 943)
        self.W = self.add_weight(shape=(input_shape[-1], self.n_hid), initializer='random_normal', trainable=True, name='W') # Shape (n_u, n_hid)
        self.b = self.add_weight(shape=(self.n_hid,), initializer='zeros', trainable=True, name='b') # Shape (n_hid,)
        super(LocalKernelLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch_size, n_m, n_u) -> (None, 1682, 943)
        item_vectors_expanded1 = tf.expand_dims(inputs, 2)
        item_vectors_expanded2 = tf.expand_dims(inputs, 1)
        dist_sq = tf.reduce_sum(tf.square(item_vectors_expanded1 - item_vectors_expanded2), axis=-1) # Shape (batch, n_m, n_m)
        w_hat = tf.maximum(0., 1. - dist_sq) # Shape (batch, n_m, n_m)
        weighted_inputs = tf.einsum('bij,bjk->bik', w_hat, inputs) # Shape (batch, n_m, n_u)
        y = keras.backend.dot(weighted_inputs, self.W) + self.b # Shape (batch, n_m, n_hid)
        return keras.activations.sigmoid(y) # Output shape (batch, n_m, n_hid)


class GlobalKernelLayer(layers.Layer):
    def __init__(self, gk_size, dot_scale, **kwargs):
        super(GlobalKernelLayer, self).__init__(**kwargs)
        self.gk_size = gk_size
        self.dot_scale = dot_scale

    def build(self, input_shape):
        # Input shape is likely (batch, n_m, n_hid)
        # Check if input_shape[-1] is correct here
        self.conv_kernel = self.add_weight(shape=(input_shape[-1], self.gk_size ** 2), initializer='random_normal', trainable=True)
        super(GlobalKernelLayer, self).build(input_shape)

    def call(self, inputs):
        # Input shape (batch, n_m, n_hid)
        avg_pooling = keras.backend.mean(inputs, axis=1, keepdims=True) # Shape (batch, 1, n_hid)
        # Dot product: (batch, 1, n_hid) @ (n_hid, gk_size^2) -> (batch, 1, gk_size^2)
        gk = keras.backend.dot(avg_pooling, self.conv_kernel) * self.dot_scale
        # Reshape needs to retain batch dimension!
        # Target shape might be (batch, gk_size, gk_size, 1) or similar for conv layers
        # Current reshape loses batch dim: return keras.backend.reshape(gk, (self.gk_size, self.gk_size, 1, 1))
        # Corrected reshape (example, adjust as needed):
        batch_size = tf.shape(inputs)[0]
        # Reshape to (batch, gk_size, gk_size, 1) - assuming channel last
        return tf.reshape(gk, (batch_size, self.gk_size, self.gk_size, 1))


def create_model(n_m, n_u, n_hid, n_dim, gk_size, dot_scale):
    # Input shape defines the shape PER SAMPLE fed to the model
    inputs = keras.Input(shape=(n_m, n_u), name='rating_matrix_input') # Expects samples of shape (1682, 943)

    # --- Actual Model Layers (Review and uncomment/correct later) ---
    x = LocalKernelLayer(n_hid, n_dim, name='local_kernel_layer_1')(inputs) # Output: (None, 1682, 500)
    # x = GlobalKernelLayer(gk_size, dot_scale, name='global_kernel_layer')(x) # Output: (None, 3, 3, 1) - Needs check
    # outputs = LocalKernelLayer(n_u, n_dim, name='local_kernel_layer_2')(x) # Input shape mismatch here

    # --- TEMPORARY OUTPUT FOR TESTING INPUT SHAPE ---
    # The target data (train_m_batched) will have shape (1, n_m, n_u) = (1, 1682, 943)
    # We need the model output to match this shape for the loss calculation.
    # The first LocalKernelLayer outputs (None, 1682, 500). We need (None, 1682, 943).
    # Add a Dense layer wrapped in TimeDistributed or apply Dense directly if appropriate.
    # Applying Dense directly to (batch, n_m, n_hid) applies it to the last dim (n_hid).
    temp_outputs = layers.Dense(n_u, activation='sigmoid', name='temp_output_fix')(x) # Output: (None, 1682, 943)
    # --- END TEMPORARY OUTPUT ---

    # Use the temporary output for now
    outputs = temp_outputs

    model = keras.Model(inputs=inputs, outputs=outputs)
    # Use a loss appropriate for comparing the output (predicted ratings) with the target (train_m)
    model.compile(optimizer='adam', loss='mean_squared_error') # MSE is common for rating prediction/reconstruction
    model.summary()
    return model

def train_model(model, train_r, train_m, epochs):
    # Determine shape from loaded data
    if train_r.shape == (model.input_shape[2], model.input_shape[1]): # Check if shape is (n_u, n_m)
        print("Transposing input data from (n_u, n_m) to (n_m, n_u)")
        train_r_processed = np.transpose(train_r) # Shape (n_m, n_u)
        train_m_processed = np.transpose(train_m) # Shape (n_m, n_u)
    elif train_r.shape == (model.input_shape[1], model.input_shape[2]): # Check if shape is already (n_m, n_u)
        print("Input data shape already (n_m, n_u)")
        train_r_processed = train_r
        train_m_processed = train_m
    else:
        raise ValueError(f"Unexpected shape for train_r: {train_r.shape}. Expected ({model.input_shape[2]}, {model.input_shape[1]}) or ({model.input_shape[1]}, {model.input_shape[2]})")

    # Add a batch dimension because the model processes the whole matrix as one sample
    train_r_batched = np.expand_dims(train_r_processed, axis=0) # Shape (1, n_m, n_u)
    # Target y should match the model's output shape (which we temporarily fixed)
    train_m_batched = np.expand_dims(train_m_processed, axis=0) # Shape (1, n_m, n_u)

    print(f"Input data shape for fit: {train_r_batched.shape}")
    print(f"Target data shape for fit: {train_m_batched.shape}")
    print(f"Model Input shape: {model.input_shape}")
    print(f"Model Output shape: {model.output_shape}")

    # Fit the model using the batched data. Use batch_size=1 as we feed the whole matrix.
    model.fit(train_r_batched, train_m_batched, epochs=epochs, batch_size=1, verbose=1)

def fine_tune_model(model, train_r, train_m, epochs):
    # Apply the same processing as in train_model
    if train_r.shape == (model.input_shape[2], model.input_shape[1]): # (n_u, n_m)
        train_r_processed = np.transpose(train_r)
        train_m_processed = np.transpose(train_m)
    elif train_r.shape == (model.input_shape[1], model.input_shape[2]): # (n_m, n_u)
        train_r_processed = train_r
        train_m_processed = train_m
    else:
        raise ValueError(f"Unexpected shape for train_r: {train_r.shape}")

    train_r_batched = np.expand_dims(train_r_processed, axis=0)
    train_m_batched = np.expand_dims(train_m_processed, axis=0)
    print(f"Input data shape for fine-tune fit: {train_r_batched.shape}")
    print(f"Target data shape for fine-tune fit: {train_m_batched.shape}")
    model.fit(train_r_batched, train_m_batched, epochs=epochs, batch_size=1, verbose=1)
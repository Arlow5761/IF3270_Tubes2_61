import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

@keras.utils.register_keras_serializable()
class LocallyConnected2D(layers.Layer):
    """A custom implementation of LocallyConnected2D for Keras 3."""
    def __init__(self, filters, kernel_size, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        _, h, w, c = input_shape
        kh, kw = self.kernel_size

        # Calculate output spatial dimensions (padding='valid')
        self.out_h = h - kh + 1
        self.out_w = w - kw + 1

        # Create unshared weights
        # Shape: (out_height, out_width, patch_size, filters)
        self.kernel = self.add_weight(
            shape=(self.out_h, self.out_w, kh * kw * c, self.filters),
            initializer="glorot_uniform",
            trainable=True,
            name="kernel"
        )
        self.bias = self.add_weight(
            shape=(self.out_h, self.out_w, self.filters),
            initializer="zeros",
            trainable=True,
            name="bias"
        )

    def call(self, inputs):
        # 1. Extract image patches
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.kernel_size[0], self.kernel_size[1], 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )

        # 2. Multiply patches by unshared weights using Einstein summation
        # b: batch, h: height, w: width, p: patch, f: filters
        out = tf.einsum('bhwp,hwpf->bhwf', patches, self.kernel)
        out = out + self.bias

        if self.activation is not None:
            out = self.activation(out)
        return out

    def get_config(self):
        # Retrieve the base layer config (includes name, dtype, etc.)
        config = super().get_config()
        # Add the custom layer's specific arguments
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            # Serialize the activation function correctly
            "activation": keras.activations.serialize(self.activation), 
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Keras uses this method to instantiate the class from the config dictionary.
        # The base implementation does `return cls(**config)`, which works perfectly here.
        return super().from_config(config)
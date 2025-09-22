import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.models import Sequential

class QuantumInspiredDense(Layer):
    """
    A custom Keras layer that mimics the behavior of a parameterized quantum circuit (PQC) layer.
    It uses a cos^2 activation function, derived from the probability of a quantum measurement.
    """

    def __init__(self, units, activation='cos_squared', **kwargs):
        super(QuantumInspiredDense, self).__init__(**kwargs)
        self.units = units
        self.activation_name = activation

    def build(self, input_shape):
        # Create the weight matrix (kernel) and bias vector
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='glorot_uniform',
                                      name='kernel',
                                      trainable=True)
        self.bias = self.add_weight(shape=(self.units,),
                                    initializer='zeros',
                                    name='bias',
                                    trainable=True)
        super(QuantumInspiredDense, self).build(input_shape)

    def call(self, inputs):
        # Compute the linear transformation: Z = X*W + b
        output = tf.matmul(inputs, self.kernel) + self.bias

        # Apply the quantum-inspired activation function
        if self.activation_name == 'cos_squared':
            output = tf.math.square(tf.math.cos(output))
        else:
            output = tf.nn.relu(output) # Fallback to ReLU
        return output

    def get_config(self):
        config = super(QuantumInspiredDense, self).get_config()
        config.update({'units': self.units, 'activation': self.activation_name})
        return config

def create_cqinn_model(input_dim, num_classes):
    """Creates the cQINN model architecture."""
    model = Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        QuantumInspiredDense(32, activation='cos_squared'),
        tf.keras.layers.Dropout(0.1),
        QuantumInspiredDense(24, activation='cos_squared'),
        tf.keras.layers.Dropout(0.1),
        QuantumInspiredDense(16, activation='cos_squared'),
        Dense(units=num_classes, activation='softmax') # Output layer for classification
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def create_dnn_model(input_dim, num_classes):
    """Creates a standard DNN model for comparison."""
    model = Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        Dense(24, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        Dense(16, activation='relu'),
        Dense(units=num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

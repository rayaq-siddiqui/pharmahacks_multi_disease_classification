import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Concatenate, Maximum, Dropout, LeakyReLU
from tensorflow.keras.models import Model


def xtanh(x):
    return x * tf.keras.activations.tanh(x)


def xsigmoid(x):
    return tf.keras.activations.swish(x)
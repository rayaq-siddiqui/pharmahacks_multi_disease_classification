import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Concatenate, Maximum, Dropout, LeakyReLU
from tensorflow.keras.models import Model


class ClassWeightMult(tf.keras.layers.Layer):
    def __init__(self, class_weight):
        super().__init__()
        self.class_weight = class_weight

    def call(self, inputs):
        return inputs * self.class_weight
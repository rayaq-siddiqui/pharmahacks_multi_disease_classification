import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Concatenate, Maximum, Dropout, LeakyReLU
from tensorflow.keras.models import Model
from . import ClassWeightMult


def deep_linear_network(class_weight):
    inp = Input(shape=(1094,))
    act = 'tanh'
    alpha = -0.5

    x = Dense(2048)(inp)
    x = LeakyReLU(alpha=alpha)(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha=alpha)(x)

    x = Dropout(0.5)(x)

    x = Dense(512)(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=alpha)(x)

    x = Dropout(0.5)(x)

    x = Dense(128)(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Dense(64)(x)
    x = LeakyReLU(alpha=alpha)(x)

    x = Dense(4, activation='softmax')(x)
    out = ClassWeightMult.ClassWeightMult(class_weight)(x)

    model = Model(inp, out)
    return model
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Concatenate, Maximum, Dropout, LeakyReLU
from tensorflow.keras.models import Model
from . import functions
from . import ClassWeightMult


def branchy_linear_network(class_weight):
    inp = Input(shape=(1094,))
    swish = tf.keras.activations.swish
    x = Dense(2048)(inp)
    x = LeakyReLU(alpha=0.5)(x)
    x = Dense(2048)(x)
    x = LeakyReLU(alpha=0.5)(x)
    x = Dense(2048)(x)
    x = LeakyReLU(alpha=0.5)(x)
    x = Dropout(0.5)(x)

    x2 = Dense(512)(x)
    x2 = functions.xtanh(x2)
    x3 = Dense(512)(x)
    x3 = swish(x3)

    x2 = Dense(512)(x3)
    x3 = Dense(512)(x2)
    x2 = functions.xtanh(x2)
    x3 = swish(x3)

    x3 = Dense(512)(x2)
    x2 = Dense(512)(x3)
    x2 = functions.xtanh(x2)
    x3 = swish(x3)

    x = Maximum()([x2,x3])
    x = Dropout(0.5)(x)

    x = Dense(128)(x)
    x = LeakyReLU(alpha=0.5)(x)
    x = Dense(128)(x)
    x = LeakyReLU(alpha=0.5)(x)
    x = Dense(128)(x)
    x = LeakyReLU(alpha=0.5)(x)

    x = Dropout(0.5)(x)

    x = Dense(4, activation='softmax')(x)
    out = ClassWeightMult.ClassWeightMult(class_weight)(x)

    model = Model(inp, out)
    return model
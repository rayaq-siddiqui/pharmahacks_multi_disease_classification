import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Concatenate, Maximum, Dropout, LeakyReLU
from tensorflow.keras.models import Model
from . import functions
from . import ClassWeightMult


def simple_branchy_linear_network(class_weight):
    inp = Input(shape=(1094,))
    x = Dense(2048)(inp)
    x = LeakyReLU(alpha=0.5)(x)
    x = Dense(2048)(x)
    x = LeakyReLU(alpha=0.5)(x)
    x = Dropout(0.5)(x)

    x1 = Dense(1024)(x)
    x1 = functions.xtanh(x1)
    x1 = Dense(1024)(x1)
    x1 = functions.xtanh(x1)
    x1 = Dense(128)(x1)
    x1 = functions.xtanh(x1)
    x1 = Dropout(0.5)(x1)

    x2 = Dense(1024)(x)
    x2 = functions.xsigmoid(x2)
    x2 = Dense(512)(x2)
    x2 = functions.xsigmoid(x2)
    x2 = Dense(128)(x2)
    x2 = functions.xsigmoid(x2)
    x2 = Dropout(0.5)(x2)

    x = Concatenate()([x1, x2])

    x = Dense(128)(x)
    x = LeakyReLU(alpha=0.5)(x)
    x = Dense(32)(x)
    x = LeakyReLU(alpha=0.5)(x)

    x = Dropout(0.5)(x)

    x = Dense(4, activation='softmax')(x)
    out = ClassWeightMult.ClassWeightMult(class_weight)(x)

    model = Model(inp, out)
    return model
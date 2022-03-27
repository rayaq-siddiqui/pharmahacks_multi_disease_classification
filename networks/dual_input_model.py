import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Concatenate, Maximum, Dropout, LeakyReLU
from tensorflow.keras.models import Model

def dual_input_model():
    inp = Input(shape=(1094,))

    x1 = Dense(512, activation='relu')(inp)
    x1 = Dense(512, activation='relu')(x1)
    x1 = Dense(256, activation='relu')(x1)
    x1 = Dense(128, activation='relu')(x1)

    x2 = Dense(512, activation='relu')(inp)
    x2 = Dense(512, activation='relu')(x2)
    x2 = Dense(256, activation='relu')(x2)
    x2 = Dense(128, activation='relu')(x2)

    x = Maximum()([x1,x2])
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    out = Dense(4, activation='softmax')(x)

    model = Model(inp, out)
    return model
import argparse
from networks import (
    branchy_linear_network,
    deep_linear_network,
    dual_input_model,
    seq_model,
    simple_branchy_linear_network
)
from data_generator import data_generator
from sklearn.metrics import cohen_kappa_score, f1_score
from tensorflow.keras import optimizers
import tensorflow as tf
import tensorflow_addons as tfa


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process the inputs')
    parser.add_argument(
        '--model', 
        type=str, 
        help='which model would you like to run',
        default='simple_branchy_linear_network'
    )

    args = parser.parse_args()
    model_ = args.model

    X_train, X_test, y_train, y_test, class_weight = data_generator(
        'data/challenge_1_gut_microbiome_data.csv'
    )

    # selecting your model
    if model_ == 'simple_branchy_linear_network':
        model = simple_branchy_linear_network.simple_branchy_linear_network(class_weight)
    elif model_ == 'branchy_linear_network':
        model = branchy_linear_network.branchy_linear_network(class_weight)
    elif model_ == 'seq_model':
        model = seq_model.seq_model()
    elif model_ == 'deep_linear_network':
        model = deep_linear_network.deep_linear_network(class_weight)
    elif model_ == 'dual_input_model':
        model = dual_input_model.dual_input_model()

    print(model.summary())

    # compile the model
    adam = optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(
        optimizer=adam, 
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[
            tfa.metrics.CohenKappa(num_classes=4, weightage='quadratic'),
            tfa.metrics.F1Score(num_classes=4),
            'accuracy'
        ]
    )

    # create call backs
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1,
        min_lr=1e-8,
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'models/checkpoint',
        monitor='val_cohen_kappa',
        verbose=1,
        save_best_only=True,
        mode='max',
        save_weights_only=False
    )

    # fit the model
    model.fit(
        x=X_train,
        y=y_train,
        batch_size=16,
        epochs=12,
        verbose=1,
        validation_data=(X_test, y_test),
        shuffle=True,
        callbacks=[
            reduce_lr,
            checkpoint
        ]
    )

    # evaluate the model
    # load the best model
    model.load_weights('models/checkpoint')

    y_prob = model.predict(X_test) 
    y_classes = y_prob.argmax(axis=-1)

    ck_score = cohen_kappa_score(
        y_test,
        y_classes,
        weights='quadratic'
    )

    f1_score = f1_score(
        y_test,
        y_classes,
        labels=[0,1,2,3],
        average='weighted',
    )

    print('Results:')
    print('cohen kappa score:', ck_score)
    print('f1_score:', f1_score)

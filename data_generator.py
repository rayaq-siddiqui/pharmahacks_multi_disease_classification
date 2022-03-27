import pandas
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


def data_generator(csv='data/challenge_1_gut_microbiome_data.csv'):
    # reading in the csv file
    df = pandas.read_csv(csv)
    bact = df[[i for i in df.columns if i != 'Sample' and i != 'disease']]

    # mean normalization
    mean = bact.mean()
    std = bact.std()
    mean_normalized_df = (bact-mean) / std

    # min max normalization
    min = bact.min()
    max = bact.max()
    min_max_normalized_df = (bact-min) / (max-min)

    # column names
    mndf_cols = mean_normalized_df.columns

    # removes the crazy outliers
    count = 0
    for i, col in enumerate(mndf_cols):
        rows = mean_normalized_df[col]
        for j, row in enumerate(rows):
            if row < -3:
                mean_normalized_df.at[j, col] = -5
            elif row > 3:
                mean_normalized_df.at[j, col] = 5

    # getting the labels
    labels = df['disease']
    possible_labels = ['Disease-1', 'Disease-3', 'Disease-2', 'Healthy']
    for i, label in enumerate(labels):
        if label == possible_labels[0]:
            labels[i] = 0
        elif label == possible_labels[1]:
            labels[i] = 1
        elif label == possible_labels[2]:
            labels[i] = 2
        elif label == possible_labels[3]:
            labels[i] = 3

    # splitting the data
    mean_normalized_df = np.asarray(mean_normalized_df).astype('float32')
    labels = np.asarray(labels).astype('float32')

    X_train, X_test, y_train, y_test = train_test_split(
        mean_normalized_df, 
        labels, 
        test_size=0.2, 
        random_state=3
    )

    # categorizing the data
    y_train_categorical = to_categorical(y_train, num_classes=4)
    y_test_categorical = to_categorical(y_test, num_classes=4)

    # compute class weights
    class_weight = compute_class_weight(
        class_weight='balanced',
        classes = [0,1,2,3], 
        y=labels
    )
    sum = 0
    for i in class_weight:
        sum += i
    class_weight = class_weight / sum

    # return
    return X_train, X_test, y_train, y_test, y_train_categorical, y_test_categorical, class_weight
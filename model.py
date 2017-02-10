import json

import cv2
import numpy as np
import csv

from keras.layers import MaxPooling2D, ELU
from keras.optimizers import Adam
from scipy.misc import imresize
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from cv2 import imread, flip, warpAffine
from random import randint, random, uniform
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf

STEERING_CORRECTION = 0.25
BATCH_SIZE = 200


def get_training_data():
    data = open('data/driving_log.csv')
    reader = csv.reader(data)
    next(reader)
    train = []
    labels = []
    for center_path, left_path, right_path, steering, throttle, breaking, speed in reader:
        steering = float(steering)

        train.append([
            'data/{}'.format(center_path.strip()),
            'data/{}'.format(left_path.strip()),
            'data/{}'.format(right_path.strip()),
        ])
        labels.append(steering)

    return train, labels

def generate_training_data(num, features, labels):
    X_train, y_train = [], []
    for i in range(num):
        f_index = randint(0, len(features)-1)
        feature = features[f_index]
        steering = labels[f_index]

        X_train.append(feature)
        y_train.append(steering)

    return np.array(X_train), np.array(y_train)

def shift_image(image, steering):
    rows, cols, depth = image.shape

    y_trans = 40 * np.random.uniform() - 20 # Shift Y between -20 and 20
    x_trans = 100 * np.random.uniform() - 50 # Shift X between -50 and 50
    M = np.float32([[1, 0, x_trans], [0, 1, y_trans]])
    image = warpAffine(image, M, (cols, rows))

    return image, steering + x_trans/100*.2

def modify_brightness(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  # convert it to hsv
    brightness = .25 + np.random.uniform()
    image[:, :, 2] =  image[:, :, 2] * brightness
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image

def get_training_batch(features, labels, batch_size=128):
    start = 0
    end = batch_size
    while True:
        features, labels = shuffle(features, labels, random_state=randint(1,1000))

        images = features[start:end]
        angles = labels[start:end]

        start = end
        end += batch_size

        if start >= len(features):
            start = 0
            end = batch_size

        X_train = []
        y_train = []
        for angle, image_list in zip(angles, images):

            prob = randint(0,2)
            if prob == 0:
                image = imread(image_list[0]).astype('float32')
            elif prob == 1:
                image = imread(image_list[1]).astype('float32')
                angle += STEERING_CORRECTION
            else:
                image = imread(image_list[2]).astype('float32')
                angle -= STEERING_CORRECTION

            image = imresize(image, (66, 200))

            prob = randint(0,1)
            if prob == 1:
                image = cv2.flip(image, 1)
                angle *= -1

            image, angle = shift_image(image, angle)
            image = modify_brightness(image)

            y_train.append(angle)
            X_train.append(image)

        yield np.array(X_train), np.array(y_train)

def get_validation_batch(features, labels, batch_size=128):
    start = 0
    end = batch_size
    while True:
        images = features[start:end]
        angles = labels[start:end]
        start = end
        end += batch_size

        if start >= len(features):
            start = 0
            end = batch_size

        X_train = []
        for image_list in images:
            image = imread(image_list[0]).astype('float32')

            image = imresize(image, (66, 200))
            X_train.append(image)

        yield np.array(X_train), np.array(angles)

def get_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255. - .5, input_shape=(66, 200, 3)))
    model.add(Convolution2D(3, 1, 1, border_mode='same'))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', activation='relu'))
    model.add(Convolution2D(48, 3, 3, subsample=(2, 2), border_mode='valid', activation='relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid', activation='relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid', activation='relu'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))

    model.add(Dense(1))

    return model


if __name__ == '__main__':
    # Pull data from csv
    features, labels  = get_training_data()
    X_train, y_train = generate_training_data(30000, features, labels)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=randint(1,1000))


    model = get_model()
    model.compile(optimizer=Adam(lr=0.0001), loss='mse')
    history = model.fit_generator(
        get_training_batch(X_train, y_train, BATCH_SIZE),
        samples_per_epoch=len(X_train),
        nb_epoch=10,
        validation_data=get_validation_batch(X_validation, y_validation, BATCH_SIZE),
        nb_val_samples=len(y_validation)
    )
    model.save('model.h5', True)

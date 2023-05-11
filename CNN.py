
# Use Keras to create and train a CNN model on the training set of compressed 
# images from the compressed data folder, then display the model's accuracy.

# I use an early stopping callback with patience set to 100 epochs.
# The weights that yield the best results on the validation set
# are kept. Additionally, I use hinge loss as the loss function.

import cv2
import os

from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.callbacks import EarlyStopping


# Define the early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)


# Specify the CNN model to create.

def create_model(input_shape, num_classes):
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


# Get the file paths of all the compressed images.

cwd = os.getcwd()

cdataset = os.path.join(cwd, 'compressed data')
capples = os.path.join(cdataset, 'compressed apples')
ctomatoes = os.path.join(cdataset, 'compressed tomatoes')

cpaths = [capples, ctomatoes]
folders = [os.listdir(capples), os.listdir(ctomatoes)]


# Convert compressed image data into numpy arrays
# and label the data according to the subfolder
# the images belong to. Labels are 
# 1 (for apples) and 0 (for tomatoes).

first_image = True

for i in range(2):
    for file in tqdm(folders[i]):

        image = cv2.imread(os.path.join(cpaths[i], file))

        if first_image:
            X = [image]
            first_image = False

        else:
            X = np.vstack((X,[image]))

y = np.zeros(len(os.listdir(capples)) + len(os.listdir(ctomatoes)))
y[len(os.listdir(capples)):] = 1

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1 , train_size=0.9)

num_classes = 2
input_shape = x_train.shape[1:]


# Preprocess the data
# The labels are one-hot-encoded so the label 0 becomes [1., 0.]
# and the label 1 becomes [0., 1.].

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Create the model
model = create_model(input_shape, num_classes)

# Compile the model
model.compile(loss='hinge',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model with early stopping
model.fit(x_train, y_train, batch_size=32, epochs=200, validation_split=0.2, callbacks=[early_stop])

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('Test accuracy:', test_acc)

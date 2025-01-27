# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 16:13:02 2019

@author: Ramya Aravind
"""

#Building CNN
#Part 1 - Importing Keras libraries and Packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initialise CNN
classifier = Sequential()

#Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3) , activation='relu'))

#Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Step 3 - Flattening
classifier.add(Flatten())

#Step 4 - Full connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

#Compilation
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])

#Image data generator
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)




















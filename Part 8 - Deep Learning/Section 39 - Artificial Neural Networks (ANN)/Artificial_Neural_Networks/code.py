# -*- coding: utf-8 -*-

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

#Label categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# replace Country France, Spain & Germany with 0,1,2
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[: ,1])

# replace Sex Female and Male with 0,1
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[: ,2])

# Create dummy variable for Country
ct = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

X = X[:, 1:]

#Splitting dataset into training and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#Preprocessing data standard scaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#import keras
import keras

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

#adding the first hidden layer (units = input dim)
classifier.add(Dense(units=6, activation="relu", input_dim = 11, kernel_initializer="uniform"))

#Adding the second hidden layer
classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform"))

#Adding the final output layer (units=1 means the output is categorical)
classifier.add(Dense(units=1, activation="sigmoid", kernel_initializer="uniform"))

#Compiling the ANN
classifier.compile(optimizer="adam", loss = "binary_crossentropy", metrics = ["accuracy"])

#training the set using ANN
classifier.fit(X_train, Y_train, batch_size=10, nb_epoch=100)

#Running on test set
y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)

#Confusion matrix to test accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)


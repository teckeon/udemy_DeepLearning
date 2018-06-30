#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 07:56:51 2018

@author: teckeon
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values # this takes the independed from 3 to 12
y = dataset.iloc[:, 13].values # this takes the dependent variable at 13

# Encoding categorical data
# since we have to categorical data fields with text, we are duplicating
# first object will be assigned  X_1
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) #index 1
# second categorical data object will be assigned  X_2
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2]) # now the next one index 2

# first categorical data has 3 different independent variables. So next set of code takes care fo dummy variable
# index 1
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

#lets remove first column to avoid dummy trap
# take all columns ":," except for first one "1:"
X = X[:, 1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Part 2 - Now let's make the ANN:
# This part 4 Section 19 in the udemy Lecture


# Importing the Keras libraries and packages

import keras 
from keras.models import Sequential
from keras.layers import Dense


# Initialising the ANN
classifier = Sequential()


# Adding the input layer and the first hidden layer
# output_dim is currently 6 (11 nodes + 1 / 2) later will teach a better
# way to identify the output_dim in part 10 of the course
classifier.add(Dense(kernel_initializer="uniform", activation="relu", input_dim=11, units=6))

# Adding the second hidden layer
# notice that input_dim is not here since we defined it in the first hidden 
# layer
classifier.add(Dense(kernel_initializer="uniform", activation="relu", units=6))

# Adding the output layer

classifier.add(Dense(kernel_initializer="uniform", activation="sigmoid", units=1))

# Compliling the ANN
# since we are looking for a binary we use binary_crossentropy, if there are more than we will use a different one
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


# Part 3 - Making the predictions and evaluating the model





# Fitting classifier to the Training set
# Create your classifier here

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
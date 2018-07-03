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
from keras.layers import Dropout # added after discussion on part 4

# Initialising the ANN
classifier = Sequential()


# Adding the input layer and the first hidden layer with dropout
# output_dim is currently 6 (11 nodes + 1 / 2) later will teach a better
# way to identify the output_dim in part 10 of the course
classifier.add(Dense(kernel_initializer="uniform", activation="relu", input_dim=11, units=6))
classifier.add(Dropout(p = 0.1)) # experient with value when it stops overfitting but try to say below .05

# Adding the second hidden layer
# notice that input_dim is not here since we defined it in the first hidden 
# layer
classifier.add(Dense(kernel_initializer="uniform", activation="relu", units=6))
classifier.add(Dropout(p = 0.1)) # experient with value when it stops overfitting but try to say below .05
# Adding the output layer

classifier.add(Dense(kernel_initializer="uniform", activation="sigmoid", units=1))

# Compliling the ANN
# since we are looking for a binary we use binary_crossentropy, if there are more than we will use a different one
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)



# Part 3 - Making the predictions and evaluating the model
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) # less than 50 may leave, higher than 50 may stay

# Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""

new_prediction = classifier.predict(sc.transform(np.array([[0.0,0,600, 1, 40, 3,60000, 2, 1,1,50000]]))) # sc.transform to ensure we have it in the same scale. The first field is 0 but added it as 0.0 to avoid the float warning when executing

new_prediction = (new_prediction > 0.5)



# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Part 4 - Evaluating, Improving and Tuning the ANN
# Here we will be running part 1 to prepare the data so we may do K-Fold Cross validation in this section

# Evaluating the ANN
# Keres wrapper
from keras.wrappers.scikit_learn   import KerasClassifier

from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

# this function architecture to build the ANN
def build_classifier():
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
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)

mean = accuracies.mean()
variances = accuracies.std()

# Improving the ANN
# Droupout Regularization to reduce overfitting if needed



# Tuning the ANN
# this Section and part 1 above are executed. The analysis takes a long time to finish. 

from keras.wrappers.scikit_learn   import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

# this function architecture to build the ANN
def build_classifier(optimizer):
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
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)

parameters = {'batch_size': [25, 32],
              'epochs':[100, 500],
              'optimizer':['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = classifier,
                          param_grid = parameters,
                          scoring = 'accuracy',
                          cv=10)

grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_




















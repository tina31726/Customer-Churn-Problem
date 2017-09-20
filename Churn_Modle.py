#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 22:10:50 2017

@author: YiChen
"""


# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
# find indepedent variables with x matrix and get result column with y matrix
X = dataset.iloc[:,3:13].values
Y = dataset.iloc[:,13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#change string variable to nominal variable
labelEncoder_x_1=LabelEncoder()
X[:,1]=labelEncoder_x_1.fit_transform(X[:,1])
labelEncoder_x_2=LabelEncoder()
X[:,2]=labelEncoder_x_2.fit_transform(X[:,2])
# then change value of nominal variable to non-continous values(dummy variables) 
oneHotEncode_x=OneHotEncoder(categorical_features=[1])
#using toarray to store dense matrix from sparese matrix (CSR) 
X = oneHotEncode_x.fit_transform(X).toarray()
#avoiding dummy variable trap, we need to drop 1 column (cuz we have 3 categories, using 2 column is enough)
X=X[ : ,1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
#Taking the average of input&output nodes is based on common practice, but it doesn't mean it is the best architecture. 
classifier.add(Dense( input_dim = 11, units = 6, activation = 'relu' , kernel_initializer = 'uniform'))
#add Dropout
claasifier.add(Dropout(p = 0.1))
#in order to prevent overfitting, Dropout randomly input to 0  neurons based on parameter p (p value not over 0.5 otherwise will underfitting)

# Adding the second hidden layer
classifier.add(Dense(  units = 6, activation = 'relu' , kernel_initializer = 'uniform'))

claasifier.add(Dropout(p = 0.1))

# Adding the output layer
#using 'sigmoid' to get probabilities for the outcome
classifier.add(Dense(  units = 1, activation = 'sigmoid' , kernel_initializer = 'uniform'))
#deal with dependent variable having three  more categories in outout, let units = num of categories, also activation = softmax 
#classifier.add(Dense(units = n, activation='softmax'))

#Model visualization
from keras.utils import plot_model
plot_model(classifier, to_file='classifier.png', show_shapes=True)

# Compiling the ANN
# optimizer :use optimizerto find the optimal weight, loss function: use logarithmic loss(logistic regression), metrics:criterion to evalute model  
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(x_train,y_train,batch_size = 10,epochs = 100)


# Part 3 - Making predictions and evaluating the model


# Predicting the Test set results
y_pred = classifier.predict(x_test)
y_pred = (y_pred>0.5)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
 
 
 
 
# Homework - predict customer leave the bank or not
#Geography: France
#Credit Score: 600
#Gender: Male
#Age: 40 years old
#Tenure: 3 years
#Balance: $60000
#Number of Products: 2
#Does this customer have a credit card ? Yes
#Is this customer an Active Member: Yes
#Estimated Salary: $50000
 
 z_test = np.array([[0,0,600,1,40,3,6.e+04,2,1,1,5.e+04]])
 #horizantal vetcor initialize : [[""]]
 
sc = StandardScaler()
z_test = sc.fit_transform(z_test)
 
#z_test = array([0,0,600,1,40,3,6.e+04,2,1,1,5.e+04])
z_pred = classifier.predict(z_test)
z_pred = (z_pred>0.5)

# Part 4 - Evaluating, Improving and Tuning the ANN

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def buiild_classifier():
    # Initialising the ANN
    classifier = Sequential()
    # Adding the input layer and the first hidden layer
    classifier.add(Dense( input_dim = 11, units = 6, activation = 'relu' , kernel_initializer = 'uniform'))
    # Adding the second hidden layer
    classifier.add(Dense(  units = 6, activation = 'relu' , kernel_initializer = 'uniform'))
    # Adding the output layer
    classifier.add(Dense(  units = 1, activation = 'sigmoid' , kernel_initializer = 'uniform'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifer = KerasClassifier(build_fn = buiild_classifier, batch_size = 10, epochs = 100)

import multiprocessing
if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')
accuracies = cross_val_score(estimator = classifer, X = x_train, y = y_train, cv = 10, n_jobs = -1)
#n_jobs : num of CPU to do the computation, -1 means all CPU

mean = accuracies.mean()
variance = accuracies.std()




# Improving the ANN
# Dropout Regularization to reduce overfitting if needed

# Tuning the ANN

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def buiild_classifier(neurons ):
    # Initialising the ANN
    classifier = Sequential()
    # Adding the input layer and the first hidden layer
    classifier.add(Dense( input_dim = 11, units = neurons, activation = 'relu' , kernel_initializer = 'uniform'))
    # Adding the second hidden layer
    classifier.add(Dense(  units = neurons, activation = 'relu' , kernel_initializer = 'uniform'))
    # Adding the output layer
    classifier.add(Dense(  units = 1, activation = 'sigmoid' , kernel_initializer = 'uniform'))
    classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifer = KerasClassifier(build_fn = buiild_classifier, batch_size = 32, epochs = 500)
#parameter_list = { 'batch_size':[25,32], 'epochs': [100,500], 'opt_algorithm':['adam', 'rmsprop'] }

#final_result = 'batch_size':32  'epochs': 500  'opt_algorithm': rmsprop

#next: test num_neuron
neurons = [5,6,7,8,9]
parameter_list = dict(neurons = neurons)

import multiprocessing
if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')
grid_search = GridSearchCV(estimator = classifer, param_grid = parameter_list, scoring = 'accuracy', cv = 10, n_jobs = 2, pre_dispatch = '2*n_jobs')
grid_search = grid_search.fit(x_train, y_train)

best_parameter = grid_search.best_params_
best_accuracy = grid_search.best_score_

# summarize results
print("Best: %f using %s" % (grid_search.best_score_, grid_search.best_params_))
means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
params = grid_search.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
#final_resulr = 'neuron' = 9


import numpy as np
import matplotlib.pyplot as plt # install this pip
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as sc
from keras.models import Sequential as classifier
from keras.layers import Dense


# labelencoder_

dataset = pd.read_csv('case_trackers.csv')

# TODO Use an encoder to switch from f and t to 0 and 1. Done with libre for now

# Split into input and output matrices
input_matrix = dataset.iloc[:, 1:16].values
output_vector = dataset.iloc[:,17].values

# Split into training:testing -> 80:20 ratio
input_matrix_train, input_matrix_test, output_vector_train, output_vector_test = \
    train_test_split(input_matrix,output_vector, test_size=0.2)

# This normalises the data - not needed in our 1/0 output system
# input_matrix_train = sc.fit_transform(input_matrix_train)
# input_matrix_test = sc.fit_transform(input_matrix_test)

# NEURAL NETWORK STUFF BELOW

# Input layer and first hidden layer.
# TODO Learn how to choose a good output dim.
# init = uniform - to initialise the stochastic gradient descent (SGD) algo - see wikipedia
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim=17))

# 2nd hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation='relu'))

# Adding an output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation='sigmoid'))

# COMPILING NEURAL NETWORK
# TODO What does compiling a neural network do?
# adam is an SGD algorithm.
# loss is a parameter needed for SGDs, and we use the binary version.
# We want accuracy to drive our performance.
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# FITTING OUR MODEL
# This changes the weightings placed on each field.
# Batch size affects how many observations before weightings are reevaluated.
# Epoch is total number of iterations - No real rules for how often to iterate.
classifier.fit(input_matrix_train, output_vector_train, batch_size = 10, nb_epoch = 100)

# Predicting the Test set results
output_vector_pred = classifier.predict(output_vector_test)
output_vector_pred = (output_vector_pred > 0.5)

# Assess our predictions
cm = confusion+matrix(output_vector_test, output_vector_pred)

# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 22:01:54 2019

@author: Saran
"""
"""
Our goal is to create a LSTM on a single stock 
Note: the paper does not mention the layers of the LSTM
We are going to build this example with 4 layers and 1 factor
We need to rebuild for all 7 factors
"""
from data import Stock_Data
import numpy as  np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
import datetime as dt
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.python.keras import backend





#high_prices = Stock_Data.stock_df.loc[:,'high'].as_matrix()
#low_prices = Stock_Data.stock_df.loc[:,'low'].as_matrix()
#mid_prices = (high_prices+low_prices)/2.0

#Identify training data set
training_set = Stock_Data.stock_df.iloc[0:2517:,4:5].values

# Feature Scaling
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 2517):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



# Part 2 - Building the RNN


# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))
regressor.add(Activation('relu'))


# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 75, batch_size = 32)
regressor.summary()



#Making the predictions and visualising the results

testing_set = Stock_Data.stock_df.iloc[2517:5033:,4:5].values


# Getting the predicted stock price of 2017
dataset_total = np.concatenate((training_set[:,], testing_set[:,]), axis = 0)


inputs = dataset_total[len(dataset_total) - len(testing_set) - 60:]
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []

for i in range(60, 2517):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(testing_set, color = 'red', label = 'Real Close Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Close Price')
plt.title('Close Price Prediction')
plt.xlabel('Time')
plt.ylabel('GE Close Price')
plt.legend()
plt.show()

import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import pymongo, re, datetime
from export_csv import load_dataset, class_selection

# fix random seed for reproducibility
seed = 7
numpy.random.seed(7)

# load the dataset
WINDOW_LEN = 400
look_back = WINDOW_LEN
dataset, label = load_dataset(WINDOW_LEN)

trainX, testX, trainY, testY = train_test_split(dataset, label, test_size=0.33, random_state=seed)
# split into train and test sets
# train_size = int(len(dataset) * 0.8)
# test_size = len(dataset) - train_size
# trainX, testX = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# trainY, testY = label[0:train_size,:], label[train_size:len(dataset),:]

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# trainY = numpy.reshape(trainY, (trainY.shape[0], 1, trainY.shape[1]))
# testY = numpy.reshape(testY, (testY.shape[0], 1, testY.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(120, input_shape=(1, look_back), dropout=10))
model.add(Dense(2, activation='softmax'))
rmsdrop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
adagrad = optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=adagrad)
# model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=200, batch_size=64, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredict, testPredict = class_selection(trainPredict, testPredict)

print(testY)
print(testPredict)

# calculate accuracy
train_total = len(trainPredict)
test_total = len(testPredict)
train_correct = 0
test_correct = 0

for x,y in zip(trainY, trainPredict):
	if numpy.array_equal(x, y):
		train_correct+=1

for x,y in zip(testY, testPredict):
	if numpy.array_equal(x, y):
		test_correct+=1

trainScore = train_correct / train_total
print('Train Accuracy: %.2f ' % (trainScore))
testScore = test_correct / test_total
print('Test Accuracy: %.2f ' % (testScore))

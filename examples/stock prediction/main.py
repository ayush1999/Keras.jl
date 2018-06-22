import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, Flatten
import time

## Loading and preprocessing.

prices_dataset = pd.read_csv("prices.csv")
yahoo = prices_dataset[prices_dataset["symbol"] == "YHOO"]
yahoo_prices = yahoo.close.values.astype('float32')
yahoo_stock_prices = yahoo_prices.reshape(1762, 1)


train_size = int(len(yahoo_stock_prices) * 0.80)
test_size = len(yahoo_stock_prices) - train_size
train, test = yahoo_stock_prices[0:train_size,
                                 :], yahoo_stock_prices[train_size:len(yahoo_stock_prices), :]

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX)/max(dataset), np.array(dataY)

look_back = 20
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# 20 timesteps
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
print(trainX.shape)
print(testX.shape)
np.save("testX.npy", testX)
np.save("testY.npy", testY)
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1), recurrent_activation="sigmoid",
               recurrent_initializer='glorot_uniform',
               kernel_initializer='glorot_uniform', return_sequences=True))
model.add(Flatten())
model.add(Dense(100, activation="relu"))
model.add(Dense(1, activation="relu"))

start = time.time()
model.compile(loss='mse', optimizer='adam')
print('compilation time : ', time.time() - start)
model.fit(
    trainX,
    trainY,
    batch_size=128,
    epochs=250,
    validation_split=0.05)

model.save_weights("weights.h5")
with open("structure.json","w") as f:
    f.write(model.to_json())

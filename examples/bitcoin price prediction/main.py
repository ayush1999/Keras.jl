import pandas as pd
import numpy as np
from keras.layers import LSTM, Dense, Flatten, Dropout
from keras.models import Sequential

data = pd.read_csv("bitcoin.csv")

# Predicting the price based on the open, high, low, close of the previous 15 days
train = data[:10000] / 1000000
test = data[350000:] / 1000000

ip = []
for i in range(5000):
    temp = []
    for j in range(i, i + 15):
        temp1 = []
        temp1.append(train.iloc[j]["Open"])
        temp1.append(train.iloc[j]["High"])
        temp1.append(train.iloc[j]["Low"])
        temp1.append(train.iloc[j]["Close"])
        temp.append(temp1)
    ip.append(temp)

ip = np.array(ip)
op = []
for i in range(5000):
    e = train.iloc[i + 15]["Weighted_Price"]
    op.append([e])

model = Sequential()
model.add(LSTM(100, input_shape=(15, 4), recurrent_activation="sigmoid"))
model.add(Dropout(0.2))
model.add(Dense(80))
model.add(Dense(40))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")
model.fit(ip, op, epochs=250)

# Save model
model.save_weights("weights.h5")

with open("structure.json","w") as f:
    f.write(model.to_json())

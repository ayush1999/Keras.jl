import numpy as np
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=5000)
X_train = pad_sequences(X_train, maxlen=500)
X_test = pad_sequences(X_test, maxlen=500)

# Save the test data
np.save("test_x.npy", X_test)
np.save("test_y.npy", y_test)

# Model
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten

model = Sequential()
model.add(Embedding(5000, 32, input_length=500))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, validation_data=(
    X_test, y_test), epochs=2, batch_size=128, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))

# Save the model

with open("model_structure.json","w") as f:
    f.write(model.to_json())

model.save_weights("model_weight.h5")
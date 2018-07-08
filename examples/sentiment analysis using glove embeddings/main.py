import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.optimizers import Adam
from keras.layers import BatchNormalization, Flatten, Conv1D, MaxPooling1D
from keras.layers import Dropout

data = pd.read_csv("Sentiment Analysis Dataset.csv", error_bad_lines=False)

data_new = data[:50000]	

# define text data
docs_combined = data_new['SentimentText']

# initialize the tokenizer
t = Tokenizer()
t.fit_on_texts(docs_combined)
vocab_size = len(t.word_index) + 1

# integer encode the text data
encoded_docs = t.texts_to_sequences(docs_combined)

# pad the vectors to create uniform length
padded_docs_combined = pad_sequences(encoded_docs, maxlen=500, padding='post')

# seperate the train and test sets

df_train_padded = padded_docs_combined[:45000]
df_test_padded = padded_docs_combined[45000:]

df_train_y = []
for ele in data_new["Sentiment"][:45000]:
    df_train_y.append([ele])

# load the glove840B embedding into memory after downloading and unzippping

embeddings_index = dict()
f = open('glove/glove.twitter.27B.200d.txt')

for line in f:
    # Note: use split(' ') instead of split() if you get an error.
	values = line.split(' ')
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()

print('Loaded %s word vectors.' % len(embeddings_index))


# create a weight matrix
embedding_matrix = np.zeros((vocab_size, 200))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

model = Sequential()
e = Embedding(vocab_size, 200, weights=[embedding_matrix],
              input_length=500, trainable=False)
model.add(e)
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Dropout(0.2))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Dropout(0.2))
model.add(Conv1D(64, 3, activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the model
Adam_opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=Adam_opt, loss='binary_crossentropy', metrics=['acc'])

model.fit(df_train_padded, df_train_y, epochs=45, verbose=1)

# save the data

model.save_weights("weights.h5")

with open("structure.json","w") as f:
    f.write(model.to_json())

# Save test data

df_test_y = []
for ele in data_new["Sentiment"][45000:]:
	df_test_y.append([ele])
	
np.save("Y.npy", df_test_y)
np.save("X.npy", df_test_padded)
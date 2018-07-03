# Sentiment Analysis using pretrained GloVe embeddings

## Requirements:

You need to have two files in order to train the model:
1. The twitter GloVe pretrained word embeddings, which can be downloaded from [here](http://nlp.stanford.edu/data/glove.twitter.27B.zip).

2. The Sentiment Analysis dataset, which can be found [here](http://thinknook.com/wp-content/uploads/2012/09/Sentiment-Analysis-Dataset.zip).


## How?

Training the model and producing the weight, structure file is simple, a `python main.py` should
do it. Four files are produced:

1. `weights.h5` file, containing the model weights.
2. `structure.json` file, containing the model structure.
3. `X.npy` the input to the model.
3. `Y.npy` the expected output.


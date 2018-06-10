# IMDB Sentiment Analysis.

## Running the Keras model.

Running the Keras model is pretty simple. From the terminal, run `python main.py`. This will train the model in Keras. (Note: It will download the IMDB dataset , if not present.). Along with this, 4 new files are created:

1. `test_x.npy`: Numpy serialized test data.

2. `test_y.npy`: Numpy serialized test target data.

3. `model_structure.json`: The Keras model structure, in JSON format.

4. `model_weight.h5`: The model weights file.

## Loading it in Flux.

Now that we have the required files, the model can be loaded into Flux easily. From the terminal, run `julia main.jl`. This will load  the entire model in Flux ,test it on the test files and return the accuracy achieved. (Note: Testing the model may take some time. However, you can reduce this by changing the number of test cases to test on (default value is 25000))
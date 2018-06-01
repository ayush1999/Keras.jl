# Keras.jl
Load Keras models in Julia.
[W.I.P]

This package can be used to read and load a Keras model into Flux. Please note: Currently, I'm working only on supporting models related to Computer Vision, will add support for other networks like LSTMs, RNNs soon. 

## How?

Loading a model in Flux is fairly simple. Clone this repository into `~/.julia/v0.6`. Make sure you have all dependencies installed. In order to load a model, you need to have two files:
1. The `model.json` file. This stores the structure of the model. This can be obtained from any Keras model using the `model.to_json()` method.
2. The `weights.h5` file. This stores the weights associated with different layers of the pre-trained Keras model. This file can be produced from a Keras model using `Keras.save_weights(weight_file_name)`.

(The files can have any other name (as long as they are in the correct format). I'm using model.json and weights.h5 as an example here)

Now that both files are present, the model can be loaded as :

```
>>> using Keras

>>> model, weight = Keras.load("model.json", "weights.h5")
```

`model` is now the corresponding model in Flux. This can be used directly as:

```
>>> model(rand(28,28,1,1))
```

## Issues

Since this is currently under development, feel free to open any issue you encounter.

### To Do:

Check if yaml support is needed. (No official Yaml parser present for Julia).

Verify operators. (Add more tests for new ops)

Investigate approaches for `Add` layer.
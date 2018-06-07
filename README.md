# Keras.jl
Load Keras models in Julia.

This is not a wrapper around [Keras](keras.io). This is built on top of [Flux](https://github.com/FluxML/Flux.jl), to directly load Keras models into Flux.
[W.I.P]

## How?

Loading a model in Flux is fairly simple. Clone this repository into `~/.julia/v0.6`. Make sure you have all dependencies installed. In order to load a model, you need to have two files:
1. The `model.json` file. This stores the structure of the model. This can be obtained from any Keras model using the `model.to_json()` method.
2. The `weights.h5` file. This stores the weights associated with different layers of the pre-trained Keras model. This file can be produced from a Keras model using `Keras.save_weights(weight_file_name)`.

(The files can have any other name (as long as they are in the correct format). I'm using model.json and weights.h5 as an example here)

Keras models can broadly be divided into two categories:

1. The models using the `sequential` API.
2. The models using the `functional` API. (Also called `Model` API)

Due to subtle differences in their structure and functioning, you need to follow different steps to run these models in Flux. You can check the type of the model by:
```
>>> using Keras

>>> Keras.check_modeltype("model.json")
```

## Running Sequential Models 

```
>>> using Keras

>>> model, weight = Keras.load("model.json", "weights.h5")
```

`model` is now the corresponding model in Flux. This can be used directly as:

```
>>> model(rand(28,28,1,1))
```

Another straight-forward way of running such models is:
```
>>> using Keras

>>> Keras.load("model.json", "weights.h5", ip)
```
Where `ip` is our input. This directly returns the models output.

## Running Functional Models.

Functional models can be tricky as they may consist of a number of sub-graphs within themselves. Running such models is similar to the second way of running Sequential models mentioned above.

```
>>> using Keras

>>> Keras.load("model.json", "weight.h5", ip)
```
Where `ip` is the input to our model. This directly returns the output. (Note: Currently there is no other way of running functional API models).

## Issues

Since this is currently under development, feel free to open any issue you encounter.

### Current Impediments:

[Lambda](https://keras.io/layers/core/#lambda) layers cannot be handled at this moment. This is because we'd need to handle the Python AST, for parsing it as JSON.

### To Do:

Check if yaml support is needed. (No official Yaml parser present for Julia).

Verify operators. (Add more tests for new ops)

Investigate approaches for `Add` layer.
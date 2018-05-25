# Keras.jl
Loads Keras models in Julia.
[W.I.P]

To Do:

1. Create graph from the layers 
2. Chainify all the layers
3. Get it running for MNIST
4. If it works, add ops support for other layers.

Current Working:

```
>>> using Keras
>>> model, weight = Keras.load("model_structure.json", "model_weights.h5")      #Returns the model and the weights
>>> model(rand(Float32, 28,28,1,1,))                         #Returns the model's prediction.                                                                          
10-element Array{Float32,1}:
  0.582318
 -0.712256
  0.915128
 -0.745874
 -0.386354
  0.791499
  3.99255 
 -1.76445 
  0.369088
 -1.81744 
```


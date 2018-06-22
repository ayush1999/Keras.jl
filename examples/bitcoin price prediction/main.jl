using Keras
using Pyall
@pyimport numpy

model = Keras.load("structure.json", "weights.h5")


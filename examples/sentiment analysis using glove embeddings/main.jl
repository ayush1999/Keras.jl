using Keras
using Flux
using PyCall
@pyimpory numpy

X = numpy.load("X.npy")
Y = numpy.load("Y.npy")

model = Keras.load("structure.json", "weights.h5")

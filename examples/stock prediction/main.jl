using Keras
using PyCall
using Plots
@pyimport numpy

ip = numpy.load("ip.npy")
op = numpy.load("op.npy")

op_obt = []

for x=1:100
    m = Keras.load("structure.json", "weights.h5")
    push!(op_obt, m(ip[x,:,:])[1])
end

plot(op_obt)
using Keras
using PyCall
using Plots
@pyimport numpy

ip = numpy.load("testX.npy")
op = numpy.load("testY.npy")

op_obt = []
m = Keras.load("structure.json", "weights.h5")
println(m(ip[1,:,:]))
#for x=1:100
#    m = Keras.load("structure.json", "weights.h5")
#   push!(op_obt, m(ip[x,:,:])[1])
#end

#plot(op_obt)
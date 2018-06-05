# Model tests, to ensure that the loaded models are running 
# with a decent accuracy.

# Tests files have been taken from ONNX.
# You need PyCall and numpy installed to run these tests.

using Keras
using PyCall
@pyimport numpy

function read_input(filename)
    f = numpy.load(open(filename))
    return convert(Array{Float64, N} where N, reshape(permutedims(f, (5,4,3,2,1)), 224,224,3,1))
end

function read_output(filename)
    f = numpy.load(open(filename))
    return reshape(permutedims(f, (3,2,1)), 1000,1)
end

# ResNet test
loss(x, y) = sum((x .- y).^2)
netloss = 0
for x=0:2
    output = Keras.load("resnet.json", "resnet.h5", read_input("inputs$x.npy"))
    op_expected = read_output("outputs$x.npy")
    netloss += loss(output, op_expected)
end

println("Average net loss is ", netloss/3)

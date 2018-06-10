using PyCall
@pyimport numpy

x = numpy.load("test_x.npy")
y = numpy.load("test_y.npy")

num_tests = 2246
println("Loading model...")
using Keras
m = Keras.load("model-structure.json", "model-weights.h5")
println("Model loaded, Testing....")
count = 0
for i=1:num_tests
    if (findmax(y[i,:])[2] == findmax(m(x[i,:]))[2])
        count += 1
    end
end

println("Model tested: $(count*100/num_tests)% accuracy achieved.")
using PyCall
@pyimport numpy
X = numpy.load("test_x.npy");
Y = numpy.load("test_y.npy");

# Get the predicted value
round_output(x) = x[1]>=0.5?1:0

using Keras

println("Loading model...")
model = Keras.load("model_structure.json", "model_weight.h5")
println("Model loaded successfully.")
# Test
println("Testing model...")
count  = 0 
for i=1:1000
    if round_output(model(X[i,:])) == Y[i]
        count += 1
    end
end

println("Test complete: $(count/10) % accuracy achieved.")
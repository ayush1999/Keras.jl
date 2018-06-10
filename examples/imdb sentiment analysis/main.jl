using PyCall
@pyimport numpy
X = numpy.load("test_x.npy");
Y = numpy.load("test_y.npy");

# Get the predicted value
round_output(x) = x[1]>=0.5?1:0

using Keras

#Change this value to test results on smaller datasets
num_tests = 25000

println("Loading model...")
model = Keras.load("model_structure.json", "model_weight.h5")
println("Model loaded successfully.")
# Test
println("Testing model, Please wait. This may take a few minutes.")
count  = 0 
for i=1:num_tests
    if round_output(model(X[i,:])) == Y[i]
        count += 1
    end
end

println("Test complete: $(count*100/num_tests) % accuracy achieved.")
# Model tests, to ensure that the loaded models are running 
# with a decent accuracy.

# Tests files have been taken from ONNX.
# You need ONNX.jl and ProtoBuf.jl installed to run these tests.

using ONNX
using ProtoBuf
using Keras

function read_value(filename)
    return ONNX.get_array(readproto(open(filename), ONNX.Proto.TensorProto()))
end

# Test loop
loss(x, y) = sum((x .- y).^2)
netloss = 0
for x=0:2
    input = read_value("input_$x.pb")
    output = Keras.load("vgg19.json", "vgg19.h5", input)
    op_expected = read_value("output_$x.pb")
    netloss += loss(output, op_expected)
end

print("Average Net loss is ", netloss)

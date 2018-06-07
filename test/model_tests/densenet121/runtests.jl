using ONNX
using Keras
using ProtoBuf
using Flux

function read_input(filename)
    return convert(Array{Float64,N} where N, readproto(open(filename), ONNX.Proto.TensorProto()) |> ONNX.get_array)
end

netloss = 0
loss(x, y) = sum((x .- y).^2)

for x=0:2
    output = softmax(Keras.load("structure.json", "weights.h5", read_input("test_data_set_$x/input_0.pb")))
    output_expected = softmax(vec(read_input("test_data_set_$x/output_0.pb")))
    netloss += loss(output, output_expected)
end

println("Net loss is ", netloss)
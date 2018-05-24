ops = Dict{Symbol, Any}()
weight = Keras.weights()
ops[:Input] = function(a)
    return vcall(:.+, a, 0)
end

ops[:Conv] = function(a)
    activation = a.fields["activation"]
    kernel_weight = permutedims(weight[a.fields["name"]][a.fields["name"]]["kernel:0"], (4,3,2,1))
    kernel_bias = weight[a.fields["name"]][a.fields["name"]]["bias:0"]
    strides = (a.fields["strides"]...)
    pads = (0,0)
    return vcall(:Conv, Symbol(activation), kernel_weight, kernel_bias, strides, pads)
end

ops[:Dropout] = function(a)
    return vcall(:Dropout, a.fields["rate"])
end

ops[:MaxPool] = function(a)
    return vcall(x->maxpool(x, (a.fields["pool_size"]...), pads=(0,0), strides=(a.fields["strides"]...)))
end

ops[:Dense] = function(a)
    weight = weights()[a.name][a.name]["kernel:0"]
    bias = weights()[a.name][a.name]["bias:0"]
    return vcall(:Dense, weight, bias)
end

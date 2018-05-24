ops = Dict{Symbol, Any}()

ops[:Input] = function(a)
    return vcall(:.+, a, 0)
end

ops[:Conv] = function(a)
    activation = a.fields["activation"]
    kernel_weight = permutedims(weights[g[1].fields["name"]][g[1].fields["name"]]["kernel:0"], (4,3,2,1))
    kernel_bias = weights[g[1].fields["name"]][g[1].fields["name"]]["bias:0"]
    strides = (g[1].fields["strides"]...)
    pads = (0,0)
    return vcall(:Conv, activation, kernel_weight, kernel_bias, strides, pads)
end

ops[:Dropout] = function(a)
    return vcall(:Dropout, a.fields["rate"])
end

ops[:MaxPool] = function(a)
    return vcall(x->maxpool(x, (a.fields["pool_size"]...), pad=(0,0), stride=(a.fields["stride"])...)
end

ops[:Dense] = function(a)
    weight = weights()[a.name][a.name]["kernel:0"]
    bias = weights()[a.name][a.name]["bias:0"]
    return vcall(:Dense, weight, bias)
end

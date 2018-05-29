ops = Dict{Symbol, Any}()
ops[:Input] = function(a)
    return vcall(:.+, a, 0)
end

ops[:Conv] = function(a)
    activation = a.fields["activation"]
    if activation=="linear"
        activation = "relu"
    end
    kernel_weight = reshape(weight[a.fields["name"]][a.fields["name"]]["kernel:0"],
                             reverse(size(weight[a.fields["name"]][a.fields["name"]]["kernel:0"])))
    kernel_bias = weight[a.fields["name"]][a.fields["name"]]["bias:0"]
    strides = (a.fields["strides"]...)
    if a.fields["padding"] == "valid"
        pads = (0,0)
    elseif a.fields["padding"] == "same"
        pads = (Int64.((a.fields["kernel_size"] .-1)./2)...)
    end
    return vcall(:Conv, Symbol(activation), kernel_weight, kernel_bias, strides, pads)
end

ops[:Dropout] = function(a)
    return vcall(:Dropout, a.fields["rate"])
end

ops[:MaxPool] = function(a)
    return x->maxpool(x, (a.fields["pool_size"]...), pad=(0,0), stride=(a.fields["strides"]...))
    #return vcall(x->maxpool(x, (a.fields["pool_size"]...), pads=(0,0), strides=(a.fields["strides"]...)))
end

ops[:Flatten] = function(a)
    return :vec
end

ops[:BatchNormalization] = function(a)
    epsilon = a.fields["epsilon"]
    momentum = a.fields["momentum"]
    return x -> BatchNorm(size(x)[3], ϵ=epsilon, momentum=momentum)(x)
end

ops[:Dense] = function(a)
    name = a.fields["name"]
    weight_kernel = weight[name][name]["kernel:0"]
    bias = weight[name][name]["bias:0"]
    if !haskey(a.fields, "activation")
       return Dense(weight_kernel, bias)
    else
        if a.fields["activation"] == "linear"
            a.fields["activation"] = "relu"
        end
        return Dense(weight_kernel, bias), Symbol(a.fields["activation"])
    end
end

ops[:Reshape] = function(a)
    return (x -> reshape(x, (a.fields["target_shape"]...)))
end

ops[:relu] = function(a)
    return :relu
end

ops[:tanh] = function(a)
    return :tanh
end

ops[:sigmoid] = function(a)
    return :sigmoid
end

ops[:elu] = function(a)
    return :elu
end

ops[:softplus] = function(a)
    return :softplus
end

ops[:softmax] = function(a)
    return :relu
end
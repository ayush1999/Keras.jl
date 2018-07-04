ops = Dict{Symbol, Any}()
has_lstm = false

ops[:InputLayer] = function(a)
    return nothing
end

ops[:Input] = function(a)
    return vcall(:.+, a, 0)
end

#data_format = "channels_last"
ops[:Conv] = function(a)
    activation = a.fields["activation"]
    if activation =="linear"
        activation = identity
    end
    if !haskey(weight[a.fields["name"]] ,a.fields["name"])
        dummy_name = a.fields["name"]*"_1"
        weight[a.fields["name"]][a.fields["name"]] = weight[a.fields["name"]][dummy_name]
    end
    w = weight[a.fields["name"]][a.fields["name"]]["kernel:0"]
    kernel_weight = permutedims(w, (3,4,2,1))
    #Flip the kernel
    num_filters = size(kernel_weight)[4]
    num_layers = size(kernel_weight)[3]
    kernel_weight = kernel_weight[end:-1:1, end:-1:1,:,:]
    #kernel_weight = reshape(kernel_weight, (2,2,1,1))
    if !haskey(weight[a.fields["name"]][a.fields["name"]], "bias:0")
        weight[a.fields["name"]][a.fields["name"]]["bias:0"] = [0]
    end
    kernel_bias = weight[a.fields["name"]][a.fields["name"]]["bias:0"]
    strides = (a.fields["strides"]...)
    if a.fields["padding"] == "valid"
        pads = (0,0)
    elseif a.fields["padding"] == "same"
        pads = (Int64.((a.fields["kernel_size"] .-1)./2)...)
    end
    f = (x,) -> begin
        x = permutedims(x, (3,2,4,1))
        temp = Conv(activation, kernel_weight, kernel_bias, strides, pads, (1,1))(x)
        return temp 
    end
    return f
end

ops[:Conv1D] = function(a)
    activation = a.fields["activation"]
    if activation == "linear"
        activation = identity
    elseif activation == "relu"
        activation = relu
    end
    if !haskey(weight[a.fields["name"]] ,a.fields["name"])
        dummy_name = a.fields["name"]*"_1"
        weight[a.fields["name"]][a.fields["name"]] = weight[a.fields["name"]][dummy_name]
    end
    kernel_weight = weight[a.fields["name"]][a.fields["name"]]["kernel:0"]
    kernel_weight = permutedims(kernel_weight, (3,2,1))
    kernel_weight = kernel_weight[end:-1:1, end:-1:1, :]
    s =size(kernel_weight)
    new_size =(s[1], s[2], s[3], 1)
    kernel_weight = reshape(kernel_weight, new_size)
    if !haskey(weight[a.fields["name"]][a.fields["name"]], "bias:0")
        weight[a.fields["name"]][a.fields["name"]]["bias:0"] = [0]
    end
    kernel_bias = weight[a.fields["name"]][a.fields["name"]]["bias:0"]
    strides = (a.fields["strides"]...)[1]
    if a.fields["padding"] == "valid"
        pads = (0,0)
    elseif a.fields["padding"] == "same"
        pads = (Int64.((a.fields["kernel_size"] .-1)./2)...)
    end
    dilation = a.fields["dilation_rate"][1]
    f = (x,) -> begin
        if ndims(x) == 2
            n_shape = (1, size(x)[1], size(x)[2])
            x = reshape(x, n_shape)
        elseif ndims(x) == 3
            x = permutedims(x, (3,1,2))
        end
        x = permutedims(x, (2,3,1))
        s = size(x)
        new_size = (s[1], s[2], s[3], 1)
        x = reshape(x, new_size)
        println(size(x))
        kernel_weight = permutedims(kernel_weight, (1,2,4,3))
        println(size(kernel_weight))
        res = Conv(activation, kernel_weight, kernel_bias, (strides, strides), pads, (dilation, dilation))(x)
        return permutedims(res[:,:,:,1], (2,1,3))
    end
    return f
end

ops[:Activation] = function(a)
    if a.fields["activation"] == "linear"
        return relu
    end
end

ops[:Concatenate] = function(a)
    return :cat
end

ops[:Dropout] = function(a)
    return vcall(:Dropout, a.fields["rate"])
end

ops[:MaxPool] = function(a)
    f = (x,) -> begin
    x = permutedims(x, (3,2,4,1))
    return maxpool(x, (a.fields["pool_size"]...), pad=(0,0), stride=(a.fields["strides"]...))
end
    return f 
end

ops[:MaxPooling1D] = function(a)
    if a.fields["padding"] == "valid"
        pad = (0,0)
    end
    pool_size = a.fields["pool_size"][1]
    stride = a.fields["strides"][1]
     
    f = (x,) -> begin
    if (size(x)[2] - pool_size + stride)/ stride % 1 == 0
        fin_size_middle = Int((size(x)[2] - pool_size + stride)/ stride)
    else
        t = size(x)[2] / pool_size
        fin_size_middle = Int(t - (t%1))
    end
    fin_size = (size(x)[1], fin_size_middle, size(x)[3])
    temp = []
    for i=1:size(x)[3]
        push!(temp, maxpool(reshape(x[1,:,i], (size(x)[2],1,1,1)), (pool_size,1), pad=pad, stride=(stride,1)))
    end
    res = temp[1]
    for i=2:size(x)[3]
        res = hcat(res, temp[i])
    end
    res = reshape(res, fin_size)
    println(size(res))
    return permutedims(res, (2,3,1))
    end
    return f
end

ops[:Flatten] = function(a)
    
    f = (x,) -> begin
    l = prod(size(x))
    if ndims(x) == 4
        return reshape(permutedims(x, (3,1,2,4)), (l,1))
    else
        x = permutedims(x, reverse(range(1, ndims(x))))
        return reshape(x, (l,1))
    end
    end
    return f
end

ops[:BatchNormalization] = function(a)
    epsilon = a.fields["epsilon"]
    momentum = a.fields["momentum"]
    return x -> BatchNorm(size(x)[3], ϵ=epsilon, momentum=momentum)(x)
end

ops[:Dense] = function(a)
    name = a.fields["name"]
    if !haskey(weight[name], name)
        weight[name][name] = weight[name][name*"_1"]
    end
    weight_kernel = weight[name][name]["kernel:0"]
    bias = weight[name][name]["bias:0"]
    if !haskey(a.fields, "activation")
       return Dense(weight_kernel, bias)
    elseif a.fields["activation"] == "linear" || a.fields["activation"] == "relu" 
        a.fields["activation"] = "relu"
        return Dense(weight_kernel, bias), Symbol(a.fields["activation"])
    elseif a.fields["activation"] == "sigmoid"
        return Dense(weight_kernel, bias), x ->(sigmoid.(x))
    elseif a.fields["activation"] == "softmax"
        return Dense(weight_kernel, bias), softmax
    end
end

ops[:ZeroPadding2D] = function(a)
    pads = (a.fields["padding"][1]...)
    return x -> meanpool(x, (1,1), pad=pads, stride=(1,1))
end

ops[:AveragePooling1D] = function(a)
    if a.fields["padding"] == "valid"
        pad = (0,0)
    end
    pool_size = a.fields["pool_size"][1]
    stride = a.fields["strides"][1]
     
    f = (x,) -> begin
    fin_size_middle = size(x)[2] / 2
    fin_size = (size(x)[1], fin_size_middle, size(x)[3])
    temp = []
    for i=1:size(x)[3]
        push!(temp, meanpool(reshape(x[1,:,i], (size(x)[2],1,1,1)), (pool_size,1), pad=pad, stride=(stride,1)))
    end
    res = temp[1]
    for i=2:size(x)[3]
        res = hcat(res, temp[i])
    end
    return reshape(res, (size(x)[1],Int(size(x)[2]/2),size(x)[3]))
    end
    return f
end

ops[:AveragePooling2D] = function(a)
    pool_size = (a.fields["pool_size"]...)
    strides = (a.fields["strides"]...)
    if a.fields["padding"] == "valid"
        pads = (0,0)
    elseif a.fields["padding"] == "same"
        pads = (Int64.((a.fields["pool_size"] .-1)./2)...)
    end
    return x -> meanpool(x, pool_size, pad=pads, stride=strides)
end

ops[:GlobalAveragePooling2D] = function(a)
    return x-> mean(x, (1,2))
end

ops[:Add] = function(a)
    return :+
end

ops[:Reshape] = function(a)
    return (x -> reshape(x, (a.fields["target_shape"]...)))
end

ops[:relu] = function(a)
    return x -> broadcast(relu, x)
end

ops[:tanh] = function(a)
    return tanh
end

ops[:sigmoid] = function(a)
    return sigmoid
end

ops[:elu] = function(a)
    return elu
end

ops[:softplus] = function(a)
    return softplus
end

ops[:softmax] = function(a)
    return relu
end

# Embeddings

ops[:Embedding] = function(a)
    name = a.fields["name"]
    embedding_matrix = weight[name][name]["embeddings:0"]
    output_dim = a.fields["output_dim"]
    input_dim = a.fields["input_dim"]
    f = (x,) -> begin
        if ndims(x) == 1 || ndims(x) == 2
        temp = embedding_matrix[:, Int64(x[1])+1]
        for i=2:length(x)
            temp = hcat(temp, embedding_matrix[:, Int64(x[i])+1])
        end
            return permutedims(temp,reverse(range(1, ndims(temp))))
        end
    end
    return f
end

ops[:LSTM] = function(a)
    name = a.fields["name"]
    lstm_weight = weight[name][name]
    lstm_recurrent_kernel = lstm_weight["recurrent_kernel:0"]
    lstm_bias = lstm_weight["bias:0"]
    lstm_kernel = lstm_weight["kernel:0"]
    vec_size = size(lstm_recurrent_kernel)[2]
    avg_ = 178 # Add general case here
    limit = sqrt(3/avg_)
    kernel_init = linspace(-1*limit, limit, 256)
    
    if a.fields["return_sequences"] == false
        f = (x,)-> begin
            res= 0
            model = LSTM(Flux.LSTMCell(lstm_kernel, lstm_recurrent_kernel, lstm_bias, zeros(vec_size), zeros(vec_size)))
            x = permutedims(x, reverse(range(1, ndims(x))))
            for i=1:size(x)[2]
                res = model(x[:, i])
            end
            return res
        end
    else
        f = (x,)-> begin
            res= 0
            model = LSTM(Flux.LSTMCell(lstm_kernel, lstm_recurrent_kernel, lstm_bias, zeros(vec_size), zeros(vec_size)))
            ans = model(x[1])
            for i=2:length(x)
                res = model(x[i])
                ans = hcat(ans, res)
            end
            return ans[:, end]
        end
    end
    return f
end 
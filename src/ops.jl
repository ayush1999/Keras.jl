ops = Dict{Symbol, Any}()

ops[:InputLayer] = function(a)
    return nothing
end

ops[:Input] = function(a)
    return vcall(:.+, a, 0)
end

ops[:Conv] = function(a)
    activation = a.fields["activation"]
    if activation =="linear"
        activation = "relu"
    end
    if !haskey(weight[a.fields["name"]] ,a.fields["name"])
        dummy_name = a.fields["name"]*"_1"
        weight[a.fields["name"]][a.fields["name"]] = weight[a.fields["name"]][dummy_name]
    end
    kernel_weight = reshape(weight[a.fields["name"]][a.fields["name"]]["kernel:0"],
                             reverse(size(weight[a.fields["name"]][a.fields["name"]]["kernel:0"])))
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
    return vcall(:Conv, Symbol(activation), kernel_weight, kernel_bias, strides, pads)
end

ops[:Concatenate] = function(a)
    return :cat
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
    length_embedding = a.fields["output_dim"]
    #embedding_zero = []
    #for x=1:length_embedding
    #    push!(embedding_zero, 0)
    #end
    f = (x,) -> begin
        temp = embedding_matrix[:, Int64(x[1])+1]
        for i=2:length(x)
            temp = hcat(temp, embedding_matrix[:, Int64(x[i])+1])
        end
        return reshape(temp, reverse(size(temp)))
        #return temp
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
            for i=1:length(x)
                res = model(x[i])
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
            return permutedims(ans, (2,1))
        end
    end
    return f
end 
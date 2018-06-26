using HDF5
using JSON
using Flux

"""
Loads the weights from the weights.h5 file.
"""
function weights(file="weights.h5")
    f = h5open(file, "r")
#    for ele in keys(read(f))
#        if contains(ele ,"/")
#            temp = split(ele, "/")
#            read(f)[temp[1]][temp[1]] = read(f)[temp[1]][temp[2]][temp[1]][temp[2]]
#        end
#    end
    weight = read(f)
    for ele in keys(weight)
        if !isempty(weight[ele])
            if haskey(weight[ele], ele*"_1")
                for ele2 in keys(weight[ele][ele*"_1"])
                    weight[ele][ele*"_1"][ele2] = convert(Array{Float64, N} where N, weight[ele][ele*"_1"][ele2])
                end
            elseif haskey(weight[ele], ele)
                for ele2 in keys(weight[ele][ele])
                    weight[ele][ele][ele2] = convert(Array{Float64, N} where N, weight[ele][ele][ele2])
                end
            elseif ele == "conv1"
                #weight[ele]["conv"][ele]["conv"] = convert(Array{Float64, N} where N, weight[ele]["conv"][ele]["conv"]["kernel:0"])
                weight[ele*"/conv"] = Dict{Any, Any}()
                weight[ele*"/conv"][ele*"/conv"] = Dict{Any, Any}()
                weight[ele*"/conv"][ele*"/conv"]["kernel:0"] = convert(Array{Float64, N} where N, weight[ele]["conv"][ele]["conv"]["kernel:0"])
            end
        end
    end
    return weight
end

"""
Load the structure of the model from the .JSON
file.
"""
function load_structure(file="structure.json")
    res = JSON.parse(String(read(open(file, "r"))))
    return res["config"]
end

"""
Check if model uses sequential/functional API.
"""
function check_modeltype(file)
    res = JSON.parse(String(read(open(file, "r"))))
    return res["class_name"]
end

struct new_type
    layer_type::Symbol
    input_nodes::Any
    fields::Any
end

"""
Get structure information from class_name
"""
function layer_type(a)
    if (a["class_name"] == "Conv2D" || a["class_name"] == "Conv3D")
        return :Conv
    elseif (a["class_name"] == "MaxPooling2D" || a["class_name"] == "MaxPooling3D")
        return :MaxPool
    elseif (a["class_name"] == "Dropout")
        return :Dropout
    elseif (a["class_name"] == "Flatten")
        return :Flatten
    elseif (a["class_name"] == "Dense")
        return :Dense
    elseif (a["class_name"] == "Activation")
        return :Activation
    elseif (a["class_name"] == "Reshape")
        return :Reshape
    elseif (a["class_name"] == "BatchNormalization")
        return :BatchNormalization
    elseif (a["class_name"] == "InputLayer")
        return :InputLayer
    elseif (a["class_name"] == "ZeroPadding2D")
        return :ZeroPadding2D
    elseif (a["class_name"] == "Add")
        return :Add
    elseif (a["class_name"] == "AveragePooling2D")
        return :AveragePooling2D
    elseif (a["class_name"] == "Concatenate")
        return :Concatenate
    elseif (a["class_name"] == "GlobalAveragePooling2D")
        return :GlobalAveragePooling2D
    elseif (a["class_name"] == "Embedding")
        return :Embedding
    elseif (a["class_name"] == "LSTM")
        return :LSTM
    elseif (a["class_name"] == "MaxPooling1D")
        return :MaxPooling1D
    elseif (a["class_name"] == "AveragePooling1D")
        return :AveragePooling1D
    end
end

"""
Extract necessary fields only from the type of layer.
"""
function fields(a)
    if layer_type(a) == :Conv
        return ["name", "strides", "activation", "kernel_size", "padding"]
    elseif layer_type(a) == :MaxPool
        return ["name", "strides", "padding", "pool_size"]
    elseif layer_type(a) == :Dropout
        return ["name", "rate"]
    elseif layer_type(a) == :Flatten
        return ["name"]
    elseif layer_type(a) == :Dense
        return ["name", "activation"]
    elseif layer_type(a) == :relu
        return ["name", "activation"]   
    elseif layer_type(a) == :softmax
        return ["name", "activation"]
    elseif layer_type(a) == :Reshape
        return ["name", "target_shape"]
    elseif layer_type(a) == :BatchNormalization
        return ["name", "momentum", "epsilon"]   
    elseif layer_type(a) == :InputLayer
        return ["name"]   
    elseif layer_type(a) == :ZeroPadding2D
        return ["name", "padding"]   
    elseif layer_type(a) == :Add
        return ["name"]   
    elseif layer_type(a) == :AveragePooling2D
        return ["name","strides","padding","pool_size"]   
    elseif layer_type(a) == :Concatenate
        return ["name","axis"]   
    elseif layer_type(a) == :GlobalAveragePooling2D
        return ["name"]   
    elseif layer_type(a) == :Embedding
        return ["name", "output_dim", "input_dim", "input_length"]   
    elseif layer_type(a) == :LSTM
        return ["name", "activation", "units", "return_sequences"]   
    elseif layer_type(a) == :Activation
        return ["name", "activation"]   
    elseif layer_type(a) == :MaxPooling1D
        return ["name","strides", "pool_size", "padding"]
    elseif layer_type(a) == :AveragePooling1D
        return ["name","strides", "pool_size", "padding"]
    end
end

"""
Extract data layerwise from the structure["layers"]
array.
"""
function load_layers(a::Array{Any, 1})
    res = Array{Any, 1}()
    for ele in a
        d = Dict{Any, Any}()
        for ele2=1:length(fields(ele))
            d[fields(ele)[ele2]] = ele["config"][fields(ele)[ele2]]
        end
        if !haskey(ele, "inbound_nodes")
            ele["inbound_nodes"] = nothing
        end
        inputs = ele["inbound_nodes"]
    push!(res, new_type(layer_type(ele), inputs, d))
    end
    return res
end

using HDF5
using JSON
using Flux

"""
Loads the weights from the weights.h5 file.
"""
function weights(file="weights.h5")
    f = h5open(file, "r")
    return read(f)
end

"""
Load the structure of the model from the .JSON
file.
"""
function load_structure(file="structure.json")
    res = JSON.parse(String(read(open(file, "r"))))
    return res["config"]
end

struct new_type
    layer_type::Symbol
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
        return Symbol(a["config"]["activation"])
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
    end
end

"""
Extract data layerwise from the structure["config"]
array.
"""
function load_layers(a::Array{Any, 1})
    res = Array{Any, 1}()
    for ele in a
        d = Dict{Any, Any}()
        for ele2=1:length(fields(ele))
            d[fields(ele)[ele2]] = ele["config"][fields(ele)[ele2]]
        end
    push!(res, new_type(layer_type(ele), d))
    end
    return res
end

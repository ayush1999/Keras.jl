using DataFlow:Call, syntax, vertex, constant
include("ops.jl")
vcall(a...) = vertex(Call(), constant.(a)...)

#function creat_graph(a::Array{Any, 1})
struct graph
    input
    output
end

function get_basic_graph(a::Array{Any, 1})
    res = Array{graph, 1}()
    for x=1:length(a)
        if x==1
            temp = graph(new_type(:Input, Dict{Any, Any}()), a[x])
        else
            temp = graph(a[x-1], a[x])
        end
        push!(res, temp)
    end
    return res
end

function create_graph(a::Array{Any, 1})
    b = Array{Any, 1}()
    for x=1:length(a)
        push!(b, ops[a[x].layer_type](a[x].fields))
    end
    return b
end

vcall_chain(a...) = vertex(Call(), :Chain, constant.(a)...)

#Create an array of ops from Keras.load_layers
function get_ops(a::Array{Any, 1})
    res = Array{Any, 1}()
    for ele in a
        push!(res, ops[ele.layer_type](ele))
    end
    return res
end

#Chainify get_ops
function chainify(a::Array{Any, 1}, ip)
    res = ip
    for ele in a
        if typeof(ele) <: Tuple
            temp = vcall(:Chain, ele[1], ele[2])
        else
            temp = vcall(:Chain, ele)
        end
        res = vcall(temp, res)
    end
    return res
end

function load(structure_file, weight_file)
    global weight = weights(weight_file)
    if check_modeltype(structure_file) == "Sequential"
        s = load_structure(structure_file)
    elseif check_modeltype(structure_file) == "Model"
        s = load_structure(structure_file)["layers"]
        filter!(x->x["class_name"]!="InputLayer", s)
    end
    l = load_layers(s)
    go = get_ops(l)
    return go, weight
end

function (m::Array{Any, 1})(x)
    return chainify(m, x) |> syntax |> eval
end

#function graphify(m::Array{Any, 1})
#    res = Dict{Any, Any}()
#    for ele in m
#        if ele["class_type"] == "InputLayer"
#            res[ele["name"]] = :ip
#        else
#            res[ele["name"]] = vcall_chain(Keras.ops[Symbol(ele["class_name"])](res[ele["inbound_nodes"][1][1][]]))
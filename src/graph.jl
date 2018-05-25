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
        temp = vcall(:Chain, ele)
        res = vcall(temp, res)
    end
    return res
end

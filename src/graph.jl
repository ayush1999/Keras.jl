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

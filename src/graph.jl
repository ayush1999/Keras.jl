using DataFlow:Call, syntax, vertex, constant
include("ops.jl")
vcall(a...) = vertex(Call, constant.(a)...)

function basic(a::Array{Any, 1})
    for ele in a
        ops[a.layer_type](a.fields)
    end
end 
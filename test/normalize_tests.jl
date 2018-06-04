using Keras
using Flux
using Base.Test
using DataFlow:Call, constant, vertex, syntax

vcall(a...) = vertex(Call(), constant.(a)...)
Keras.new_type(a, b) = Keras.new_type(a, nothing, b)
ip = rand(32,32,3,1)

#BatchNorm Test
d = Dict{Any, Any}()
d["momentum"] = 0.9
d["epsilon"] = 0.1
a = Keras.new_type(:BatchNormalization, d)
@test Keras.ops[:BatchNormalization](a)(ip) == BatchNorm(3, ϵ=0.1, momentum=0.9)(ip)
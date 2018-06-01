using Keras
using Flux
using Base.Test
using DataFlow:Call, constant, vertex, syntax

vcall(a...) = vertex(Call(), constant.(a)...)

a = rand(10,10)

#relu test
temp = vcall(Keras.ops[:relu], a)
@test vcall(temp, a) |> syntax |> eval == relu.(a) 

#tanh test
temp = vcall(Keras.ops[:tanh], a)
@test vcall(temp, a) |> syntax |> eval == tanh.(a)

#sigmoid test
temp = vcall(Keras.ops[:sigmoid], a)
@test vcall(temp, a) |> syntax |> eval == sigmoid(a)

#elu test
temp = vcall(Keras.ops[:elu], a)
@test vcall(:broadcast, temp, a) |> syntax |> eval == elu.(a)

#softplus test
temp = vcall(Keras.ops[:softplus], a)
@test vcall(temp, a) |> syntax |> eval == softplus.(a)

#softmax test
a = rand(10)
temp = vcall(Keras.ops[:softmax], a)
#@test vcall(temp, a) |> syntax |> eval == softmax(a)

#ZeroPadding2D test
a = rand(5,5,1,1)
d = Dict{Any, Any}()
d["padding"] = [[2,2], [2,2]]
b = Keras.new_type(:ZeroPadging2D, d)
@test sum(meanpool(a, (1,1), pad=(2,2), stride=(1,1))) ==
            sum((Keras.ops[:ZeroPadding2D](b))(a))
@test collect(size(meanpool(a, (1,1), pad=(2,2), stride=(1,1)))) - collect(size((Keras.ops[:ZeroPadding2D](b))(a))) ==
            [0,0,0,0]
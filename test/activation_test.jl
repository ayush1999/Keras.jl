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


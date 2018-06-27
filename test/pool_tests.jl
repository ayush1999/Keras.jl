using Keras
using Flux
using Base.Test
using DataFlow:Call, constant, vertex, syntax

Keras.new_type(a, b) = Keras.new_type(a, nothing, b)

vcall(a...) = vertex(Call(), constant.(a)...)

a = rand(10,10,1,1)

# MaxPool test
temp = Keras.new_type(:MaxPool, Dict{Any,Any}(Pair{Any,Any}("name", "max_pooling2d_1"), 
        Pair{Any,Any}("strides", Any[1, 1]),Pair{Any,Any}("padding", "valid"),
            Pair{Any,Any}("pool_size", Any[2, 2])))
@test maxpool(a, (2,2), pad=(0,0), stride=(1,1)) ==
            (Keras.ops[:MaxPool](temp))(a)

# MeanPool test
temp = Keras.new_type(:MaxPool, Dict{Any,Any}(Pair{Any,Any}("name", "mean_pooling2d_1"), 
        Pair{Any,Any}("strides", Any[1, 1]),Pair{Any,Any}("padding", "valid"),
            Pair{Any,Any}("pool_size", Any[2, 2])))
@test meanpool(a, (2,2), pad=(0,0), stride=(1,1)) ==
            (Keras.ops[:AveragePooling2D](temp))(a)

# Concatenate tests
t1, t2 = rand(4,4,4), rand(4,4,4)
@test vcall(Keras.ops[:Concatenate](temp), 3, t1, t2) |> syntax |> eval ==
            cat(3, t1, t2)

# GlobalAveragePooling2D
@test (Keras.ops[:GlobalAveragePooling2D](a))(a) == mean(a, (1,2))

# Conv test
w = rand(3,3,1,32)
b = rand(32)
temp = Keras.new_type(:Conv, Dict{Any,Any}(Pair{Any,Any}("name", "conv2d_1"),
                Pair{Any,Any}("strides", Any[3, 3]),Pair{Any,Any}("kernel_size",
                     Any[3, 3]),Pair{Any,Any}("activation", "relu")))
weight = Dict{Any, Any}()
weight["conv2d_1"] = Dict{Any, Any}()
weight["conv2d_1"]["conv2d_1"] = Dict{Any, Any}()
weight["conv2d_1"]["conv2d_1"]["kernel:0"] = w
weight["conv2d_1"]["conv2d_1"]["bias:0"] = b
@test Conv(relu, w, b, (3,3), (0,0), (1,1))(a) == 
            vcall(Keras.ops[:Conv](temp), a) |> syntax |> eval

# Flatten test

@test vcall(Keras.ops[:Flatten](rand(5,5)), a) |> syntax |> eval == 
            vec(a)

#Reshape test
d = Dict{Any, Any}()
d["target_shape"] = [20,5]
a = Keras.new_type(:Reshape, d)
t = rand(10,10)
@test (Keras.ops[:Reshape](a))(t) == reshape(t, (20,5))
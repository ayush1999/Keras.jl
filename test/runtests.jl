using Flux
using Base.Test
using DataFlow
using Keras

Keras.new_type(a, b) = Keras.new_type(a, nothing, b)
@testset begin

    include("pool_tests.jl")
    include("activation_test.jl")
    include("normalize_tests.jl")
end

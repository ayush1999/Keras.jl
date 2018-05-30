using Flux
using Base.Test
using DataFlow

@testset begin

    include("pool_tests.jl")
    include("activation_test.jl")
    include("normalize_tests.jl")
end

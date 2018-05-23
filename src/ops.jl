ops = Dict{Symbol, Any}()

ops[:Input] = function(a)
    return vcall(:.+, a, 0)
end

ops[:Conv] = function(a)
    name = a["name"]
    weight = weights()[name][name]["kernel:0"]
    bias = weights()[name][name]["bias:0"]
    return vcall(:Conv, weight, bias, Symbol(a["activation"]),  a["strides"])
end

ops[:Dropout] = function(a)
    vcall(:Dropout, a["rate"])
end

ops[:MaxPool] = function(a)
    vcall(:MaxPool, a["rate"])
end

ops[:Dense] = function(a)
    weight = weights()[a.name][a.name]["kernel:0"]
    bias = weights()[a.name][a.name]["bias:0"]
    return vcall(:Dense, weight, bias)
end

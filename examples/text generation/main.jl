using Keras
using JSON

a = JSON.parse(String(read(open("char.json"))))

function int_to_char(d)
    res = Dict{Any, Any}()
    for ele in keys(d)
        res[d[ele]+1] = ele
    end
    return res
end

r = int_to_char(a)

function ip_from_text(text)
    arr = Array{Float64, 1}()
    for ele in text[end-99:end]
        push!(arr ,a[lowercase(string(ele))])
    end
    return reshape(arr, (100,1))
end


model = Keras.load("model-structure.json", "model-weights.h5")

# Generate text upto num_chars
function generate_text(text, num_chars)
    for i=1:num_chars
        text = text*r[findmax(model(ip_from_text(text)))[2]]
    end
    println(text)
end
#Input to the network
text = "The rabbit-hole went straight on like a tunnel for some way, and then
dipped suddenly down, so suddenly that alice had not a moment to think
about"
generate_text(text, 10)


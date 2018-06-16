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
    tex = lowercase(text)
    for ele in text[end-99:end]
        push!(arr ,a[string(ele)])
    end
    return arr./60
end


model = Keras.load("model-structure.json", "model-weights.h5")

# Generate text upto num_chars
function generate_text(text, num_chars)
    println("Original text: ", text, "\n")
    for i=1:num_chars
        text = text*r[findmax(model(ip_from_text(text))[:, end])[2]]
    end
    println("Produced text: ", text)
end
#Input to the network
text = "went straight on like a tunnel for some way, and then
dipped suddenly down, so suddenly that alice had not a mome"
#println(length(text[end-99: end]))

generate_text(text, 100)
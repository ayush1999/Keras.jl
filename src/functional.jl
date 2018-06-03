# This handles loading all models that are nonsequential (Model type)

# load_layers(load_structure(model)["layers"])
# Create a graph from the above array.

function graphify(a::Array{Any, 1})
    global weight = weights("resnet.h5")
    res = Dict{Any, Any}()
    for ele in a
        if ele.layer_type == :InputLayer
            res[ele.fields["name"]] = :ip
            #println(ele.fields["name"], "-->",res[ele.fields["name"]])
        elseif ele.layer_type == :Add
            inputs = ele.input_nodes[1]
            res[ele.fields["name"]] = vcall(ops[:Add](ele), res[inputs[1][1]], res[inputs[2][1]])
        else
            inputs = ele.input_nodes[1][1][1]
            res[ele.fields["name"]] = vcall(ops[ele.layer_type](ele), res[inputs])
            #println(ele.fields["name"], "-->",res[ele.fields["name"]])
        end
    end
    return res
end

function get_outputlayer(structure_file)
    res = JSON.parse(String(read(open(structure_file, "r"))))
    return res["config"]["output_layers"][1][1]
end

function get_op(structure_file, ll)
    dic = graphify(ll)
    op = get_outputlayer(structure_file)
    return dic[op] |> syntax
end

#function load_nonsequential_model(structure_file, weight_file)
#    global weight = weights(weight_file)
#    return 
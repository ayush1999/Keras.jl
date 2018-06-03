# This handles loading all models that are nonsequential (Model type)

# load_layers(load_structure(model)["layers"])
# Create a graph from the above array.
vcall_cat(a...) = vcall(:cat, 3, )

function graphify(a::Array{Any, 1}, structure_file, weight_file, ip)
    global weight = weights(weight_file)
    res = Dict{Any, Any}()
    for ele in a
        if ele.layer_type == :InputLayer
            res[ele.fields["name"]] = vcall(:identity, ip)
            #println(ele.fields["name"], "-->",res[ele.fields["name"]])
        elseif ele.layer_type == :Add
            inputs = ele.input_nodes[1]
            res[ele.fields["name"]] = vcall(ops[:Add](ele), res[inputs[1][1]], res[inputs[2][1]])
        elseif ele.layer_type == :Concatenate
            inputs = ele.input_nodes[1]
            ips = Array{Any, 1}()
            for ip in inputs
                push!(ips, ip[1])
            end
            if length(ips) == 4
                res[ele.fields["name"]] = vcall(:cat, 3, res[ips[1]],res[ips[2]],res[ips[3]],res[ips[4]])  #ToDo: Find an efficient way
            elseif length(ips) == 3                                                                         # for this.
                res[ele.fields["name"]] = vcall(:cat, 3, res[ips[1]],res[ips[2]],res[ips[3]])
            elseif length(ips) == 2
                res[ele.fields["name"]] = vcall(:cat, 3, res[ips[1]],res[ips[2]])
            end
        elseif ele.layer_type == :Dense
            op_dense = ops[:Dense](ele)[1]
            op_activation = ops[:Dense](ele)[2]
            inputs = ele.input_nodes[1][1][1]
            res[ele.fields["name"]] = vcall(op_activation, vcall(op_dense, res[inputs]))
        else
            inputs = ele.input_nodes[1][1][1]
            res[ele.fields["name"]] = vcall(ops[ele.layer_type](ele), res[inputs])
            #println(ele.fields["name"], "-->",res[ele.fields["name"]])
        end
    end
    return res[get_outputlayer(structure_file)] |> syntax |> eval
end

function graphify_dummy(a::Array{Any, 1}, structure_file, weight_file, ip, num)
    global weight = weights(weight_file)
    res = Dict{Any, Any}()
    for ele in a[1:num]
        if ele.layer_type == :InputLayer
            res[ele.fields["name"]] = vcall(:identity, ip)
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
    return res[a[num].fields["name"]] |> syntax |> eval
end

function get_outputlayer(structure_file)
    res = JSON.parse(String(read(open(structure_file, "r"))))
    return res["config"]["output_layers"][1][1]
end

function get_op(structure_file,weight_file,  ll)
    dic = graphify(ll,weight_file)
    op = get_outputlayer(structure_file)
    #op2 = vcall(:Chain, dic[op])
    return dic[op] |> syntax |> eval
end

function load_nonsequential_model(structure_file, weight_file)
    global weight = weights(weight_file)
    ll = load_layers(load_structure(structure_file)["layers"])
    return graphify(ll, structure_file, weight_file)
end
function maxpool1d(ip, kernel, strides=1, pads=0)
    l = size(ip)[2]
    ip1 = ip[:, 1]
    temp = []
    for x=1:length(ip1)-kernel+1
        push!(temp, findmax(ip1[x:x+kernel])[1])
    end
    return temp
end
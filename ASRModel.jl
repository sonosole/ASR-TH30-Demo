using Delta

import Delta.forward
import Delta.predict
import Delta.paramsof
import Delta.nparamsof


# [1]. define model struct
mutable struct Model
    blocks::Vector
    function Model(featdims::Int, nclass::Int)
        c1 = conv1d(featdims,256,8,stride=3)
        c2 = conv1d(256,256,3,stride=2)
        c3 = conv1d(256,512,3,stride=1)
        c4 = conv1d(512,512,3,stride=1)
        c5 = conv1d(512,512,3,stride=1)
        r1 = indrnn(512,512)
        r2 = indrnn(512,512)
        f1 = linear(512, nclass)
        new([c1,c2,c3,c4,c5, Chain(r1,r2,f1)])
    end
end


# [2]. model's length, indexing and iterate method
Base.length(m::Model) = length(m.blocks)
Base.getindex(m::Model, k...) = m.blocks[k...]
Base.lastindex(m::Model)  = length(m)
Base.firstindex(m::Model) = 1
Base.iterate(m::Model, i=firstindex(m)) = i>length(m) ? nothing : (m[i], i+1)

# calculate how many params model uses
function nparamsof(model::Model)
    nparams = 0
    for m in model
        nparams += nparamsof(m)
    end
    return nparams
end

# calculate how many bytes model uses
function bytesof(model::Model, unit::String="MB")
    n = nparamsof(model)
    u = lowercase(unit)
    if u == "kb" return n * sizeof(eltype(model[1].w)) / 1024 end
    if u == "mb" return n * sizeof(eltype(model[1].w)) / 1048576 end
    if u == "gb" return n * sizeof(eltype(model[1].w)) / 1073741824 end
    if u == "tb" return n * sizeof(eltype(model[1].w)) / 1099511627776 end
end


# [3]. define how to extrac model's params
function paramsof(m::Model)
    params = Vector{Variable}(undef,0)
    for i = 1:length(m)
        append!(params, paramsof(m[i]))
    end
    return params
end


# [4]. define model's forward calculation
function forward(m::Model, input::Variable)
    x1 = forward(m[1], input)
    x1 = relu(x1)
    x2 = forward(m[2], x1)
    x2 = relu(x2)
    x3 = forward(m[3], x2)
    x3 = relu(x3)
    x4 = forward(m[4], x3)
    x4 = relu(x4)
    x5 = forward(m[5], x4)
    x5 = relu(x5)
    return PackedSeqForward(m[6], x5)
end


# [5]. define predict calculation
function predict(m::Model, input::AbstractArray)
    x1 = predict(m[1], input)
    x1 = relu(x1)
    x2 = predict(m[2], x1)
    x2 = relu(x2)
    x3 = predict(m[3], x2)
    x3 = relu(x3)
    x4 = predict(m[4], x3)
    x4 = relu(x4)
    x5 = predict(m[5], x4)
    x5 = relu(x5)
    return PackedSeqPredict(m[6], x5)
end

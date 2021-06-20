# [1]. define a dataset
struct th30 <: DataSet
    files
    function th30(txtfile)
        files = []
        for line in readlines(txtfile)
            infos = split(line," ")
            file  = infos[1]
            label = infos[2:end]
            push!(files, (file, label))
        end
        new(files)
    end
end


# [2]. dataset's length, indexing and iterate method
Base.length(d::th30) = length(d.files)
Base.firstindex(d::th30) = 1
Base.lastindex(d::th30)  = length(d)
Base.iterate(d::th30, i=firstindex(d)) = i>length(d) ? nothing : (d[i], i+1)

function Base.getindex(d::th30, k::Int)
    file, label = d.files[k]
    return (wavread(file)[1], parse.(Int,label))
end

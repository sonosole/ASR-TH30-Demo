include("./ASRModel.jl")
include("./ASRDataSet.jl")

using WAV:wavread
using SpeechAugment
using Fbank

# [1]. data augmentation
fs     = 16000;
echo   = initAddEcho(fs, (0.05,0.4), (3.0,3.2,2.5,3.5,2.0,3.0));
cliper = initClipWav((0.5,2.0));
drop   = initDropWav(fs, (0.09,0.15));
far    = initFarfieldWav(fs, (0.5,0.9));
speed  = initSpeedWav((0.8,1.2));
fnlist = [echo,cliper,drop,far,speed];


# [2]. feat extraction feat
EPS1  = 1e-6
EPS2  = log(EPS1)
fbp   = fbankparams(epsilon=EPS1)
fbank = initfbank(fbp)


# [3]. collate function
function collateth30(samples::Vector{Tuple{Array{Float64,2},Vector{Int64}}})
    datas = []
    labels = []
    for sample in samples
        data, label = sample
        push!(datas, data)
        push!(labels, label)
    end
    datas = augmentWavs(fnlist, datas)
    for i = 1:length(datas)
        datas[i] = fbank(datas[i])
    end
    return (PadSeqPackBatch(datas, epsilon=EPS2), labels)
end


FPATH = "./doc/records.txt"
BATCHSIZE = 2
Th30DataSet = th30(FPATH)
TH30DataLoader = DataLoader(Th30DataSet, batchsize=BATCHSIZE, collatefn=collateth30)
DevTH30DataLoader = DataLoader(Th30DataSet, batchsize=1, collatefn=collateth30)

featdims  = 32
nclass    = 1209   # 1 blank + 1208 toned pinyin
asrmodel  = Model(featdims, nclass);
asrparams = paramsof(asrmodel);
optimizer = Adam(asrparams; learnRate=1e-5);

epochs  = 5
iters   = length(TH30DataLoader)
LOGLOSS = zeros(iters, epochs)

for e = 1:epochs
    # [1]. training
    tic = time()
    for (b, data) in enumerate(TH30DataLoader)
        feats, labels = data
        y = forward(asrmodel, Variable(feats))
        LOGLOSS[b,e] = CRNN_Batch_CTC_With_Softmax(y, labels);
        loglikely = trunc(LOGLOSS[b,e], digits=5)
        backward()
        update(optimizer, asrparams, clipvalue=99.0)
        zerograds(asrparams)
        println("batch:$b/$iters, epoch:$e/$epochs, loss=$loglikely");
    end
    toc = time()
    Δt  = trunc(toc-tic, digits=3)

    # [2]. evaluating wer
    dist = 0
    DIST = 0
    for data in DevTH30DataLoader
        feats, labels = data
        F, T = size(feats)
        y = predict(asrmodel, reshape(feats,F,T,1))
        z = softmax(y, dims=1)
        r = CTCGreedySearch(z)
        dist += edistance(labels[1], r)
        DIST += length(labels[1])
    end
    WER = trunc(dist/DIST * 100.0, digits=3)
    tmp = trunc(sum(LOGLOSS[:,e]), digits=3)
    println(yellow("epoch:$e/$epochs, loss=$tmp, WER=$WER%, time=$Δt s"));
end

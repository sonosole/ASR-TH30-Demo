using Statistics
using Plots

# [1]. weights' pdf
x = asrmodel[1].w.value;
m = trunc(Statistics.mean(x),digits=2);
s = trunc(Statistics.std(x),digits=2);
p1=histogram(x[:], bins=floor(Int,0.02*length(x)), legend=:none, ticks=nothing, framestyle=:box);
title!("size=$(size(x)), (mean=$m, std=$s)",titlefont=8,normalize=:pdf);
xlims!(-0.25,0.25);

x = asrmodel[2].w.value;
m = trunc(Statistics.mean(x),digits=2);
s = trunc(Statistics.std(x),digits=2);
p2=histogram(x[:], bins=floor(Int,0.02*length(x)), legend=:none, ticks=nothing, framestyle=:box);
title!("size=$(size(x)), (mean=$m, std=$s)",titlefont=8,normalize=:pdf);
xlims!(-0.15,0.15);

x = asrmodel[3].w.value;
m = trunc(Statistics.mean(x),digits=2);
s = trunc(Statistics.std(x),digits=2);
p3=histogram(x[:], bins=floor(Int,0.02*length(x)), legend=:none, ticks=nothing, framestyle=:box);
title!("size=$(size(x)), (mean=$m, std=$s)",titlefont=8,normalize=:pdf);
xlims!(-0.15,0.15);

x = asrmodel[4].w.value;
m = trunc(Statistics.mean(x),digits=2);
s = trunc(Statistics.std(x),digits=2);
p4=histogram(x[:], bins=floor(Int,0.02*length(x)), legend=:none, ticks=nothing, framestyle=:box);
title!("size=$(size(x)), (mean=$m, std=$s)",titlefont=8,normalize=:pdf);
xlims!(-0.15,0.15);

x = asrmodel[5].w.value;
m = trunc(Statistics.mean(x),digits=2);
s = trunc(Statistics.std(x),digits=2);
p5=histogram(x[:], bins=floor(Int,0.02*length(x)), legend=:none, ticks=nothing, framestyle=:box);
title!("size=$(size(x)), (mean=$m, std=$s)",titlefont=8,normalize=:pdf);
xlims!(-0.15,0.15);

x = asrmodel[6][1].w.value;
m = trunc(Statistics.mean(x),digits=2);
s = trunc(Statistics.std(x),digits=2);
p6=histogram(x[:], bins=floor(Int,0.02*length(x)), legend=:none, ticks=nothing, framestyle=:box);
title!("size=$(size(x)), (mean=$m, std=$s)",titlefont=8,normalize=:pdf);
xlims!(-0.2,0.2);

x = asrmodel[6][2].w.value;
m = trunc(Statistics.mean(x),digits=2);
s = trunc(Statistics.std(x),digits=2);
p7=histogram(x[:], bins=floor(Int,0.02*length(x)), legend=:none, ticks=nothing, framestyle=:box);
title!("size=$(size(x)), (mean=$m, std=$s)",titlefont=8,normalize=:pdf);
xlims!(-0.2,0.2);

x = asrmodel[6][3].w.value;
m = trunc(Statistics.mean(x),digits=2);
s = trunc(Statistics.std(x),digits=2);
p8=histogram(x[:], bins=floor(Int,0.02*length(x)), legend=:none, ticks=nothing, framestyle=:box);
title!("size=$(size(x)), (mean=$m, std=$s)",titlefont=8,normalize=:pdf);
xlims!(-0.2,0.2);

layout = @layout([[a;b;c;d] [e;f;g;h]])
plt1 = plot(p1,p2,p3,p4,p5,p6,p7,p8, layout=layout, size=(700,900));


# [2]. spectrum and blank state's outputs
WAVF = "./data/A4_74.wav"
WAV  = wavread(WAVF)[1]
feat = fbank(WAV)
F, T = size(feat)
y = predict(asrmodel, reshape(feat,F,T,1))
z = softmax(y, dims=1)
r = CTCGreedySearch(z)

p1 = plot(z[1,:], xlims=(0,length(z[1,:])), legend=nothing, framestyle=:box);
p2 = heatmap(feat, legend=nothing);
plt2 = plot(p2,p1, layout=@layout[a;b], size=(700,550));

plot(plt1, plt2, layout=grid(2, 1, heights=[0.65, 0.35]))

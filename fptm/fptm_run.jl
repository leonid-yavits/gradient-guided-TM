# Driver for the unmodified FPTM reference implementation (BooBSD/FuzzyPatternTM, commit e25a51e).
# Same Booleanized arrays, class-balanced sampling and metrics as tmexp.py. No model selection on test.
# Usage: julia -t 1 fptm_run.jl <dataset> <clauses_per_class> <T> <S> <L> <LF> <epochs> <P> <seed> <out.json>
# Path to FuzzyPatternTM.jl from https://github.com/BooBSD/FuzzyPatternTM at commit e25a51e (set FPTM_SRC).
include(get(ENV, "FPTM_SRC", joinpath(@__DIR__, "FuzzyPatternTM", "src", "FuzzyPatternTM.jl")))
using .FuzzyPatternTM: TMInput, TMClassifier, train!, predict, initialize!
using Random, Printf

ds = ARGS[1]
cpc, T, S, L, LF, epochs, P, seed = parse.(Int, ARGS[2:9])
out = ARGS[10]
Random.seed!(seed)
const BIN = get(ENV, "GGTH_BIN", joinpath(@__DIR__, "..", "data", "bin"))   # written by fptm/export_bin.py
info = read("$BIN/$ds.json", String)
shape(s) = parse.(Int, match(Regex("\"$s\": \\[(\\d+), (\\d+)\\]"), info).captures)
m = parse(Int, match(r"\"m\": (\d+)", info).captures[1])

function loadsplit(s)
    n, o = shape(s)
    B = Vector{UInt8}(undef, n * o); read!("$BIN/$(ds)_B$s.u8", B)
    y = Vector{Int64}(undef, n); read!("$BIN/$(ds)_y$s.i64", y)
    Bm = reshape(B, o, n)                       # numpy row-major (n, o) = Julia column-major (o, n)
    X = [TMInput(Bm[:, i] .== 0x01) for i in 1:n]
    return X, y
end
Xtr, ytr = loadsplit("tr"); Xva, yva = loadsplit("va"); Xte, yte = loadsplit("te")

# P samples per class per epoch, with replacement only when the class is smaller than P (as in tmexp.py)
function balanced(y, P, m)
    idx = Int[]
    for k in 0:m-1
        ik = findall(==(k), y)
        isempty(ik) && continue
        append!(idx, length(ik) < P ? rand(ik, P) : ik[randperm(length(ik))[1:P]])
    end
    return shuffle(idx)
end

# Accuracy, balanced accuracy (macro recall) and macro F1 over classes present in y (as in tmexp.py)
function metrics(y, yp)
    rec = Float64[]; f1 = Float64[]
    for k in sort(unique(y))
        tp = sum((yp .== k) .& (y .== k)); fn = sum((yp .!= k) .& (y .== k)); fp = sum((yp .== k) .& (y .!= k))
        r = tp / (tp + fn); p = tp + fp > 0 ? tp / (tp + fp) : 0.0
        push!(rec, r); push!(f1, p + r > 0 ? 2p * r / (p + r) : 0.0)
    end
    return sum(yp .== y) / length(y), sum(rec) / length(rec), sum(f1) / length(f1)
end

function mean_literals(tm)
    n = 0; tot = 0
    for ta in values(tm.clauses)
        for (a, b) in zip(ta.positive_included_literals, ta.positive_included_literals_inverted)
            tot += length(a) + length(b); n += 1
        end
        for (a, b) in zip(ta.negative_included_literals, ta.negative_included_literals_inverted)
            tot += length(a) + length(b); n += 1
        end
    end
    return tot / n
end

tm = TMClassifier{Int64}(cpc, T, S, L=L, LF=LF, states_num=256, include_limit=200)
initialize!(tm, Xtr, ytr)
recs = String[]
for ep in 1:epochs
    idx = P > 0 ? balanced(ytr, P, m) : randperm(length(ytr))
    t = @elapsed train!(tm, Xtr[idx], ytr[idx], shuffle=true)
    r = @sprintf("{\"epoch\": %d, \"sec\": %.2f, \"mean_literals\": %.2f", ep, t, mean_literals(tm))
    for (s, X, y) in (("val", Xva, yva), ("test", Xte, yte))
        a, ba, f = metrics(y, predict(tm, X))
        r *= @sprintf(", \"%s_acc\": %.6f, \"%s_bacc\": %.6f, \"%s_f1\": %.6f", s, a, s, ba, s, f)
    end
    r *= "}"
    println(r); flush(stdout)
    push!(recs, r)
    open(out, "w") do f
        write(f, "{\"args\": {\"dataset\": \"$ds\", \"clauses_per_class\": $cpc, \"T\": $T, \"S\": $S, \"L\": $L, \"LF\": $LF, \"P\": $P, \"seed\": $seed}, \"epochs\": [" * join(recs, ", ") * "]}")
    end
end

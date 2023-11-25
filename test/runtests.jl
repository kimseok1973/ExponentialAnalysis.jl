using ExponentialAnalysis
using Test

using Distributions,CSV,DataFrames

@testset "ExponentialAnalysis.jl" begin
    # make a sample data
    ws = [0.0002, 0.9998]
    ys = [2.1, 781829.0]
    mdist = MixtureModel([Exponential(y) for y in ys], ws)
    ms = [43_000_000 , 20_000_000, 10_000_000];
    s_size = 50
    grp = zeros(Float64,s_size,3) ; grp[20:30,:] .= 100
    ps = (; ys = [757258.8, 2.10], w = [0.9997, 0.0003], r = [0.4225, 0.01284], s = 610)
    dl = predict_effect(grp, ms, ps, s_size)
    @show dl

    test_dl = sum(dl, dims=3)[:,1,1]
    # reverse-finding parameters
    model=expmodel(test_dl, grp, ms)
    test_ps,chain = analysis(model ; sample_size=200)
    rs = describe_effect(test_dl, grp, ms, ps)

    @show chain
    @show ps

    rs_eff, lb = describe_effect(test_dl, grp, ms, test_ps)

    sum(rs_eff, dims=3) |> t-> reshape(t,(:, 3)) |> t -> DataFrame([ e => t[:,i] for (i,e) in enumerate(lb)]...)

end

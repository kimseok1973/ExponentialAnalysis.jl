module ExponentialAnalysis

using Distributions,Turing, DynamicPPL

export expmodel,analysis,describe_effect, inference_recognition, predict_effect

first_marker(grp)=[i for i = 1:length(grp) if grp[i] > 0] |> t-> minimum(t)
exp_decomposed(us::Array, term::Real, i) =  [us[k] * (cdf(Exponential(term), i-k+1) - cdf(Exponential(term), i-k)) for k = 1:i] |> sum
exp_decomposed(us::Array, dist, i) =  [us[k] * (cdf(dist, i-k+1) - cdf(dist, i-k)) for k = 1:i] |> sum
exp_decomposed(u::Real, term) =  u * cdf(Exponential(term), 1)

mexp_decomposed(m, us, dist, i) = begin
    # non GRP effect
    u1 = m * cdf(dist, 1)
    # GRP effect
    u2 = [us[k] * (cdf(dist, i-k+1) - cdf(dist, i-k)) for k = 1:i] |> sum
    return u1 + u2, u1, u2
end

# when data would been included the missing data, function can find the sum of data
# missing data is inference medium of datum.

#sum_missing(xs) = [e for e in xs if e !== missing] |> sum
sum_missing(xs::Vector{Float64}) = sum(xs)
sum_missing(xs::Vector{Int64}) = sum(xs)
sum_missing(xs) = begin
    rs = [e for e in xs if e !== missing] |> mean
    rs * length(rs)
end

"""
    expmodel(dl::Vector, grp::Vector or Matrix , m::Real or Vector)

return the model for anlaysis function.
dl is the number of customers' action, e.g. download of apps or purchase of materials.
grp is the number of media action, e.g. GRP of TV. Finally, it will be exchange to the number of people that exposed by media.
m is the population parameters that is muliplied by grp (media action).

examples

m = expmodel(dl::Vector, grp::Vector, m::Real)

m = expmodel(dl::Vector, grp::Matrix, m::Vector)

Option

m = expmodel(dl, grp, m ; K=2, sum_subtract=true)

sum_subtract : deafult is false. when it is true, mother is subtracted sum(dl) t=1:n.
this means that dl action is one-time action or repetition action.
When it is one time action, option must be false. Otherwise, repetition action must be ture.

"""
expmodel(dl, grp, m) = expdecompose(dl, grp, m)
expmodel(dl::Vector, grp::Vector, m) = expdecompose(dl::Vector, grp::Vector, m)

@model function expdecompose(dl::Vector, grp::Vector , m = 120_000_000 ; 
                    n = length(grp), J=first_marker(grp), K=2, sum_subtract=false)
    s ~ Exponential(1)
    #r ~ filldist(truncated(TDist(3), 0.000000001, 0.9999),2)
    r ~ filldist(InverseGamma(2,3), 2)
    w  ~ Dirichlet(ones(K)/K)
    ys ~ filldist(truncated(Cauchy(0,2),0.00001,Inf), K)
    
    for i = 1:n
        if dl[i] !== missing
            mdist = MixtureModel([Exponential(y) for y in ys], w)
            sum_subtract == true ? mz = m - sum_missing(dl[1:i]) : mz = m
            ag1= 1/r[1] * mz
            ag2= 1/r[2] .* grp .* mz 
            u, _, _ = mexp_decomposed(ag1 ,ag2 , mdist, i)
            dl[i] ~ Normal(u, s)
        end
    end
end

@model function expdecompose(dl::Vector, grp::Matrix, ms::Vector ; K=2)
    n,m =size(grp)
    s ~ Exponential(1)
    r ~ filldist(InverseGamma(2,3), 2)
    w  ~ Dirichlet(ones(K)/K)
    ys ~ filldist(truncated(Cauchy(0,2),0.00001,Inf), K)
    for i = 1:n
        if dl[i] !== missing
            mdist = MixtureModel([Exponential(y) for y in ys], w)
            us = 0.0
            for j = 1:m
                u,_,_ = mexp_decomposed( 1/r[1] * ms[j] , 1/r[2] .* grp[:,j] .* ms[j] , mdist, i)
                us += u
            end    
            dl[i] ~ Normal(us, s)
        end
    end
end

"""
    analysis(m::Model, ; nuts_step=0.65, sample_size=2000)

execute sampling of model by Turing MCMC and return the tuple of parameter's mean.
return parameter are (tuple of parameters, chain data).
use sample

    ps, chain = analysis(m)

"""
function analysis(model ; nuts_step=0.65, sample_size=2000, progress=false)
    chain=sample(model, NUTS(nuts_step),sample_size, progress=progress)
    t = get(chain, [:r, :w, :ys, :s])
    ps = (; ys=[mean(e) for e in t.ys], w=[mean(e) for e in t.w], r=[1/mean(e) for e in t.r], s=mean(t.s) )
    return ps, chain
end

function describe_effect(dl::Vector, grp::Matrix, ms::Vector, ps) 
    n,m = size(grp)
    rs = zeros(length(dl),3,m)
    n = length(dl)
    #mdist = MixtureModel([Exponential(y) for y in ps.ys], ps.w)
    edist1 = Exponential(ps.ys[1])
    edist2 = Exponential(ps.ys[2])
    for i = 1:n
        for j = 1:m
            _, u11, u12 = mexp_decomposed( ps.r[1] * ms[j] , ps.r[2] .* grp[:,j] .* ms[j] , edist1, i)
            _, u21, u22 = mexp_decomposed( ps.r[1] * ms[j] , ps.r[2] .* grp[:,j] .* ms[j] , edist2, i)
          
            recog_base = u11 * ps.w[1] + u21 * ps.w[2] #u11 and u21 is recog-base
            tv_effect_1 = u12 * ps.w[1]
            tv_effect_2 = u22 * ps.w[2]
            rs[i,:,j] = [recog_base, tv_effect_1, tv_effect_2]
        end
    end
    ps.ys[1] > ps.ys[2] ? effect_1 = "long_grp_effect" : effect_1 = "short_grp_effect"
    ps.ys[1] > ps.ys[2] ? effect_2 = "short_grp_effect" : effect_2 = "long_grp_effect"
    label=["recog_base" effect_1 effect_2]
    return rs, label
end

# It will be duplicated
describe_effect(dl::Vector, grp::Vector, ms::Integer, ps) = describe_effect(grp::Vector, ms::Integer, ps) 

function describe_effect(grp::Vector, ms::Integer, ps) 
    n=length(grp)
    m = 1
    rs = zeros(n,3,m)
    #mdist = MixtureModel([Exponential(y) for y in ps.ys], ps.w)
    edist1 = Exponential(ps.ys[1])
    edist2 = Exponential(ps.ys[2])
    for i = 1:n
        for j = 1:m
            _, u11, u12 = mexp_decomposed( ps.r[1] * ms[j] , ps.r[2] .* grp[:,j] .* ms[j] , edist1, i)
            _, u21, u22 = mexp_decomposed( ps.r[1] * ms[j] , ps.r[2] .* grp[:,j] .* ms[j] , edist2, i)
          
            recog_base = u11 * ps.w[1] + u21 * ps.w[2] #u11 and u21 is recog-base
            tv_effect_1 = u12 * ps.w[1]
            tv_effect_2 = u22 * ps.w[2]
            rs[i,:,j] = [recog_base, tv_effect_1, tv_effect_2]
        end
    end
    ps.ys[1] > ps.ys[2] ? effect_1 = "long_grp_effect" : effect_1 = "short_grp_effect"
    ps.ys[1] > ps.ys[2] ? effect_2 = "short_grp_effect" : effect_2 = "long_grp_effect"
    label=["recog_base" effect_1 effect_2]
    return rs, label
end



"""
    predict_effect(grp::Matrix, ms::Vector, ps::parameter_tuple, sample_size::Integer)

execute predict sampling of parameters(ps) and other conditions based on Exponential Modeling. 

"""
function predict_effect(grp::Matrix, ms::Vector, ps, sample_size::Integer)
    n,m=size(grp)
    rs = zeros(Float64, sample_size, 3, m)
    mdist = MixtureModel([Exponential(y) for y in ps.ys], ps.w)

    r1, r2 = ps.r
    for i = 1:n
        for j = 1:m
           u, u1, u2 = mexp_decomposed(r1* ms[j], r2 .* grp[:,j] .* ms[j], mdist, i)
           rs[i, :, j] += [u, u1, u2]
        end
    end
    return rs
end

cdf_r_pop(θ, recog, m) = cdf(Exponential(θ), 1.0) * recog * m

@model function recoginference(dl ; n = length(dl), m = 120_000_000)
    r ~ truncated(Normal(10,10), 0.1, Inf)
    p ~ truncated(Normal(0,10), 0.0001, 0.9999)
    for i = 1:n
        if dl[i] !== missing
            dl[i] ~ NegativeBinomial(r,p)
        end
    end
     
    g ~ Gamma(r, (1-p)/p)
    recog ~ truncated(TDist(3), 0.0000001, 0.999999)
    s ~ Exponential(1)
    for i = 1:n
        if g <= 0 
            DynamicPPL.@addlogprob! -Inf
        else
            if dl[i] !== missing
                dl[i] ~ Normal( cdf_r_pop(g, recog, m) , s)
            end
        end
    end
end

function inference_recognition(xs)
    model = recoginference(xs)
    chain  = sample(model, NUTS(), 2000 ; progress=false) 
    t = get(chain, [:r, :p, :g, :recog, :s]) 
    ps = (; r=mean(t.r), p=mean(t.p), th=(1 - mean(t.p)) /mean(t.p),  recog = mean(t.recog), s=mean(t.s))
    return ps.recog, ps
end


end

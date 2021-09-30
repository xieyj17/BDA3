module TuringToolkit

using MCMCChains
using Random
using Statistics
using Plots
using StatsPlots
using LaTeXStrings
using DataFrames
using KernelDensity
using Interpolations
using QuadGK

function extract_param_posterior(names::Vector{String}, chain::Chains)
    var_name = [Symbol(n) for n in names]
    posterior_samples = chain[var_name]
    n_param = size(posterior_samples)[2]
    posteriors = DataFrame()
    for i in 1:n_param
        tp = posterior_samples[:,i,:]
        tp = mapreduce(c->tp[:,c], vcat, 1:size(tp)[2])
        posteriors[!,String(var_name[i])] = tp
    end
    return posteriors
end

function tempint(p, a, b)
    int, err = quadgk(x->pdf(p, x),a,b)
    return int
end

function find_hdi(data::Vector, level::Float64)
    k = kde(data)
    p = InterpKDE(k)
    
    xs = [i for i in range(minimum(data), stop=quantile(data, 1-level), step = 0.001)] 
    delta = []
    interval = []
    for x in xs
        bs = quantile(data, level)
        for j in range(bs, stop = maximum(data), step = 0.001)
            if tempint(p, x, j) > level
                push!(delta,(j-x))
                push!(interval, [x,j])
                break
            end
        end
    end
    return interval[argmin(delta)]
end

function credible_interval(df::DataFrame; type = "HDI", level::Float64 = 0.94)
    if type == "HDI"
        res = []
        for i in 1:ncol(df)
            td = df[:,i]
            push!(res, find_hdi(td, level))
        end
    elseif type == "ETI"
        res = []
        for i in 1:ncol(df)
            td = df[:i]
            push!(res, [quantile(td, (1-level)/2), quantile(td, 1-(1-level)/2)])
        end
    end
    return res
end

function plot_posterior(chain::Chains, var_names::Vector{String}; 
    kind::String="hist", type::String="HDI", level::Float64=0.94)

    data = extract_param_posterior(var_names, chain)
    cis = credible_interval(data; type, level)
    nofp = length(cis)
    if kind == "hist"
        pnames = []
        for i in 1:nofp
            ci = cis[i]
            tn = Symbol("plot_"*var_names[i])
            push!(pnames, tn)
            pn = histogram(data[:,i], label = var_names[i],fillalpha = 0.5);
            vline!(ci, lw = 3, color=:red, label = false);
            title!("$(round(Int, 100*level))% "*type*" is [$(round(ci[1]; digits=2)), $(round(ci[2]; digits=2))]");
            display(pn)
        end
    elseif kind == "dens"
        for i in 1:nofp
            ci = cis[i]
            tn = Symbol("plot_"*var_names[i])
            push!(pnames, tn)
            pn = density(data[:,i], label = var_names[i],linealpha = 0.7,lw = 3,);
            vline!(ci, lw = 3, color=:red, label = false);
            title!("$(round(Int, 100*level))% "*type*" is [$(round(ci[1]; digits=2)), $(round(ci[2]; digits=2))]");
            display(pn)
        end
    end
end

export extract_param_posterior, credible_interval, plot_posterior

end

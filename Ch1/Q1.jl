using Statistics
using Distributions
using Plots

N = 10000

function problem1(N::Int)
    theta_distribution = Bernoulli()
    thetas = rand(theta_distribution, N).+1
    y1 = Normal(1, 2)
    y2 = Normal(2,2)
    ys = Vector{Float64}(undef, N)
    for i in 1:N
        thetas[i] == 1 ? ys[i] = rand(y1) : ys[i] = rand(y2)
    end

    return ys
end

res = problem1(N)

y = Normal(1.5, 2)
x = collect(-10: 0.01: 10)
histogram(res,normalize=:pdf, label="Simulated distribution")
plot!(x, pdf.(y, x), label="Theoretical distribution")

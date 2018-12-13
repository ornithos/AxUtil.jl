module gmm
using Pkg
using Flux, Flux.Tracker
using Flux: ADAM
using StatsFuns: logsumexp
using Distributions
import Distributions: partype, logpdf
using Formatting: format
using Random # randperm, MersenneTwister
using LinearAlgebra  #: cholesky, logdet, diag, inv, triu
using NNlib: softmax
using AxUtil
using BSON
using Logging
using PyPlot
using ProgressMeter

export GMM, importance_sample, logpdf
# ==== already exists in Distributions.jl
# struct MixComp
#     components::Array{T,1} where T <: MvNormal
#     weights::Array{T, 1} where T <: Real
# end
# Base.convert(::Type{MixComp}, x::MvNormal) = MixComp([x], [1.])
# Base.length(x::MixComp) = length(x.weights)
# Base.size(x::MixComp) = length(x.components[1])
#
# Base.show(io::IO, z::MixComp) = print(io, format("{:d} component GMM in {:d} dimensions. Fields: " *
#                             "`.components` and `.weights`.", length(z), size(z)))
# rand(x::MixComp) = ...

# const GMM = MixtureModel{Multivariate,Continuous,T} where T <: MvNormal

# we're not using the MixtureModel{} form in Distributions.jl, as the structure
# is different to that which is useful for the methods I've defined, and every-
# thing is a little clunky. My sampling routine is 30x faster than theirs on
# some benchmarks.


# =================> Gaussian stuff, should live elsewhere, but don't know
# yet how to import stuff from a different module in the same package.


# ==> Responsibilities take a while... I was toying with a custom procedure for
# this, but the below (commented out) is (slightly) (~25%) slower than
# the current (easy to understand) code in the main body.

# @benchmark softmax(reduce(vcat, map(j -> AxUtil.gmm.log_gauss_llh(S, dGMM.mus[j,:], dGMM.sigmas[:,:,j], bypass=inactive_ixs[j]), 1:15)'))
# BenchmarkTools.Trial:
#   memory estimate:  25.60 MiB
#   allocs estimate:  712
#   --------------
#   minimum time:     14.678 ms (0.00% GC)
#   median time:      18.178 ms (0.00% GC)
#   mean time:        24.905 ms (19.52% GC)
#   maximum time:     473.150 ms (90.19% GC)
#   --------------
#   samples:          201
#   evals/sample:     1

# benchmark responsibilities(S, dGMM.mus, dGMM.sigmas, inactive_ixs)
# BenchmarkTools.Trial:
#   memory estimate:  29.47 MiB
#   allocs estimate:  394889
#   --------------
#   minimum time:     21.312 ms (0.00% GC)
#   median time:      23.391 ms (0.00% GC)
#   mean time:        33.607 ms (24.94% GC)
#   maximum time:     701.121 ms (95.72% GC)
#   --------------
#   samples:          149
#   evals/sample:     1

# function responsibilities(X::Matrix{T}, mus::Matrix{T}, sigmas::Array{T,3}, inactive_ixs) where T <: AbstractFloat
#     d, n = size(X)
#     k = size(mus, 1)
#     out = Matrix{T}(undef, k, n)
#     active_ixs = findall(.!inactive_ixs)
#
#     invLTs = [Matrix(inv(cholesky(sigmas[:,:,j]).L)) for j in 1:k]  # easier to just do inactive_ixs anyway...
#     normconsts = [-2*sum(log.(diag(invLT))) for invLT in invLTs]
#     @threads for i in 1:n
#         @inbounds begin
#             _max = -Inf   # not type stable
#             for j in 1:k
#                 if inactive_ixs[j]
#                     out[j, i] = 0. # not type stable; will ignore later
#                 else
#                     Δ = X[:, i] - mus[j, :]
#                     out[j, i] = -0.5 * (sum(x->x^2, invLTs[j] * Δ) + normconsts[j])
#                     _max = out[j, i] > _max ? out[j, i] : _max
#                 end
#             end
#             _sum = 0. # not type stable
#             for j in active_ixs
#                 out[j, i] = exp(out[j, i] - _max)
#                 _sum += out[j, i]
#             end
#             for j in active_ixs
#                 out[j, i] /= _sum
#             end
#         end # inbounds
#     end # loop over datapoints
#     return out
# end


function log_gauss_llh(X, mu, sigma; bypass=false)
    if !bypass
        out = _log_gauss_llh(X, mu, sigma)
        # try
        #     out = _log_gauss_llh(X, mu, sigma)
        # catch e
        #     out = - ones(size(X, 1))*Inf
        # end
    else
        out = - ones(size(X, 2))*Inf
    end
    return out
end


function _log_gauss_llh(X, mu, sigma)
    d = size(X,1)
#     invUT = Matrix(cholesky(inv(sigma)).U)
    invLT = Matrix(inv(cholesky(sigma).L))
    Z = invLT*(X .- mu)
    exponent = -0.5*sum(Z.^2, dims=1)  |> dropdim1
    lognormconst = -d*log(2*pi)/2 -0.5*logdet(sigma)  #.-0.5*(-2*sum(log.(diag(invLT))))
    return exponent .+ lognormconst
end

function _log_gauss_llh_invLT(X, mu, invLT)
    d = size(X,1)
    Z = invLT*(X .- mu)
    exponent = -0.5*sum(Z.^2, dims=1)  |> dropdim1
    lognormconst = -d*log(2*pi)/2 .-0.5*(-2*sum(log.(diag(invLT))))
    return exponent .+ lognormconst
end
# ==========================================================================


# ===================> Misc stuff, also should live elsewhere when I get imports right
function logsumexpcols(X::AbstractArray{T}) where {T<:Real}
    n = size(X,2)
    out = zeros(n)
    Base.Threads.@threads for i = 1:n
        @views out[i] = logsumexp(X[:,i])
    end
    return out
end

unsqueeze = Flux.unsqueeze
# =============================================================================


dropdim1(x) = dropdims(x, dims=1)   # useful for pipes
dropdim2(x) = dropdims(x, dims=2)

struct GMM{T}
    mus::Array{T,2}      # k * d
    sigmas::Array{T,3}   # d * d * k
    pis::Array{T, 1}     # k
    GMM{T}(mus, sigmas, pis) where T <: Number = begin; x = new(mus, sigmas, pis); isvalidGMM(x); x; end
end

function isvalidGMM(x::GMM)
    @assert (x isa GMM{T} where T <: AbstractFloat) "type must be <: AbstractFloat"
    @assert (size(x.mus, 1) == size(x.sigmas,3) == length(x.pis)) "inconsistent ncomponents"
    @assert (size(x.mus, 2) == size(x.sigmas,1) == size(x.sigmas, 2)) "inconsistent dimension"
end

GMM(mus::Array{T,2}, sigmas::Array{T,3}, pis::Array{T, 1}) where T <: Number = GMM{T}(mus, sigmas, pis)
partype(x::GMM{T}) where T <: AbstractFloat = T

Distributions.ncomponents(x::GMM) = length(x.pis)

Base.size(x::GMM) = size(x.mus, 2)

Base.convert(::Type{GMM}, d::MvNormal) = GMM{partype(d)}(unsqueeze(d.μ, 1), unsqueeze(Matrix(d.Σ), 3), [1.])
Base.convert(::Type{MixtureModel}, d::GMM) = MixtureModel([MvNormal(d.mus[j,:], d.sigmas[:,:,j]) for j in 1:ncomponents(d)], d.pis)
function Base.convert(::Type{GMM}, d::MixtureModel{Multivariate,Continuous,T}) where T <: MvNormal
    return GMM{partype(d)}(reduce(vcat, [d.components[j].μ' for j in 1:ncomponents(d)]),
               cat([Matrix(d.components[j].Σ) for j in 1:ncomponents(d)]..., dims=3), d.prior.p)
end

Base.show(io::IO, z::GMM) = begin; print(io, format("{:d} component GMM in {:d} dimensions. Fields: " *
                            "`.mus`, `.sigmas`, and `.pis`.\n", ncomponents(z), size(z)));
                            show(convert(MixtureModel, z)); end

Base.rand(d::GMM, n::Int; shuffle=true) = sample_from_gmm(n, d.pis, d.mus, d.sigmas, shuffle=shuffle)

function rmcomponents(d::GMM, ixs::Vector{T}) where T <: Signed
    bad_ixs = (sum(Flux.onehotbatch(Tuple(ixs), 1:ncomponents(d)), dims=2) .> 0)  |> dropdim2
    rmcomponents(d, bad_ixs)
end
rmcomponents(d::GMM, ixs::Vector{T}) where T <: Bool = rmcomponents(d, convert(BitArray, ixs))
rmcomponents(d::GMM, ixs::BitArray{1}) = GMM{partype(d)}(d.mus[.!ixs, :], d.sigmas[:, :, .!ixs], d.pis[.!ixs]/sum(d.pis[.!ixs]))
update(d::GMM; mus=nothing, sigmas=nothing, pis=nothing) = GMM(something(mus, d.mus), something(sigmas, d.sigmas), something(pis, d.pis))


function plot_gmm(dGMM::GMM; figsize=(10,10), alpha=0.5, bins=50, fill=false, axs=nothing) where T <: AbstractFloat
    d, k = size(dGMM), ncomponents(dGMM)
    (d > 20) && throw("will not plot for d > 20")

    # d == 2 fits on a single axis: deal with this case first
    if d == 2
        if axs == nothing
            fig, axs = PyPlot.subplots(1, 1, figsize=figsize)
        end
        for j in 1:k
            levcurv = AxUtil.Plot.gaussian_2D_level_curve_pts(dGMM.mus[j,:], dGMM.sigmas[:,:,j])
            axs[plottype](levcurv[:,1], levcurv[:,2], alpha=alpha*dGMM.pis[j]/maximum(dGMM.pis))
        end
        return ax
    end

    # d > 2:
    if axs == nothing
        fig, axs = PyPlot.subplots(d, d, figsize=figsize)
    end
    plottype = fill ? :fill : :plot
    rsmp = rand(dGMM, 5000)

    for ix = 1:d, iy = 1:d
        if ix != iy
            for j in 1:k
                levcurv = AxUtil.Plot.gaussian_2D_level_curve_pts(dGMM.mus[j,:][[ix,iy]], dGMM.sigmas[[ix,iy],[ix,iy],j])
                axs[iy, ix][plottype](levcurv[:,1], levcurv[:,2], alpha=alpha*dGMM.pis[j]/maximum(dGMM.pis))
            end
        else
            axs[iy, ix][:hist](rsmp[ix, :], bins=bins)
        end
    end
    return axs
end


function logpdf(d::GMM, X::Matrix{T}; thrsh_comp=0.005) where T <: AbstractFloat
    return gmm_llh(X, d.pis, d.mus, d.sigmas; thrsh_comp=thrsh_comp)
end

function importance_sample(d::GMM, n::Int, log_f::Function; shuffle=false)
    S = rand(d, n; shuffle=shuffle)
    logW = log_f(S) - logpdf(d, S);
    return S, logW;
end


function is_eff_ss_per_comp(d::GMM, S::Matrix{T}, W::Vector{T}) where T <: AbstractFloat
    k = ncomponents(d)
    out = zeros(T, k)
    rs = softmax(reduce(vcat, map(j -> log_gauss_llh(S, d.mus[j,:], d.sigmas[:,:,j], 1:k)')))
    out = map(j->AxUtil.MCDiagnostic.is_eff_ss(W[rand(Categorical(rs[j,:]), length(W))]), 1:k)
    return out
end


function sample_from_gmm(n, pis, mus, covs; shuffle=true)
    k, p = size(mus)
    Ns = rand(Multinomial(n, pis[:]))
    active_ixs = findall(Ns[:] .>= 1)

    ixs = hcat(vcat(1, 1 .+ cumsum(Ns[1:end-1], dims=1)), cumsum(Ns, dims=1))
    out = zeros(p, n)
    for j=active_ixs
        out[:, ixs[j,1]:ixs[j,2]] = rand(MvNormal(mus[j,:], covs[:,:,j]), Ns[j])
    end
    if shuffle
        out = out[:, randperm(n)]
    end
    return out
end


function gmm_llh(X, pis, mus, sigmas; disp=false, thrsh_comp=0.005)
    p, n = size(X)
    k = length(pis)
    inactive_ixs = pis[:] .< thrsh_comp

    P = zeros(k, n)
    for j = 1:k
        P[j,:] = log_gauss_llh(X, mus[j,:], sigmas[:,:,j],
            bypass=inactive_ixs[j]) .+ log(pis[j])
    end
    if disp
        display(P)
        display(logsumexpcols(P))
    end
    return logsumexpcols(P)
end


responsibilities(X, d::GMM) = softmax(reduce(vcat, [log_gauss_llh(X, d.mus[j,:], d.sigmas[:,:,j]) .+ log(d.pis[j]) for j in 1:ncomponents(d)]'))



function gmm_llh_invLT(X, pis, mus, invLTs::Array{T,1}; disp=false, thrsh_comp=0.005) where T <: AbstractMatrix
    p, n = size(X)
    k = length(pis)
    inactive_ixs = pis[:] .< thrsh_comp

    P = zeros(k, n)
    for j = 1:k
        if !inactive_ixs[j]
            P[j,:] = _log_gauss_llh_invLT(X, mus[j,:], invLTs[j]) .+ log(pis[j])
        else
            P[j,:] .= -Inf
        end
    end
    if disp
        display(P)
        display(logsumexpcols(P))
    end
    return logsumexpcols(P)
end


function gmm_fit(X::Matrix{T}, d::GMM; max_iter=100, tol=1e-3, verbose=true, rm_inactive=false,
                 thrsh_comp=0.005, prior_strength=1.0) where T <: AbstractFloat
    gmm_fit(X, d.pis, d.mus, d.sigmas; max_iter=max_iter, tol=tol, verbose=verbose,
           rm_inactive=rm_inactive, thrsh_comp=thrsh_comp, prior_strength=prior_strength)
end


function gmm_fit(X::Matrix{T}, weights::Vector, d::GMM; max_iter=100, tol=1e-3, verbose=true,
                rm_inactive=false, thrsh_comp=0.005, prior_strength=1.0) where T <: AbstractFloat
    gmm_fit(X, weights, d.pis, d.mus, d.sigmas; max_iter=max_iter, tol=tol, verbose=verbose,
            rm_inactive=rm_inactive, thrsh_comp=thrsh_comp, prior_strength=prior_strength)
end


function gmm_fit(X, pi_prior, mu_prior, cov_prior; max_iter=100, tol=1e-3, verbose=true,
                 rm_inactive=false, thrsh_comp=0.005, prior_strength=1.0)
    gmm_fit(X, trues(size(X,2)), pi_prior, mu_prior, cov_prior; max_iter=max_iter,
            tol=tol, verbose=verbose, rm_inactive=rm_inactive, thrsh_comp=thrsh_comp, prior_strength=prior_strength)
end


function gmm_fit(X, weights, pi_prior, mu_prior, cov_prior; max_iter=100, tol=1e-3, verbose=true,
                 rm_inactive=false, thrsh_comp=0.005, prior_strength=1.0)
    p, n = size(X)
    k = length(pi_prior)
    @assert size(weights) == (n,)
    @assert size(mu_prior) == (k, p)
    @assert size(cov_prior) == (p, p, k)
    pis = pi_prior/sum(pi_prior)
    mus = copy(mu_prior)
    sigmas = copy(cov_prior)

    weights = weights / mean(weights)

    inactive_ixs = pi_prior[:] .< thrsh_comp
    pi_prior = copy(pi_prior) * prior_strength

    for i in range(1, stop=max_iter)
        # E-step
        rs = reduce(vcat, map(j -> log_gauss_llh(X, mus[j,:], sigmas[:,:,j], bypass=inactive_ixs[j]), 1:k)')
        try
            rs .+= log.(pis)[:] + log.(pi_prior)[:]
            catch e
            @warn "(gmm) rs and (log) pis are not conformable. The respective values are:"
            display(rs)
            display(log.(pis))
            rethrow(e)
        end
        @debug format("(gmm) ({:3d}/{:3d})", i, max_iter) llh=round(sum(log.(sum(pis .* (exp.(rs) .* weights'), dims=1))), digits=2)
        rs = softmax(rs)

        @debug format("(gmm) ({:3d}/{:3d}) original responsibilities", i, max_iter) rs=vec(sum(rs, dims=2))
        # reweight according to importance weights (see Adaptive IS in General Mix. Cappé et al. 2008)
        rs .*= weights'
        @debug format("(gmm) ({:3d}/{:3d}), wgtd responsibilities", i, max_iter) rs=vec(sum(rs, dims=2))

        # M-step
        Ns = vec(sum(rs, dims=2))
        inactive_ixs = Ns .< thrsh_comp
        active_ixs = .! inactive_ixs
        if any(inactive_ixs)
            pis[inactive_ixs] .= 0.0
            pi_prior[inactive_ixs] .= 0.0
        end
        pis = Ns + pi_prior[:]
        pis /= sum(pis)
        @debug format("(gmm) ({:3d}/{:3d})", i, max_iter) thrsh_comp=thrsh_comp n_inactive=sum(inactive_ixs) pis=pis
        # ==========>  .... SORT OUT X IS NOW p * n
        _mus = reduce(hcat, map(j -> sum(X .* rs[j:j,:], dims=2) .+ pi_prior[j]*mu_prior[j,:], findall(active_ixs)))'
        _mus ./= vec(Ns[active_ixs] + pi_prior[active_ixs])
        mus[active_ixs,:] = _mus

        @views for j in findall(active_ixs)
            Δx = X .- mus[j, :]
            Δμ = (mus[j,:] - mu_prior[j,:])'
            sigmas[:,:,j] = (Δx.*rs[j:j,:])*Δx' + pi_prior[j]*(Δμ'Δμ + cov_prior[:,:,j])
            sigmas[:,:,j] ./= (Ns[j] + pi_prior[j] + p + 2)     # normalizing terms from Wishart prior
            sigmas[:,:,j] = (sigmas[:,:,j] + sigmas[:,:,j]')/2 + AxUtil.Arr.eye(p)*1e-6   # hack: prevent collapse
        end
        # bson(format("dbg/gmm.bson"), pis=pis, sigmas=sigmas, mus=mus, X=X, weights=weights, active_ixs=active_ixs)

    end

    out = GMM{typeof(X[1])}(mus, sigmas, pis)
    if rm_inactive
        out = rmcomponents(out, inactive_ixs)
    end

    return out
end



# llh_unnorm(Scentered, Linv) = let white = Linv * Scentered; -0.5*sum(white .* white, dims=1); end


# A little like a Gaussian sum approximation for nonlinear dynamical system.
# (oh.. but.. it is an *inner* variational approximation, not an outer one.
# In our case this is not *so* terrible since we do not assume the previous GMM
# is the true posterior as in GS approx., so we do not get catastrophic shrinkage.
# But we should be a little worried.)
@inline function build_mat(x_lt, x_diag, d::Int)
    AxUtil.Flux.make_lt_strict(x_lt, d) + AxUtil.Flux.diag0(exp.(x_diag))
end
@inline _llh_unnorm(Scentered, Linv) = -0.5*sum(x->x*x, Linv * Scentered, dims=1)

function optimise_components_bbb(d::GMM, log_f::Function, epochs::Int, batch_size_per_cls::Int;
            converge_thrsh::AbstractFloat=0.999, lr::AbstractFloat=1e-3, auto_lr::Bool=true,
            fix_mean::Bool=false, fix_cov::Bool=false,
            log_f_prev::Union{Function, Nothing}=nothing, anneal_sched::AbstractArray=[1.])
    success = 0
    nanfail = 0   # permit up to 3 in a row failures due to NaN (as this can be from blow-up.)
    haszygote = haskey(Pkg.installed(), "Zygote")
    @assert all( 0. .<= anneal_sched .<= 1.) "annealing schedule array must be ∈ [0,1]^N"
    @assert !xor(log_f_prev==nothing, all(anneal_sched .== 1.)) "both log_f_prev and non-trivial annealing schedule must be specified."

    local history  # local scope of history for persistence outside loop.
    while success < 1
        @debug format("(bbb) LEARNING RATE: {:.3e}", lr)
        # if haszygote
        #     d, history, success = _optimise_components_bbb_zygote(d, log_f, epochs, batch_size_per_cls; converge_thrsh=converge_thrsh, lr=lr, exitifnan=(nanfail<3), auto_lr=auto_lr)
        # else
        d, history, success = _optimise_components_bbb(d, log_f, epochs, batch_size_per_cls;
                                                       converge_thrsh=converge_thrsh,
                                                       lr=lr, exitifnan=(nanfail<3),
                                                       auto_lr=auto_lr,
                                                       fix_mean=fix_mean, fix_cov=fix_cov,
                                                       log_f_prev=log_f_prev, anneal_sched=anneal_sched)
        # end
        lr *= 0.5
        nanfail = (success < 0) * (nanfail - success)   # increment if success = -1, o.w. reset
    end
    return d, history
end

function _failure_dump(ee, dGMM_orig, mupars, invLTpars, invDiagPars)
    display(format("ee = {:d}", ee))
    display("mus originally:")
    display(dGMM_orig.mus)
    display("sigmas originally:")
    display([dGMM_orig.sigmas[:,:,j] for j in 1:size(dGMM_orig.sigmas, 3)])
    display("mu pars:")
    display(mupars.data)
    display("mu grad:")
    display(mupars.grad)
    display("LT pars:")
    display([x.data for x in invLTpars])
    display("LT grad:")
    display([x.grad for x in invLTpars])
    display("Diag pars:")
    display([x.data for x in invDiagPars])
    display("Diag grad:")
    display([x.grad for x in invDiagPars])
end

function _optimise_components_bbb(d::GMM, log_f::Function, epochs::Int, batch_size_per_cls::Int;
        converge_thrsh::AbstractFloat=0.999, lr::AbstractFloat=1e-3, exitifnan::Bool=false,
        auto_lr::Bool=true, fix_mean::Bool=false, fix_cov::Bool=false,
        log_f_prev::Union{Function, Nothing}=nothing, anneal_sched::AbstractArray=[1.])
    # @debug "(bbb) Input GMM: " dGMM=d
    # (ncomponents(d) > 8) && @debug "(bbb) Input GMM: " dGMM=rmcomponents(d, collect(1:8))
    # (ncomponents(d) > 16) && @debug "(bbb) Input GMM: " dGMM=rmcomponents(d, collect(1:16))
    n_d = size(d)
    k = ncomponents(d)
    # if Logging.min_enabled_level(current_logger()) ≤ LogLevel(-2000)
    #     for j = 1:k
    #         @logmsg LogLevel(-500) format("sigma {:d} = ", j) d.sigmas[:,:,j]
    #     end
    # end

    # Set up pars and gradient containers
    invLTpars = [Matrix(inv(cholesky(d.sigmas[:,:,j]).L)) for j in 1:k]  # note that ∵ inverse, Σ^{-1} = L'L
    invDiagPars = [Flux.param(log.(x[diagind(x)])) for x in invLTpars]
    invLTpars = [Flux.param(x[tril!(trues(n_d, n_d), -1)]) for x in invLTpars]

    mupars = Flux.param(d.mus)
    ∇mu = Matrix{partype(d)}(undef, k, n_d)
    ∇Ls = [Matrix{partype(d)}(undef, n_d, n_d) for i in 1:k]
    opt = ADAM(Tracker.Params((mupars, invLTpars..., invDiagPars...)), lr)

    # Admin
    hist_freq = 5   # Sampling freq for history of objective
    s_wdw = 50      # Smoothing window for history for convergence (i.e. s_wdw * hist_freq)
    history = zeros(Int(floor(epochs/hist_freq)))
    s_hist = zeros(Int(floor(epochs/hist_freq)))
    n_anneal = something(findlast(anneal_sched .< 1.), 0)

    rng = MersenneTwister()   # to ensure common random variates for reconstruction and entropy grads.
    @showprogress 1 for ee in 1:epochs
        invLT = [build_mat(x_lt, x_dia, n_d) for (x_lt, x_dia) in zip(invLTpars, invDiagPars)]
        objective = 0.

        η = (ee <= n_anneal) ? anneal_sched[ee] : 1.0  # [1 - annealing amount]

        # Take sample from each component and backprop through recon. term of KL
        # ===> FLUX / AD PART OF PROCEDURE ===================
        for j in 1:k
            ee_seed = rand(rng, 1:2^32 - 1)
            Random.seed!(ee_seed)
            ϵ = randn(n_d, batch_size_per_cls)
            x = mupars[j,:] .+ inv(invLT[j])*ϵ
            objective_j = - η * sum(log_f(x))/batch_size_per_cls  # reconstruction

            Tracker.back!(objective_j)  # accumulate gradients
            objective += objective_j.data  # for objective value
            # ====================================================

            # ---- Annealing (default: none) ----------------
            if η < 1.0
                x = mupars[j,:] .+ inv(invLT[j])*ϵ
                objective_j = - (1-η) * sum(log_f_prev(x))/batch_size_per_cls  # reconstruction

                Tracker.back!(objective_j)  # accumulate gradients
                objective = objective + objective_j.data  # for objective value
            end
            # -----------------------------------------------

            # ===> Calculate entropy of GMM (and gradient thereof)
            Random.seed!(ee_seed)   # common r.v.s (variance reduction)
            _obj = _gmm_entropy_and_grad!(mupars.data, [x.data for x in invLT], ∇mu, ∇Ls; M=batch_size_per_cls, ixs=[j])
            objective += _obj

            mupars.grad .+= ∇mu
            fix_mean && (mupars.grad .= 0.)

            if any(isnan.(mupars.grad))
                display("FAILURE IN MU GRAD")
                _failure_dump(ee, d, mupars, invLTpars, invDiagPars)
                # if exitifnan
                #     return d, -1
                # end
            end

            # convert gradient of ∇L --> gradient of ltri and logdiag components.
            for s = 1:k
                invLTpars[s].grad .+= ∇Ls[s][tril!(trues(n_d, n_d), -1)] # extract lowertri elements
                invDiagPars[s].grad .+= (∇Ls[s][diagind(∇Ls[s])]  .* exp.(invDiagPars[s].data))

                fix_cov && (invLTpars[s].grad .= 0.; invDiagPars[s].grad .= 0.)

                if any(isnan.(invLTpars[s].grad)) || any(isnan.(invDiagPars[s].grad))
                    display(format("FAILURE IN INVLT GRAD {:d}", s))
                    _failure_dump(ee, d, mupars, invLTpars, invDiagPars)
                    # if exitifnan
                    #     return d, -1
                    # end
                end
            end
            # ====================================================


            # ==> [DEBUGGING] check for problems, and dump a bunch of data if so.
            if isnan(objective)
                @warn "objective is NaN"
                _failure_dump(ee, d, mupars, invLTpars, invDiagPars)
                if exitifnan
                    return d, history, -1
                end
            end
            # ====================================================

            opt()  # Perform gradient step / zero grad.
        end
        # Objective and convergence
        if ee % hist_freq == 0
            @debug format("(bbb) ({:3d}/{:3d}), objective: {:.3f}", ee, epochs, objective)
            c_ix = ee÷hist_freq
            history[c_ix] = objective
            # If lr too high, objective explodes: exit
            if auto_lr && c_ix > 20 && ((history[c_ix] - history[c_ix-1]) > 10 * std(history[(c_ix-20):(c_ix-10)]))
                @warn "possible exploding objective: restarting with lower lr."
                @debug format("(bbb) ({:3d}/{:3d}) FAIL. obj_t {:.3f}, obj_t-1 {:.3f}", ee, epochs, history[c_ix], history[c_ix-1])
                return d, history, false
            end
            # Capture long term trend in stochastic objective
            if ee > s_wdw*hist_freq
                # Moving window
                s_hist[(c_ix - s_wdw+1)] = mean(history[(c_ix - s_wdw+1):(c_ix)])
                if ee > 3*s_wdw*hist_freq &&
                    s_hist[(c_ix- s_wdw+1)] > converge_thrsh*s_hist[(c_ix - 3*s_wdw+1)]
                    history = history[1:c_ix]
                    break
                end
            end
        end
    end
    μs = Tracker.data(mupars)
    Σs = zeros(partype(d), n_d, n_d, k)

    for j in 1:k
        Σs[:,:,j] = let xd=Tracker.data(build_mat(invLTpars[j], invDiagPars[j], n_d)); s=inv(xd'xd); (s+s')/2; end
    end

    d_out = GMM(μs, Σs, ones(k)/k)
    return d_out, history, true
end


function _gmm_entropy_and_grad!(mupars::Matrix{T}, invLT::Array, ∇mu::Matrix{T}, ∇L::Array; M::Int=1, ixs=nothing) where T <: AbstractFloat
    k, n_d = size(mupars)
    normcnst = [sum(log.(diag(invLT[i]))) for i in 1:k]

    Ls = invLT
    Linvs = [inv(x) for x in invLT]
    precs = [L'L for L in Ls]

    ∇mu .= 0
    for i in 1:k
        ∇L[i] .= 0
    end
    objective = 0.

    ixs = something(ixs, 1:k)
    for c in ixs
        x = mupars[c,:] .+ Linvs[c]*randn(n_d, M)

        log_q = Array{T, 2}(undef, k, M)
        for j in 1:k
            log_q[j,:] = _llh_unnorm(x .- mupars[j,:], invLT[j]) .+ normcnst[j]
        end
        R, _mllh = AxUtil.Math.softmax_lse(log_q)
        objective += sum(_mllh)/M - log(k)/M
        Mcj = sum(R, dims=2)  # effective number of points from each cluster j for sample from c

        # Calculate ∇μ, ∇L
        for j in 1:k
            if j == c
                ∇L[c] += Linvs[c]' * Mcj[j] / M
                continue
            end
            @views RXmMu = R[j:j, :] .* (x .- mupars[j, :])

            μ_term = mean(precs[j] * RXmMu, dims=2)
            ∇mu[j,:] += μ_term
            ∇mu[c,:] -= μ_term

            # ===== more difficult ∇L terms ============
            # (s == c) update component c for L_c in MC simulation
            @views ∇L[c] += Linvs[c]' * precs[j] * (RXmMu * (x .- mupars[c, : ])')/M # L_c^{-T}*L_j'L_j(x-μ_j)(x-μ_c)^T
            # (s == j) update component j for factors in likelihood
            @views ∇L[j] += (Mcj[j] * Linvs[j]' - Ls[j] * RXmMu * (x .- mupars[j,:])')/M # L^{-T} - L(x-μ)(x-μ)^T
        end
    end
    return objective
end

function _gmm_entropy_and_grad(mupars::Matrix{T}, invLT::Array; M::Int=1, ixs=nothing) where T <: AbstractFloat
    k, n_d = size(mupars)
    ∇mu = Matrix{T}(undef, k, n_d)
    ∇L = [Matrix{T}(undef, n_d, n_d) for i in 1:k]
    objective = _gmm_entropy_and_grad!(mupars, invLT, ∇mu, ∇L; M=M, ixs=ixs)

    return objective, ∇mu, ∇L
end


function optimise_components_bbb_revkl(d::GMM, log_f::Function, epochs::Int, batch_size_per_cls::Int;
        converge_thrsh::AbstractFloat=0.999, lr::AbstractFloat=1e-3, exitifnan::Bool=false, auto_lr::Bool=true,
        ixs=nothing, reference_gmm=nothing)
    n_d = size(d)
    k = ncomponents(d)
    ixs = something(ixs, 1:k)

    # Set up parameters for optimisation (NOTE: not using AD, but using implementation of ADAM in Flux)
    invLTval = [Matrix(inv(cholesky(d.sigmas[:,:,j]).L)) for j in 1:k]  # note that ∵ inverse, Σ^{-1} = L'L
    invDiagPars = [Flux.param(log.(x[diagind(x)])) for x in invLTval]
    invLTPars = [Flux.param(x[tril!(trues(n_d, n_d), -1)]) for x in invLTval]
    mupars = Flux.param(copy(d.mus))
    opt = ADAM(Tracker.Params((mupars, invLTPars..., invDiagPars...)), lr)

    # create data views of the Flux params for use in algo
    mus = mupars.data
    invDiagval = [x.data for x in invDiagPars]
    invLTval = [x.data for x in invLTPars]

    # Allocate memory for gradient arrays
    ∇_mu = zeros(k, n_d)
    ∇_invdiag = [zeros(n_d) for _ in 1:k]
    ∇_invlt = [zeros(Int(n_d *(n_d-1)/2)) for _ in 1:k]

    # Save objective value history: note because using local exploration of global
    # integral, the original estimate will be heavily under-estimated, and obj usually increases.
    hist_freq = 5   # Sampling freq for history of objective
    history = zeros(Int(floor(epochs/hist_freq)))

    for ee in 1:epochs
        objective = 0.

        # Take sample from each component and backprop through recon. term of KL
        for c in ixs

            # zero gradient (rather than reallocate): I THINK THIS IS UNNECESSARY.
            # nesting all within one call seems to take a performance hit:
            #  --> perhaps 3 level function recursion cannot be inlined?
            zero_arrays!(∇_mu); zero_arrays!(∇_invdiag); zero_arrays!(∇_invlt);

            # useful quantities reqd by objective and gradient.
            normcnst = [sum(invDiagval[r]) for r in 1:k] #.-n_d/2*log(2π)
            invLT = [build_mat(x_lt, x_dia, n_d) for (x_lt, x_dia) in zip(invLTval, invDiagval)]
            invinvLT = [inv(x) for x in invLT]

            # Sample from current approximation and calculate importance weights
            # *IMPORTANT*: we don't want to take gradient of parameters in q: this is
            # similar to EM where the expectation is wrt a *previous* version of the same q.
            x = mus[c,:] .+ inv(invLT[c])*randn(n_d, batch_size_per_cls)
            w = log_f(x) - AxUtil.gmm.gmm_llh_invLT(x, ones(k)/k, mus, invLT) # log imp_wgt
            NNlib.softmax!(w)  # exp(w)/sum(exp(w)) => self normalised imp weights

            # Calculate objective (approx. integral of log GMM density)
            log_q_terms = reduce(vcat, [_llh_unnorm(x .- mus[j,:], invLT[j]) for j in 1:k])
            R, _mllh = AxUtil.Math.softmax_lse(log_q_terms .+ normcnst)
            objective += -dot(_mllh, w)

            # Calculate gradients
            MWcj = sum(R .* w', dims=2)  # effective number of points from each cluster j for sample from c
            precs = [L'L for L in invLT]
            for j in 1:k
                @views RXmMu = R[j:j, :] .* (x .- mus[j, :])
                ∇_mu[j,:] = -sum(w' .* (precs[j] * RXmMu), dims=2)

                wRXmMu = (w' .* RXmMu)
                @views ∇L_term = -(MWcj[j] * invinvLT[j]' - invLT[j] * wRXmMu * (x .- mus[j,:])') # L^{-T} - L(x-μ)(x-μ)^T
                ∇_invlt[j] = ∇L_term[tril!(trues(n_d, n_d), -1)] # extract lowertri elements
                ∇_invdiag[j] = diag(∇L_term) .* exp.(invDiagval[j])
            end

            # tfer gradient for optimisation
            mupars.grad .= ∇_mu
            for j in 1:k
                invLTPars[j].grad .= ∇_invlt[j]
                invDiagPars[j].grad .= ∇_invdiag[j]
            end

            opt()   # take optimisation step (ADAM)
        end

        # Objective and convergence
        if ee % hist_freq == 0
            @debug format("(bbb) ({:3d}/{:3d}), objective: {:.3f}", ee, epochs, objective)
            history[ee÷hist_freq] = objective
        end

    end

    Σs = zeros(partype(d), n_d, n_d, k)
    for j in 1:k
        Σs[:,:,j] = let xd=Tracker.data(build_mat(invLTval[j], invDiagval[j], n_d)); s=inv(xd'xd); (s+s')/2; end
    end
    cgmm = GMM(mus, Σs, ones(k)/k)

    return cgmm, history
end

end  # module

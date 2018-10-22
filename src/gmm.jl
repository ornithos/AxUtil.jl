module gmm
import Flux
using StatsFuns: logsumexp
using Distributions
import Distributions: partype
using Formatting: format
using Random: randperm
using LinearAlgebra: cholesky, logdet
using NNlib: softmax
using AxUtil

export GMM, importance_sample
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
rmcomponents(d::GMM, ixs::BitArray{1}) = GMM{partype(d)}(d.mus[.!ixs, :], d.sigmas[:, :, .!ixs], d.pis[.!ixs])
update(d::GMM; mus=nothing, sigmas=nothing, pis=nothing) = GMM(something(mus, d.mus), something(sigmas, d.sigmas), something(pis, d.pis))

function logpdf(d::GMM, X::Matrix{T}; thrsh_comp=0.005) where T <: AbstractFloat
    return gmm_llh(X, d.pis, d.mus, d.sigmas; thrsh_comp=thrsh_comp)
end

function importance_sample(d::GMM, n::Int, log_f::Function; shuffle=false)
    S = rand(d, n; shuffle=shuffle)
    logW = log_f(S) - logpdf(d, S);
    return S, logW;
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


function gmm_fit(X::Matrix{T}, d::GMM; max_iter=100, tol=1e-3, verbose=true, rm_inactive=false) where T <: AbstractFloat
    gmm_fit(X, d.pis, d.mus, d.sigmas; max_iter=max_iter, tol=tol, verbose=verbose, rm_inactive=rm_inactive)
end


function gmm_fit(X::Matrix{T}, weights::Vector, d::GMM; max_iter=100, tol=1e-3, verbose=true, rm_inactive=false) where T <: AbstractFloat
    gmm_fit(X, weights, d.pis, d.mus, d.sigmas; max_iter=max_iter, tol=tol, verbose=verbose, rm_inactive=rm_inactive)
end


function gmm_fit(X, pi_prior, mu_prior, cov_prior; max_iter=100, tol=1e-3, verbose=true, rm_inactive=false)
    gmm_fit(X, trues(size(X,2)), pi_prior, mu_prior, cov_prior; max_iter=max_iter,
        tol=tol, verbose=verbose, rm_inactive=rm_inactive)
end


function gmm_fit(X, weights, pi_prior, mu_prior, cov_prior; max_iter=100, tol=1e-3, verbose=true, rm_inactive=false)
    p, n = size(X)
    k = length(pi_prior)
    @assert size(weights) == (n,)
    @assert size(mu_prior) == (k, p)
    @assert size(cov_prior) == (p, p, k)
    pis = pi_prior/sum(pi_prior)
    mus = copy(mu_prior)
    sigmas = copy(cov_prior)

    weights = weights / mean(weights)    # diff. to Cappé et al. due to prior

    thrsh_comp = 0.005
    inactive_ixs = pi_prior[:] .< thrsh_comp
    pi_prior = copy(pi_prior)

    for i in range(1, stop=max_iter)
        # E-step
        rs = reduce(vcat, map(j -> log_gauss_llh(X, mus[j,:], sigmas[:,:,j], bypass=inactive_ixs[j]), 1:k)')
        try
            rs .+= log.(pis)[:]
            catch e
            @warn "(gmm) rs and (log) pis are not conformable. The respective values are:"
            display(rs)
            display(log.(pis))
            rethrow(e)
        end
        rs = softmax(rs)

        @debug format("(gmm) ({:3d}/{:3d}), responsibilities: {:s}", i, max_iter, AxUtil.Arr.arr2str(vec(sum(rs, dims=2))))
        # reweight according to importance weights (see Adaptive IS in General Mix. Cappé et al. 2008)
        rs .*= weights'
        @debug format("(gmm) ({:3d}/{:3d}), wgtd responsibilities: {:s}", i, max_iter, AxUtil.Arr.arr2str(vec(sum(rs, dims=2))))

        # M-step
        Ns = vec(sum(rs, dims=2))
        inactive_ixs = Ns .< 1
        @debug format("(gmm) ({:3d}/{:3d}), n_inactive: {:d}", i, max_iter, sum(inactive_ixs))
        active_ixs = .! inactive_ixs
        if any(inactive_ixs)
            pis[inactive_ixs] .= 0.0
            pi_prior[inactive_ixs] .= 0.0
        end
        pis = Ns + pi_prior[:]
        pis /= sum(pis)
        @debug format("(gmm) ({:3d}/{:3d}), π: {:s}", i, max_iter, AxUtil.Arr.arr2str(pis))
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

    end

    out = GMM{typeof(X[1])}(mus, sigmas, pis)
    if rm_inactive
        out = rmcomponents(out, inactive_ixs)
    end

    return out
end



end  # module

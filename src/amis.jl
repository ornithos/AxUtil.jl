module AMIS

function gmm_llh(X, weights, pis, mus, sigmas; disp=false)
    n, p = size(X)
    k = length(pis)
    thrsh_comp = 0.005
    inactive_ixs = pis[:] .< thrsh_comp
    
    P = zeros(n, k)
    for j = 1:k 
        P[:,j] = log_gauss_llh(X, mus[j,:], sigmas[:,:,j], 
            bypass=inactive_ixs[j]) .+ log(pis[j])
    end
    P .*= weights
    if disp
        display(P)
        display(logsumexprows(P))
    end
    return logsumexprows(P)
end

function gmm_prior_llh(pis, mus, sigmas, pi_prior, mu_prior, cov_prior)
    d = size(cov_prior, 1)
    ν = pi_prior # alias
    k = length(pis)
    out = zeros(k)
    @views for j = 1:k
        out[j] += logpdf(MvNormal(mu_prior[j,:], sigmas[:,:,j]/ν[j]), mus[j,:])
        out[j] += -(ν[j] + d + 1)*logdet(sigmas[:,:,j])/2 
        out[j] += -ν[j]*sum(diag(cov_prior[:,:,j]*inv(sigmas[:,:,j])))/2
        out[j] += (ν[j] - 1)*log(pis[j])
    end
    return sum(out)
end


function log_gauss_llh(X, mu, sigma; bypass=false)
    if bypass 
        return -ones(size(X, 1))*Inf
    else
        retval = try _log_gauss_llh(X, mu, sigma)
            catch e
                return -ones(size(X, 1))*Inf
            end
        return retval
    end
end
    
function _log_gauss_llh(X, mu, sigma)
    d = size(X,2)
#     invUT = Matrix(cholesky(inv(sigma)).U)
    invUT = inv(cholesky(sigma).L)
    Z = (X .- mu')*invUT'
    exponent = -0.5*sum(Z.^2, dims=2)
    lognormconst = -d*log(2*pi)/2 -0.5*logdet(sigma) #.+ sum(log.(diag(invUT)))
    return exponent .+ lognormconst
end

function gmm_custom(X, weights, pi_prior, mu_prior, cov_prior; max_iter=100, tol=1e-3, verbose=true)
    n, p = size(X)
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
    Ns = zeros(6)
    
    for i in range(1, stop=max_iter)
        # E-step
        rs = reduce(hcat, map(j -> log_gauss_llh(X, mus[j,:], sigmas[:,:,j], bypass=inactive_ixs[j]), 1:k))
        try
            rs .+= log.(pis)[:]'
            catch e
            display(rs)
            display(log.(pis))
            rethrow(e)
        end
        
        rs = softmax2(rs, dims=2)
        # reweight according to importance weights (see Adaptive IS in General Mix. Cappé et al. 2008)
        rs .*= weights
        
        # M-step
        Ns = sum(rs, dims=1)
        inactive_ixs = Ns[:] .< 1
        active_ixs = logical_not(inactive_ixs)  # can't find a native NOT for BitArrays in Julia
        if any(inactive_ixs)
            pis[inactive_ixs] .= 0.0
            pi_prior[inactive_ixs] .= 0.0
        end
        pis = Ns[:] + pi_prior[:]
        
        pis /= sum(pis)
        
        _mus = reduce(vcat, map(j -> sum(X .* rs[:,j], dims=1) .+ pi_prior[j]*mu_prior[j,:]', findall(active_ixs)))
        _mus ./= vec(Ns[active_ixs] + pi_prior[active_ixs])
        mus[active_ixs,:] = _mus
        
        @views for j in findall(active_ixs)
            Δx = X .- mus[j, :]'
            Δμ = (mus[j,:] - mu_prior[j,:])'
            sigmas[:,:,j] = (Δx.*rs[:,j])'Δx + pi_prior[j]*(Δμ'Δμ + cov_prior[:,:,j])
            sigmas[:,:,j] ./= (Ns[j] + pi_prior[j] + p + 2)     # normalizing terms from Wishart prior
            sigmas[:,:,j] = (sigmas[:,:,j] + sigmas[:,:,j]')/2 + eye(p)*1e-6   # hack: prevent collapse
        end

    end
    
    return pis, mus, sigmas
end

function sample_from_gmm(n, pis, mus, covs; shuffle=true)
    k, p = size(mus)
    @mytimeit to "multinomial"  Ns = rand(Multinomial(n, pis[:]))
    @mytimeit to "active" active_ixs = findall(Ns[:] .>= 1)
    
    @mytimeit to "findixs" ixs = hcat(vcat(1, 1 .+ cumsum(Ns[1:end-1], dims=1)), cumsum(Ns, dims=1))
    out = zeros(n, p)
    for j=active_ixs
        @mytimeit to "MvNorm rand" out[ixs[j,1]:ixs[j,2],:] = rand(MvNormal(mus[j,:], covs[:,:,j]), Ns[j])'
    end
    if shuffle
        out = out[randperm(n),:]
    end
    return out
end


@with_kw struct amis_opt
    epochs::Int64 = 5 
    nodisp::Bool = true
    gmm_smps::Int64 = 1000
    IS_tilt::Float64 = 1.
end



function AMIS(S, logW, k, log_f; kwargs...)
    @unpack_amis_opt reconstruct(amis_opt(), kwargs)
    n, p = size(S)
    
    begin
    W = softmax2(logW, dims=1)
    km = kmeans(copy(S'), k, weights=W)
    
    cmus = zeros(k, p)
    ccovs = zeros(p, p,k)
    for i in range(1, stop=k)
        ixs = findall(x -> isequal(x,i), km.assignments)
        cX = S[ixs, :]; cw = ProbabilityWeights(W[ixs])
        cmus[i,:] = cX' * cw/cw.sum
        ccovs[:,:,i] = StatsBase.cov(cX, cw, corrected=true)
    end

    cpis = zeros(k)
    cnts = countmap(km.assignments)
    for i in 1:k
        try
            cpis[i] = cnts[i]/10
        catch e
            @warn "Cluster $i has no assigned points."
        end
    end
    
    if !nodisp
        f, axs = PyPlot.subplots(5,3, figsize=(8,12))
        plot_is_vs_target(S, W, ax=axs[1,1], c_num=7)

#         plot_level_curves_all(mus, UTs, ax=axs[1,1], color="red")
        for i = 1:k
            axs[1,2][:plot](splat(gaussian_2D_level_curve(cmus[i,:], ccovs[:,:,i]))...);
        end
    end
    
    ν_S = S; ν_W = W;
    end
    
    nsmp = gmm_smps
    
    for i = 1:epochs
        cpis, cmus, ccovs = gmm_custom(ν_S, ν_W, cpis, cmus, ccovs; max_iter=3, tol=1e-3, verbose=false);
        ν_S = sample_from_gmm(1000, cpis, cmus, ccovs*IS_tilt, shuffle=false)
        
        ν_W = log_f(ν_S) - gmm_llh(ν_S, 1, cpis, cmus, ccovs*IS_tilt);
        ν_W = fastexp.(ν_W[:]);
#         @noopwhen (nodisp || i > 5) display(reduce(hcat, [log_f(ν_S), gmm_llh(ν_S, 1, cpis, cmus, ccovs*IS_tilt), ν_W]))
        @noopwhen (nodisp || i > 5) ax = axs[i, 1]
        @noopwhen (nodisp || i > 5) plot_is_vs_target(ν_S, ν_W, ax=ax, c_num=7);
        @noopwhen (nodisp || i > 5) for j = 1:k ax[:plot](splat(gaussian_2D_level_curve(cmus[j,:], ccovs[:,:,j]))...); end
        @noopwhen (nodisp || i > 5) axs[i, 2][:scatter](splat(ν_S)..., alpha=0.2);
        # display(ν_W)
        # display(ν_S)
    end
    return ν_S, ν_W, cpis, cmus, ccovs
end

function GMM_IS(n, pis, mus, covs, log_f)
    S = sample_from_gmm(n, pis, mus, covs, shuffle=false)
    W = log_f(S) - gmm_llh(S, 1., pis, mus, covs, disp=false);
    return S, W;
end

end
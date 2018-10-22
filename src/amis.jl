# module AMIS

# using Distributions, Random
# using LinearAlgebra, StatsBase
# using Parameters, Formatting
# # using Clustering


# # ==> copied from misc.jl. Not sure how to include misc.jl so I can avoid this.
# macro noopwhen(condition, expression)
#     quote
#         if !($condition)
#             $expression
#         end
#     end |> esc
# end

# dropdim1(x) = dropdims(x, dims=1)   # useful for pipes


# function gmm_prior_llh(pis, mus, sigmas, pi_prior, mu_prior, cov_prior)
#     d = size(cov_prior, 1)
#     ν = pi_prior # alias
#     k = length(pis)
#     out = zeros(k)
#     @views for j = 1:k
#         out[j] += logpdf(MvNormal(mu_prior[j,:], sigmas[:,:,j]/ν[j]), mus[j,:])
#         out[j] += -(ν[j] + d + 1)*logdet(sigmas[:,:,j])/2 
#         out[j] += -ν[j]*sum(diag(cov_prior[:,:,j]*inv(sigmas[:,:,j])))/2
#         out[j] += (ν[j] - 1)*log(pis[j])
#     end
#     return sum(out)
# end


# function log_gauss_llh(X, mu, sigma; bypass=false)
#     if bypass 
#         return -ones(size(X, 1))*Inf |> dropdim1
#     else
#         retval = try _log_gauss_llh(X, mu, sigma)
#             catch e
#                 return -ones(size(X, 1))*Inf |> dropdim1
#             end
#         return retval
#     end
# end
    
# function _log_gauss_llh(X, mu, sigma)
#     d = size(X,1)
# #     invUT = Matrix(cholesky(inv(sigma)).U)
#     invLT = inv(cholesky(sigma).L)
#     Z = invLT*(X .- mu)
#     exponent = -0.5*sum(Z.^2, dims=1)  |> dropdim1
#     lognormconst = -d*log(2*pi)/2 -0.5*logdet(sigma)  #.+ sum(log.(diag(invUT)))
#     return exponent .+ lognormconst
# end


# @with_kw struct amis_opt
#     epochs::Int64 = 5 
#     nodisp::Bool = true
#     gmm_smps::Int64 = 1000
#     IS_tilt::Float64 = 1.
# end



# function amis_old(S, logW, k, log_f; kwargs...)
#     @unpack_amis_opt reconstruct(amis_opt(), kwargs)
#     n, p = size(S)
    
#     begin
#     W = softmax2(logW, dims=1)
#     km = kmeans(copy(S'), k, weights=W)
    
#     cmus = zeros(k, p)
#     ccovs = zeros(p, p,k)
#     for i in range(1, stop=k)
#         ixs = findall(x -> isequal(x,i), km.assignments)
#         cX = S[ixs, :]; cw = ProbabilityWeights(W[ixs])
#         cmus[i,:] = cX' * cw/cw.sum
#         ccovs[:,:,i] = StatsBase.cov(cX, cw, corrected=true)
#     end

#     cpis = zeros(k)
#     cnts = countmap(km.assignments)
#     for i in 1:k
#         try
#             cpis[i] = cnts[i]/10
#         catch e
#             @warn "Cluster $i has no assigned points."
#         end
#     end
    
#     if !nodisp
#         f, axs = PyPlot.subplots(5,3, figsize=(8,12))
#         plot_is_vs_target(S, W, ax=axs[1,1], c_num=7)

# #         plot_level_curves_all(mus, UTs, ax=axs[1,1], color="red")
#         for i = 1:k
#             axs[1,2][:plot](splat(gaussian_2D_level_curve(cmus[i,:], ccovs[:,:,i]))...);
#         end
#     end
    
#     ν_S = S; ν_W = W;
#     end
    
#     nsmp = gmm_smps
    
#     for i = 1:epochs
#         cpis, cmus, ccovs = gmm_custom(ν_S, ν_W, cpis, cmus, ccovs; max_iter=3, tol=1e-3, verbose=false);
#         ν_S = sample_from_gmm(1000, cpis, cmus, ccovs*IS_tilt, shuffle=false)
        
#         ν_W = log_f(ν_S) - gmm_llh(ν_S, 1., cpis, cmus, ccovs*IS_tilt);
#         ν_W = fastexp.(ν_W[:]);
# #         @noopwhen (nodisp || i > 5) display(reduce(hcat, [log_f(ν_S), gmm_llh(ν_S, 1, cpis, cmus, ccovs*IS_tilt), ν_W]))
#         @noopwhen (nodisp || i > 5) ax = axs[i, 1]
#         @noopwhen (nodisp || i > 5) plot_is_vs_target(ν_S, ν_W, ax=ax, c_num=7);
#         @noopwhen (nodisp || i > 5) for j = 1:k ax[:plot](splat(gaussian_2D_level_curve(cmus[j,:], ccovs[:,:,j]))...); end
#         @noopwhen (nodisp || i > 5) axs[i, 2][:scatter](splat(ν_S)..., alpha=0.2);
#         # display(ν_W)
#         # display(ν_S)
#     end
#     return ν_S, ν_W, cpis, cmus, ccovs
# end

# function gmm_is(n, pis, mus, covs, log_f)
#     S = sample_from_gmm(n, pis, mus, covs, shuffle=false)
#     W = log_f(S) - gmm_llh(S, 1., pis, mus, covs, disp=false);
#     return S, W;
# end

# end
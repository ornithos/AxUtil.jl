using Distributions, Random
using Base.Threads: @threads
# loosely based / adapted from code originally written by Vadim Smolyakov.
# https://github.com/vsmolyakov


#= groupinds(x)
  for x isa Vector{Signed}.
  Extract the group of indices which match each unique value in x.
  Return, both the list of groups, and the unique elements they correspond to.
  adapted from GroupSlices.jl
 =#
function groupinds(ic::Vector{T}) where T <: Signed  # SignedInt
    d = Dict{T, T}()
    ia = unique(ic)
    n = length(ia)
    for i = 1:n
        d[ia[i]]= i
    end

    ib = Array{Vector{T}}(undef, n)
    for k = 1:n
        ib[k] = T[]
    end

    for h = 1:length(ic)
        push!(ib[d[ic[h]]], h)
    end
    return ib, ia
end


#=
  dist_to_centers.
  Helper function for dpmeans: calculate all distances ||x_i - μ_j|| over i, j.
 =#
function dist_to_centers(X::Matrix{T}, M::Matrix{T}; sq=false) where T <: Real
    n, d = size(X)
    k, _d2 = size(M)
    @assert (d == _d2) "X and M must have the same row dimension."

    X2    = sum(X .* X, dims=2)
    X_Mt  = X * M'
    M2    = sum(M .* M, dims=2)
    norm2 = (X2 .- 2X_Mt) .+ M2'
    if sq
        return norm2
    else
        return sqrt.(norm2)
    end
end

#=
  row_mins.
  Helper function for dpmeans. Calculate the minimum of each row, and return
  the argmin in the second argument. MUCH MUCH faster than mapslices to
  `findmin`.
=#
function row_mins(X::Matrix{T}) where T <: Real
    n, d = size(X)
    out_v = Vector{T}(undef, n)
    out_ind = Vector{Int32}(undef, n)
    for nn = 1:n
        _min = X[nn, 1]
        _argmin = 1
        for dd = 2:d
            if X[nn, dd] < _min
                _min = X[nn, d]
                _argmin = dd
            end
        end
        out_v[nn] = _min
        out_ind[nn] = _argmin
    end
    return out_v, out_ind
end


# function kpp_init(X::Array{Float64,2}, k::Int64)
#
#     n, d = size(X)
#     mu = zeros(k, d)
#     dist = Inf * ones(n)
#     λ = 1
#
#     idx = rand(1:n)
#     mu[1,:] = X[idx,:]
#     for i = 2:k
#         D = X .- mu[i-1,:]'
#         dist = min.(dist, sum(D.*D, dims=2))
#         idx_cond = rand() .< cumsum(dist/sum(dist))
#         idx_mu = findfirst(idx_cond)
#         mu[i,:] = X[idx_mu,:]
#         λ = maximum(dist)
#         println(λ)
#     end
#     return mu, λ
# end


function dpmeans_fit(X::Array{Float64, 2}; k_init::Int64=3, max_iter::Int64=100,
                     shuffleX::Bool=true)

    #init params
    k = k_init
    n, d = size(X)
    # shuffleX && X = X[randperm(n), :]

    mu = mean(X, dims=1)
    # display(mu)
    # throw("Need to set lambda intelligently otherwise will continually add clusters.")
    λ = 3
    # mu, λ = kpp_init(X, k)

    # println("lambda: ", λ)

    obj = zeros(max_iter)
    em_time = zeros(max_iter)

    obj_tol = 1e-3
    n, d = size(X)

    for iter = 1:max_iter
        dist = zeros(n, k)

        #assignment step
        # custom functions are > order of magnitude faster
        dist = dist_to_centers(X, mu; sq=true)
        dmin, z = row_mins(dist)

        #find(dist->dist==0, dist)
        bad_idx = findall(dmin .> λ)
        if length(bad_idx) > 0
            k += 1
            z[bad_idx] .= k
            new_mean = mean(view(X, bad_idx, :), dims=1)
            mu = vcat(mu, new_mean)
            Xm = X .- new_mean
            dist = hcat(dist, sum(Xm.*Xm, dims=2))
        end

        #update step
        k_inds, ks = groupinds(z)
        for j = 1:length(ks)
            @views mu[ks[j],:] = mean(X[k_inds[j], :], dims=1)
            @views obj[iter] += sum(dist[k_inds[j], ks[j]])
        end
        obj[iter] = obj[iter] + λ * k

        #check convergence
        if (iter > 1 && abs(obj[iter] - obj[iter-1]) < obj_tol * obj[iter])
            println("converged in ", iter, " iterations.")
            global Z = z
            break
        elseif iter == max_iter
            @warn "DPmeans not converged"
            global Z = z
        end
    end

    return Z, mu, obj
end

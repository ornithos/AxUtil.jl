using Distributions, Random

# adapted from code initially written by Vadim Smolyakov.
# https://github.com/vsmolyakov

function kpp_init(X::Array{Float64,2}, k::Int64)

    n, d = size(X)
    mu = zeros(k, d)
    dist = Inf * ones(n)
    λ = 1

    idx = rand(1:n)
    mu[1,:] = X[idx,:]
    for i = 2:k
        D = X .- mu[i-1,:]'
        dist = min.(dist, sum(D.*D, dims=2))
        idx_cond = rand() .< cumsum(dist/sum(dist))
        idx_mu = findfirst(idx_cond)
        mu[i,:] = X[idx_mu,:]
        λ = maximum(dist)
        println(λ)
    end
    return mu, λ
end

function dpmeans_fit(X::Array{Float64, 2}; k_init::Int64=3, max_iter::Int64=100)

    #init params
    k = k_init
    n, d = size(X)

    mu = mean(X, dims=1)
    display(mu)
    throw("Need to set lambda intelligently otherwise will continually add clusters.")
    λ = 3
    # mu, λ = kpp_init(X, k)

    println("lambda: ", λ)

    obj = zeros(max_iter)
    em_time = zeros(max_iter)

    obj_tol = 1e-3
    n, d = size(X)

    for iter = 1:max_iter
        dist = zeros(n, k)

        #assignment step
        for kk = 1:k
            Xm = X .- mu[kk, :]'
            dist[:,kk] = sum(Xm.*Xm, dims=2)
        end

        #update labels
        distmin = mapslices(x->collect(findmin(x)), dist, dims=2)
        dmin, z = distmin[:, 1], distmin[:, 2]   # row-wise min, argmin

        #find(dist->dist==0, dist)
        bad_idx = findall(dmin .> λ)
        if length(bad_idx) > 0
            k += 1
            z[bad_idx] .= k
            new_mean = mean(X[bad_idx, :], dims=1)
            mu = vcat(mu, new_mean)
            Xm = X .- new_mean
            dist = hcat(dist, sum(Xm.*Xm, dims=2))
        end

        #update step
        for kk = 1:k
            matches = z .== k
            if sum(matches) > 0
                mu[kk,:] = mean(X[matches, :], dims=1)
                obj[iter] += sum(dist[matches, kk])
            end
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

    return Z, k, obj
end

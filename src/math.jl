module Math

using Base.Threads: @threads
using LinearAlgebra
using StatsFuns: logsumexp
using NNlib
using ..Arr: eye

function softmax2(logp; dims=2)
    p = exp.(logp .- maximum(logp, dims=dims))
    p ./= sum(p, dims=dims)
    return p
end
@deprecate softmax2(logp; dims=2) NNlib.softmax(logp)


function softmax_lse!(out::AbstractVecOrMat{T}, xs::AbstractVecOrMat{T}) where T<:AbstractFloat
    #=
    Simultaneous softmax and logsumexp. Useful for calculating an estimate of logp(x) out of
    lots of MC samples, as well as calculating the approximate posterior.
    We essentially get the Logsumexp for free here.
    (adapted from Mike I's code in NNlib.)
    =#
    lse = Array{T, 1}(undef, size(out, 2))

    @threads for j = 1:size(xs, 2)
        @inbounds begin
            # out[end, :] .= maximum(xs, 1)
            m = xs[1, j]
            for i = 2:size(xs, 1)
                m = max(m, xs[i, j])
            end
            m = out[end, j]
            # out .= exp(xs .- out[end, :])
            for i = 1:size(out, 1)
                out[i, j] = exp(xs[i, j] - m)
            end
            # out ./= sum(out, 1)
            s = zero(eltype(out))
            for i = 1:size(out, 1)
                s += out[i, j]
            end
            for i = 1:size(out, 1)
                out[i, j] /= s
            end
            lse[j] = log(s) + m
        end
    end
    return out, lse
end

softmax_lse!(xs) = softmax_lse!(xs, xs)
softmax_lse(xs) = softmax_lse!(similar(xs), xs)


# logsumexprows: see flux.jl

function logsumexpcols(X::AbstractArray{T}) where {T<:Real}
    n = size(X,2)
    out = zeros(n)
    Base.Threads.@threads for i = 1:n
        @views out[i] = logsumexp(X[:,i])
    end
    return out
end


function sq_diff_matrix(X, Y)
    """
    Constructs (x_i - y_j)^T (x_i - y_j) matrix where
    X = Array{Float}(n_x, d)
    Y = Array{Float}(n_y, d)
    return: np.array(n_x, n_y)
    """
    normsq_x = sum(X.^2, dims=2)
    normsq_y = sum(Y.^2, dims=2)
    out = normsq_x .+ normsq_y'
    out -= 2*X * Y'
    return out
end


function medoid(x::Matrix{T}) where T <: Number
    """
    Calculates the medoid of a set of datapoints
    stored as rows in matrix x.
    This element is the one that has the minimum
    L2 distance to all other points in the set.
    """
    distMatrix = AxUtil.Math.sq_diff_matrix(x, x)
    ix = findmin(sum(distMatrix, dims=1)[:])[2]
    return x[ix,:], ix
end

#=================================================================================
                    Special Matrix Constructors
==================================================================================#

function make_lt(x, d::Int)
    @assert (length(x) == Int(d*(d+1)/2))
    M = zeros(d,d)
    x_i = 1
    for j=1:d, i=j:d
        M[i,j] = x[x_i]
        x_i += 1
    end
    return M
end


function unmake_lt(M, d)
    return M[tril!(trues(d,d))]
end


function make_lt_strict(x, d::Int)
    @assert (length(x) == Int(d*(d-1)/2))
    M = zeros(d,d)
    x_i = 1
    for j=1:d-1, i=j+1:d
        M[i,j] = x[x_i]
        x_i += 1
    end
    return M
end


function unmake_lt_strict(M, d)
    return M[tril!(trues(d,d), -1)]
end


function make_skew(x, d)
    S = make_lt_strict(x, d::Int)
    return S - S'
end


function cayley_orthog(x, d)
    S = make_skew(x, d)
    I = eye(d)
    return (I - S) / (I + S)  # (I - S)(I + S)^{-1}. Always nonsingular.
end


 #=================================================================================
                        Numerical Gradient checking
 ==================================================================================#

function num_grad(fn, X, h=1e-8; verbose=true)
    """
    Calculate finite differences gradient of function fn evaluated at numpy
    array X. Not calculating central diff to improve speed and because some
    authors disagree about benefits.

    Most of the code here is really to deal with weird sizes of inputs or
    outputs. If scalar input and multi-dim output or vice versa, we return
    gradient in the shape of the multi-dim input or output. However, if both
    are multi-dimensional then we return as n_output vs n_input matrix
    where both input/outputs have been vectorised if necessary.
    """
    shp = size(X)
    resize_x = ndims(X) > 1
    rm_xdim = isa(X, Real)
    n = length(X)

    f_x = fn(X)
    if isa(f_x, Real)
        im_f_shp = 0
        resize_y = false
    else
        im_f_shp = size(f_x)
        resize_y = !(ndims(f_x) <= 2 && any(im_f_shp .== 1))
        @assert ndims(f_x) <= 2 "image of fn is tensor. Not supported."
    end

    m = Int64(prod(max.(im_f_shp, 1)))

    X = X[:]
    g = zeros(m, n)
    for ii in range(1, stop=n)
        Xplus = convert(Array{Float64}, copy(X))
        Xplus[ii] += h
        Xplus = reshape(Xplus, shp)
        grad = (fn(Xplus) - f_x) ./ h
        if ndims(grad) >= 1
            grad = ndims(grad) > 1 ? grad[:] : grad
            g[:, ii] = grad
        else
            g[ii] = grad
        end
    end

    verbose && resize_x && resize_y &&
        println("WARNING: Returning gradient as matrix size n(fn output) x n(variables)")

    if rm_xdim && size(g,2) == 1
        g = g[:]
    elseif resize_x && !any(im_f_shp .> 1)
        g = reshape(g, shp)
    elseif resize_y && !any(shp .> 1)
        g = reshape(g, im_f_shp)
    end

    return g
end


function num_grad_spec(fn, X, cart_ix, h=1e-8; verbose=true)
    """
    As per num_grad, but specifying a Cartesian Index ('cart_ix').
    Thus a n(fn output) x 1 array will always be returned. At
    some point it makes sense to roll this into my base numgrad, but
    I don't want to yet as it complicates things.
    """
    shp = size(X)
    resize_x = ndims(X) > 1
    rm_xdim = isa(X, Real)
    n = length(X)

    f_x = fn(X)
    if isa(f_x, Real)
        im_f_shp = 0
        resize_y = false
    else
        im_f_shp = size(f_x)
        resize_y = !(ndims(f_x) <= 2 && any(im_f_shp .== 1))
        @assert ndims(f_x) <= 2 "image of fn is tensor. Not supported."
    end

    m = Int64(prod(max.(im_f_shp, 1)))

    g = zeros(m, 1)
    Xplus = convert(Array{Float64}, copy(X))
    Xplus[cart_ix] += h
    Xplus = reshape(Xplus, shp)
    g = (fn(Xplus) - f_x) ./ h

    return g
end

end

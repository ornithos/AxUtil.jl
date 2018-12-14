module Flux

using LinearAlgebra
import StatsFuns: logsumexp

using Flux, Test
using Flux: Tracker
using Flux.Tracker: @grad, gradcheck
import NNlib

using ..Arr: eye
using ..Math: make_lt, unmake_lt, make_lt_strict, unmake_lt_strict


FLUX_TESTS = false   # perform gradient checks

# Make Lower Triangular Matrix / Can backprop through
# ===================================
# => Moved to Math
# function make_lt(x, d::Int)
#     @assert (length(x) == Int(d*(d+1)/2))
#     M = zeros(d,d)
#     x_i = 1
#     for j=1:d, i=j:d
#         M[i,j] = x[x_i]
#         x_i += 1
#     end
#     return M
# end
#
# function unmake_lt(M, d)
#     return M[tril!(trues(d,d))]
# end

make_lt(x::TrackedArray, d::Int) = Tracker.track(make_lt, x, d)

@grad function make_lt(x, d::Int)
    return make_lt(Tracker.data(x), d), Δ -> (unmake_lt(Δ, d), nothing)
end


# Make Strictly Lower Triangular Matrix
# ===================================
# => Moved to Math
# function make_lt_strict(x, d::Int)
#     @assert (length(x) == Int(d*(d-1)/2))
#     M = zeros(d,d)
#     x_i = 1
#     for j=1:d-1, i=j+1:d
#         M[i,j] = x[x_i]
#         x_i += 1
#     end
#     return M
# end
#
# function unmake_lt_strict(M, d)
#     return M[tril!(trues(d,d), -1)]
# end

make_lt_strict(x::TrackedArray, d::Int) = Tracker.track(make_lt_strict, x, d)

@grad function make_lt_strict(x, d::Int)
    return make_lt_strict(Tracker.data(x), d), Δ -> (unmake_lt_strict(Δ, d), nothing)
end


# Make Diagonal Matrix (current Flux version is more general but is Tracked{Tracked} :/  ).
# ===================================
function diag0(x)
    d = length(x)
    M = zeros(d,d)
    M[diagind(M)] = x
    return M
end

diag0(x::TrackedArray) = Tracker.track(diag0, x)

@grad function diag0(x)
    return diag0(Tracker.data(x)), Δ -> (Δ[diagind(Δ)],)
end


# Gradient Tester Utilities from Flux
gradtest(f, xs::AbstractArray...) = gradcheck((xs...) -> sum(sin.(f(xs...))), xs...)
gradtest(f, dims...) = gradtest(f, rand.(Float64, dims)...)


# For use by logsumexprows/cols. However, using the StatsFuns version with
# usual Flux broadcasting etc. works faster usually!
logsumexp(X::TrackedVector) = Tracker.track(logsumexp, X)

@grad function logsumexp(X)
  return logsumexp(X.data), Δ -> (NNlib.softmax(X.data) .* Δ',)
end


# Row-wise logsumexp. See math.jl in AxUtil. Gradient is fairly efficient with below.
# ===================================
function logsumexprows(X::Matrix{T}) where {T<:Real}
    #= iterate over rows of matrix with StatsFuns' logsumexp.
       This is primarily useful since we have a Flux-enabled
       version in the flux src in AxUtil.
    =#
    n = size(X,1)
    out = zeros(n)
    Threads.@threads for i = 1:n
        out[i] = logsumexp(X[i,:])
    end
    return out
end

logsumexprows(X::TrackedArray) = Tracker.track(logsumexprows, X)

@grad function logsumexprows(X)
  return logsumexprows(X.data), Δ -> (Δ .* NNlib.softmax(X.data')',)
end


function logsumexpcols(X::Matrix{T}) where {T<:Real}
    n = size(X,2)
    out = zeros(n)
    Threads.@threads for i = 1:n
        @views out[i] = logsumexp(X[:,i])
    end
    return out
end

logsumexpcols(X::TrackedArray) = Tracker.track(logsumexpcols, X)

@grad function logsumexpcols(X)
  return logsumexpcols(X.data), Δ -> (NNlib.softmax(X.data) .* Δ',)
end


if FLUX_TESTS
    @test gradtest((x, A) -> make_lt(log.(x), 5) * A, 10, (5,5))
    @test gradtest((x, A) -> diag0(cos.(x)) * A, 5, (5,5))
    @test gradtest((X) -> sin.(logsumexp(X .* X)), (6,))
    @test gradtest((X) -> sin.(logsumexprows(X .* X)), (6,10))
    @test gradtest((X) -> sin.(logsumexpcols(X .* X)), (6,10))
end


#==================================================================
                Flux Recurrent Cells
==================================================================#


mutable struct LDSCell_simple
    A::Union{AbstractArray, TrackedArray}
    h::Array{Float64,1}
end

# Operation
function (m::LDSCell_simple)(h, x)
    A = m.A
    h = m.A * h
    return h, h
end

hidden(m::LDSCell_simple) = m.h



mutable struct LDSCell_simple_u
    A::Union{AbstractArray, TrackedArray}
    B::Union{AbstractArray, TrackedArray}
    h::Array{Float64,1}
end

# Operation
function (m::LDSCell_simple_u)(h, x)
    A, B = m.A, m.B
    h = A * h + B * x
    return h, h
end

hidden(m::LDSCell_simple_u) = m.h


mutable struct LDSCell_simple_diag_u
    A::Union{AbstractArray, TrackedArray}
    B::Union{AbstractArray, TrackedArray}
    h::Array{Float64,1}
end

# Operation
function (m::LDSCell_simple_diag_u)(h, x)
    A, B = m.A, m.B
    h = A .* h + B .* x
    return h, h
end

hidden(m::LDSCell_simple_diag_u) = m.h


LDSCell_scalar_u(a::Tracker.TrackedReal, b::Tracker.TrackedReal, h) = LDSCell_scalar_u_track(a, b, h)
LDSCell_scalar_u(a::AbstractFloat, b::AbstractFloat, h) = LDSCell_scalar_u_notrk(a, b, h)

mutable struct LDSCell_scalar_u_notrk
    A::AbstractFloat
    B::AbstractFloat
    h::AbstractFloat
end

# Operation
function (m::LDSCell_scalar_u_notrk)(h, x)
    A, B = m.A, m.B
    h = A * h + B * x
    return h, h
end

hidden(m::LDSCell_scalar_u_notrk) = m.h


mutable struct LDSCell_scalar_u_track
    A::Tracker.TrackedReal
    B::Tracker.TrackedReal
    h::Tracker.TrackedReal
end

# Operation
function (m::LDSCell_scalar_u_track)(h, x)
    A, B = m.A, m.B
    h = A * h + B * x
    return h, h
end

hidden(m::LDSCell_scalar_u_track) = m.h
end

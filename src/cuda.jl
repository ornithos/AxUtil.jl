module CUDA

using ..CuArrays
import ..Math: @argcheck, LinearAlgebra, tril!
import ..Math: make_lt, make_lt_strict, make_lt!, make_lt_strict!, cayley_orthog,
            unmake_lt, unmake_lt_strict, make_skew
import ..Arr: eye
import ..Flux: TrackedArray, diag0
import Base: \, inv

function make_lt(x::CuArray{T,1}, d::Int)::CuArray{T,2} where T <: Real
    @argcheck (length(x) == Int(d*(d+1)/2))
    make_lt!(CuArrays.zeros(T, d, d), x, d)
end

function make_lt_strict(x::CuArray{T,1}, d::Int)::CuArray{T,2} where T <: Real
    @argcheck (length(x) == Int(d*(d-1)/2))
    return make_lt_strict!(CuArrays.zeros(T, d,d), x, d)
end

function unmake_lt(M::CuMatrix{T}, d)::CuArray{T,1} where T <: Real
    return M[tril!(convert(CuMatrix{Bool}, trues(d,d)))]
end

function unmake_lt_strict(M::CuMatrix{T}, d::Int)::CuArray{T,1} where T <: Real
    return M[tril!(convert(CuMatrix{Bool}, trues(d,d)), -1)]
end

function \(A::CuArray{T, 2}, B::CuArray{T, 2}) where T <: AbstractFloat
        @assert size(A) == size(B) "Only square matrices implemented for ldiv CuArrays (via LU)! (Needs QR for rectangular.)"
        luA, ipiv = CuArrays.CUSOLVER.getrf!(copy(A))
        CuArrays.CUSOLVER.getrs!('N', luA, ipiv, copy(B))
end

function inv(A::CuArray{T, 2}) where T <: AbstractFloat
    d = LinearAlgebra.checksquare(A)
    A \ convert(CuMatrix{T}, eye(d))
end


const CuVecPossiblyTracked{T} = Union{CuVector, TrackedArray{T, 1, CuVector{T}}} where T <: AbstractFloat
const CuMatPossiblyTracked{T} = Union{CuVector, TrackedArray{T, 2, CuMatrix{T}}} where T <: AbstractFloat

function cayley_orthog(x::CuVecPossiblyTracked{T}, d)::CuMatPossiblyTracked{T} where T <: AbstractFloat
    I = convert(CuMatrix{T}, eye(d))            # segfault using UniformScaling I from LinearAlgebra. Not ideal.
    S = make_skew(x, d)
    return (I + S) \ (I - S)  # note this is the more efficient way around than for CPU (legacy)
end

function diag0(x::CuArray{T,1})::CuArray{T,2} where T <: Real
    d = length(x)
    M = CuArrays.zeros(T, d,d)
    M[diagind(M)] = x
    return M
end

# So far, so good. NOW I NEED TO IMPLEMENT INVERSE (INV) JUST LIKE I DID FOR
# LDIV, FOR THE BACKWARD PASS. (So go through generic.jl in LinearAlgebra again...)

end

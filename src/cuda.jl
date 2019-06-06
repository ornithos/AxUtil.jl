module CUDA

using ..CuArrays
using ..ArgCheck
import ..Math: make_lt, make_lt_strict

function make_lt(x::CuArray{T,1}, d::Int)::CuArray{T,2} where T <: Real
    @argcheck (length(x) == Int(d*(d+1)/2))
    make_lt!(CuArrays.zeros(T, d, d), x, d)
end
function make_lt_strict(x::CuArray{T,1}, d::Int)::CuArray{T,2} where T <: Real
    @argcheck (length(x) == Int(d*(d-1)/2))
    return make_lt_strict!(CuArrays.zeros(T, d,d), x, d)
end

end

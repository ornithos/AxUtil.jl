using Flux: TrackedVector, TrackedMatrix
using Flux.Tracker: TrackedVecOrMat, TrackedReal
using Dates, Formatting
import Base: convert

macro noopwhen(condition, expression)
    quote
        if !($condition)
            $expression
        end
    end |> esc
end


function countmap(x::Vector{T}; d::Int=-1) where T <: Signed
    # approx. 10x faster than StatsBase for small arrays of ints.
    # Note that it always enumerates each possible integer up to max.
    d = d > 0 ? d : maximum(x)
    out = zeros(Int64, d)
    for cx in x
        out[cx] += 1
    end
    return out
end


function countmap_labelled(x::Vector{T}) where T <: Signed
    # as countmap, but not array of contiguous labels: a labelled
    # vector is also returned, and any key of zero count is removed.
    sort!(x)
    lbls = unique(x)
    d = length(lbls)
    out = zeros(Int64, d)
    px = x[1]
    k = 1
    for cx in x
        if cx != px
            px = cx
            k += 1
        end
        out[k] += 1
    end
    return out, lbls
end


invert_index(x::Vector{T}) where T <: Signed = sortperm(x)

# useful for pipes
dropdim1(x::Union{Number, TrackedReal}) = x
dropdim1(x::Union{VecOrMat, TrackedVecOrMat}) = dropdims(x, dims=1)

dropdim2(x::Union{Matrix, TrackedMatrix}) = dropdims(x, dims=2)
dropdim2(x::Union{Vector, TrackedVector}) = x
dropdim2(x::Union{Number, TrackedReal}) = x


# converting NamedTuple -> Dict fails if multiple types in the NamedTuple.
# The *collect* in the second zip arg here is the work-around.
convert(::Type{Dict}, nt::NamedTuple) = Dict(k => v for (k,v) in zip(keys(nt), collect(values(nt))))


# issue #29560 mkborregaard solution to repelem/rep like behaviour for vectors.
# Will likely be obselete when PR accepted.
#
# NOTICE THAT THE FIRST FUNCTION WILL RETURN A 2D MATRIX EVEN IF MATRIX GIVEN.
# (Hence my vector function below.)
function Base.repeat(a::AbstractMatrix, m::AbstractVector{<:Integer}, n::Integer=1)
    o, p = size(a,1), size(a,2)
    length(m) == o || throw(ArgumentError("m must be an AbstractVector{<:Integer} of same dimension as a"))
    b = similar(a, sum(m), p*n)
    for j=1:n
        d = (j-1)*p+1
        R = d:d+p-1
        c = 0
        for i ∈ eachindex(m)
            r = m[i]
            v = a[i,:]
            while r > 0
                b[c+=1, R] = v
                r -= 1
            end
        end
    end
    return b
end

function Base.repeat(a::AbstractVector, m::AbstractVector{<:Integer})
    o, p = size(a,1), size(a,2)
    length(m) == o || throw(ArgumentError("m must be an AbstractVector{<:Integer} of same dimension as a"))
    b = similar(a, sum(m))
    c = 0
    for i ∈ eachindex(m)
        r = m[i]
        v = a[i]
        while r > 0
            b[c+=1] = v
            r -= 1
        end
    end
    return b
end



function construct_unique_filename(filestem; path="./", date_fmt="yyyy_mm_dd", ext="")
    fnm_base = filestem * Dates.format(now(), date_fmt)
    fnm_uid = 0
    _construct_fnm(uid) = joinpath(path, fnm_base * "_" * format("{:03d}", uid) * ext)
    while isfile(_construct_fnm(fnm_uid))
        fnm_uid += 1
    end
    return _construct_fnm(fnm_uid)
end

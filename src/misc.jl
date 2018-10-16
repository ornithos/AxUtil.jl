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
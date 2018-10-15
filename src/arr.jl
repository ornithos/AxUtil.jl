module Arr

using LinearAlgebra

arange(start; kwargs...) = collect(range(start; kwargs...))

unpack_arr(x::AbstractArray) = [x[:,i] for i in range(1,stop=size(x,2))]
hstack(x) = reduce(hcat, x)
vstack(x) = reduce(vcat, x)
eye(d) = Matrix(I, d, d)

end
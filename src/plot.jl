module Plot

using PyPlot
using LinearAlgebra
using Flux: Tracker


 #=================================================================================
                                     Utilities
 ==================================================================================#

function subplot_gridsize(num)
    poss = [[x,Int(ceil(num/x))] for x in range(1,stop=Int(floor(sqrt(num)))+1)]
    choice = findmin([sum(x) for x in poss])[2]  # argmin
    return sort(poss[choice])
end


function gaussian_2D_level_curve_pts(mu::Array{Float64,1}, sigma::Array{Float64, 2}, alpha=2, ncoods=100)
    @assert size(mu) == (2,) "mu must be vector in R^2"
    @assert size(sigma) == (2, 2) "sigma must be 2x2 array"

    U, S, V = svd(sigma)

    sd = sqrt.(S)
    coods = range(0, stop=2*pi, length=ncoods)
    coods = hcat(sd[1] * cos.(coods), sd[2] * sin.(coods))' * alpha
    
    coods = (V' * coods)' # project onto basis of ellipse
    coods = coods .+ mu' # add mean
    return coods
end


 #=================================================================================
                                   Plotting zoo
 ==================================================================================#

function pairplot(X::AbstractArray{T, 2}; figsize=(10,10), alpha=0.5, bins=50) where T <: AbstractFloat
    n, d = size(X)
    (d > 20) && throw("will not plot for d > 20")

    fig, axs = PyPlot.subplots(d, d, figsize=figsize)
    for ix = 1:d, iy = 1:d
        if ix != iy
            if alpha isa Number
                axs[ix, iy][:scatter](X[:, ix], X[:, iy], alpha=alpha)
            elseif alpha isa AbstractArray
                scatter_alpha(X[:,ix], X[:,iy], alpha, ax=axs[ix,iy])
            end
        else
            axs[ix, iy][:hist](X[:, ix], bins=bins)
        end
    end
end


function scatter_arrays(xs...)
    n = length(xs)
    sz = subplot_gridsize(n)
    f, axs = PyPlot.subplots(sz..., figsize=(5 + (sz[2]>1), sz[1]*3))
    if n == 1   # axs is not an array!
        axs[:scatter](unpack_arr(xs[1])...)
        return
    else
        for i in eachindex(xs)
            ax = axs[i]; x = xs[i]
            ax[:scatter](unpack_arr(x)...)
        end
    end
end


function scatter_alpha(x1::Vector{T}, x2::Vector{T}, alpha::Vector{T2}; cmap_ix::Int=0, cmap::String="tab10", 
        rescale_alpha::Bool=true, ax=nothing) where T <: Real where T2 <: AbstractFloat
    n = length(alpha)
    ax = something(ax, gca())
    cols = repeat(collect(ColorMap(cmap)(cmap_ix))', n, 1)
    rescale_alpha ? (alpha /= maximum(alpha)) : nothing
    cols[:,4] = alpha
    ax[:scatter](x1, x2, color=cols)
end

 #=================================================================================
                        MNIST-style image grid
 ==================================================================================#


function _gen_cube_grid(vmin, vmax; nsmp_dim=8, ndim=2, force=false)
    #=
    Generate coordinates for equally spaced meshgrid in `ndim` dimensions.
    The output is in 'long' rather than 'wide' form: i.e. if `ndim=2`,
    an n x 2 matrix is returned.
    =#
    (!force) && nsmp_dim^ndim > 1e4 && throw("More than 10,000 points requested.")
    xs = collect(range(vmin, stop=vmax, length=nsmp_dim))
    xs = hcat(repeat(xs, outer=(nsmp_dim,1)), repeat(xs, inner=(nsmp_dim,1)))
    return xs
end


function _gen_grid(xrng, yrng; nsmp_dim=8, ndim=2, force=false)
    #=
    Generate coordinates for equally spaced meshgrid in `ndim` dimensions.
    The output is in 'long' rather than 'wide' form: i.e. if `ndim=2`,
    an n x 2 matrix is returned.
    =#
    (!force) && nsmp_dim^ndim > 1e4 && throw("More than 10,000 points requested.")
    xs = collect(range(xrng[1], stop=xrng[2], length=nsmp_dim))
    ys = collect(range(yrng[1], stop=yrng[2], length=nsmp_dim))
    return hcat(repeat(xs, outer=(nsmp_dim,1)), repeat(ys, inner=(nsmp_dim,1)))
end


function _tile_image_grid(Ms; gridsz=nothing)
    
    num = length(Ms)
    # primarily to catch user fails of passing in a single array
    @assert num <= 256 "too many images. Expecting at most 256."
    @assert ndims(Ms[1]) == 2 "each element of Ms should be a 2 dimensional image array"
    @assert maximum(size(Ms[1]))*sqrt(float(num)) <= 4000 "resulting image array is too large. Reduce number of Ms."
    @assert length(unique([size(Ms[i]) for i in 1:num])) == 1 "all arrays in Ms should be same size."
    
    Ms_sz = size(Ms[1])
    
    if gridsz == nothing
        poss = [[x,Int(ceil(num/x))] for x in range(1,stop=Int(floor(sqrt(num)))+1)]
        resultsz = [x .* Ms_sz for x in poss]
        choice = findmin([sum(x) for x in resultsz])[2]
        gridsz = sort(poss[choice])
    else
        gridsz = collect(gridsz)
    end
    @assert isa(gridsz, Array{Int, 1}) && size(gridsz) == (2,)
    
    gridsz = (gridsz == nothing) ? abutils.subplot_gridsize(num) : gridsz

    
    out = zeros((gridsz .* Ms_sz)...)
    for ii in 1:gridsz[1], jj in 1:gridsz[2]
        i = (ii-1) * gridsz[2] + jj
        ix_xs = ((gridsz[1] -ii )*Ms_sz[1] + 1) : ((gridsz[1] - ii +1)*Ms_sz[1])
        ix_ys = ((jj-1)*Ms_sz[2] + 1) : (jj*Ms_sz[2])
        out[ix_xs, ix_ys] = Ms[i]   
    end
    return copy(out)  # not sure if the copy is necessary. Freq is for PyPlot.
end


function plot_2dtile_imshow(forward_network; nsmp_dim=8, scale=2., xrng=nothing, yrng=nothing,
        im_shp=(28,28), transpose=false, cmap="viridis", ax=nothing)
    #=
    :param forward_network: decoder for VAE. Should accept 2 x n matrices.
    :param nsmp_dim: number of (equally spaced) samples in both x and y dims.
    :param scale: scale of grid on latent space (will be [-scale, scale] in both dims)
    :param im_shp: size/shape of each image. Default = (28,28) for MNIST.
    :param transpose: images need transposing before displaying (e.g. if in row major format)
    :param cmap: colormap -- default "viridis", suggest also "binary_r" for greyscale.
    =#
    @assert (xrng == nothing && yrng == nothing) || (xrng != nothing && yrng != nothing) "If " *
        "either xrng or yrng specified, both must be specified."
    xrng != nothing && scale != 2. && @warn "Ignoring scale specification, as rng given."
    xrng = xrng == nothing ? [-scale, scale] : xrng
    yrng = yrng == nothing ? [-scale, scale] : yrng
    
    zs = _gen_grid(xrng, yrng, nsmp_dim=nsmp_dim)
    tr = (transpose == true ? Base.transpose : x -> x)
    ims = [reshape(Tracker.data(forward_network(zs[ii,:])), im_shp) for ii in 1:size(zs,1)]
    imtiled = tr(_tile_image_grid(ims))
    ax = ax == nothing ? gca() : ax
    ax[:imshow](imtiled, cmap=cmap)
    Lx = max(all([typeof(x) <: Int for x in xrng]) ? diff(xrng)[1]+1 : 3, 3)
    Ly = max(all([typeof(x) <: Int for x in yrng]) ? diff(yrng)[1]+1 : 3, 3)
    ax[:set_xticks](arange(1,stop=im_shp[1]*nsmp_dim, length=Lx));  ax[:set_xticklabels](arange(xrng[1],stop=xrng[2], length=Lx))
    ax[:set_yticks](arange(1,stop=im_shp[2]*nsmp_dim, length=Ly));  ax[:set_yticklabels](arange(yrng[2],stop=yrng[1], length=Ly))
    return ax
end



 #=================================================================================
                                  Axis manipulation
 ==================================================================================#

function ax_lim_one_side(ax, xy; limstart=nothing, limend=nothing, type="constant")
    #=
    Ported from pyalexutil. Manipulates one current axis limit without changing others.
    The interfaces `x_lim_one_side`, `y_lim_one_side` use this functionality and may
    be preferred to this function as they read better.
    :param ax           - axis object to manipulate
    :param xy           - either "x", "y", the dimension of the axis to manipulate
    :param limstart     - the argument for changing the start number (nothing = no change)
    :param limend       - the     "     "     "      "   end    " 
    :param type         - what to do with the start/end axis: "constant" specifies
                          overriding current value with limstart/limend, "multiply"/"*"
                          and "add"/"+" also accepted which multiply/add curr. number.
    =#
    if xy == "x"
        lims = ax[:get_xlim]()
    else
        lims = ax[:get_ylim]()
    end
    lims = collect(lims)  # make mutable (is tuple typed)
    
    if type == "m" || type == "multiply" || type == "*"
        f = *
    elseif type == "a" || type == "add" || type == "+"
        f = +
    elseif type == "c" || type == "constant"
        f = (x, y) -> y
    else
        throw("Unexpected limtype (expecting 'constant', 'add', 'multiply')")
    end
    
    if limstart != nothing; lims[1] = f(lims[1], limstart); end
    if limend != nothing; lims[2] = f(lims[2], limend); end
    
    if xy == "x"
        ax[:set_xlim](lims)
    else
        ax[:set_ylim](lims)
    end
end


# convenience wrappers for clean code
function x_lim_one_side(ax; s=nothing, e=nothing, type="constant")
    #=
    Change either the start or end of the specified x-axis. See `ax_lim_one_side`.
    =#
    ax_lim_one_side(ax, "x", limstart=s, limend=e, type=type)
end


function y_lim_one_side(ax; s=nothing, e=nothing, type="constant")
    #=
    Change either the start or end of the specified y-axis. See `ax_lim_one_side`.
    =#
    ax_lim_one_side(ax, "y", limstart=s, limend=e, type=type)
end


x_lim_start_zero(ax) = x_lim_one_side(ax; s=0.)
y_lim_start_zero(ax) = y_lim_one_side(ax; s=0.)



end
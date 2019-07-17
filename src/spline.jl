
# ported from my MATLAB utils
struct BSpline{T}
    k               # degree of each piecewise polynomial
    t::Vector{T}    # knots; breakpoints
    bdknots         # number of knots at boundaries
end

Base.length(s::BSpline) = length(s.t)

"""
  bspline(k, t)
create bspline object of degree k, and knots -- possibly coin-cident.
Constructor will assume discontinuity at boundaries.

  bspline(k, t, 'boundaries', bb)
as above, but constructor will adapt continuity at the
boundaries:

* bb = k     => discontinuity
* bb = k - a => discontinuity in a'th derivative.

with methods
  * basis_eval(bspline, x)

The resulting basis can then be used with your favourite loss function as reqd.
Unfortunately I have not taken advantage of any sparsity properties (e.g. the
bounded support of the spline basis), because I have better things to do :D.
The Interpolations.jl package does indeed do this, but I can't find an easy way
to extract the basis from this for use in e.g. penalised fits.

follows de Boor - A Practical Guide to Splines (1978 / 2001).
"""
function BSpline(k::Int, t::AbstractVector{T}; bdknots=k) where T <: Real
    @argcheck k > 0
    @argcheck bdknots <= k
    @assert issorted(t)
    @argcheck length(unique(t)) > 1

    if bdknots > 0
        # add additional boundary knots to beginning of vector (if reqd)
        first_not_bd = findfirst(t .!= t[1]) - 1
        numaddl = max(0, bdknots - first_not_bd);
        t       = vcat(repeat(t[1:1], numaddl), t);

        # add additional boundary knots to end of vector (if reqd)
        last_not_bd = findlast(t .!= t[end]) + 1
        numaddl = max(0, bdknots - (length(t) - last_not_bd + 1));
        t       = vcat(t, repeat(t[end:end], numaddl));
    end
    return BSpline{eltype(t)}(k, t, bdknots)
end

function basis_eval(obj::BSpline, x::AbstractVector{T}) where T <: Real
    """
        basis_eval(obj::BSpline, x [,nderiv])
    evaluate the given basis at positions x = [x1, x2, ..., xm].
    Uses recurrence B_{jk} = w_{jk} B_{j,degk-1} + (1-w_{jk}) B_{j+1,degk-1}.

    Derivatives can be calculated instead by specifying the order in the third
    argument (i.e. 0 = fn, 1 = 1st deriv, 2 = 2nd deriv, ...).

    This function owes a lot to James Ramsay's implementation in the FDA
    toolbox.
    """
    N               = length(x);
    # if do_slow === nothing
    #     do_slow = N < 100 ? true : false
    # end

    # recurrence assumes nondecreasing input sequence. Ensure this
    # is so. (cf James Ramsay)
    if minimum(diff(x)) < 0
        reorder_perm = sortperm(x)
        x = x[reorder_perm]
        reordered = true;
    else
        reordered = false;
    end
    nt          = length(obj);

    nbasis      = nt - obj.bdknots;
    degk        = obj.k;
    knotslower  = obj.t[1:nbasis];
    index       = sortperm(vcat(knotslower, x));
    pointer     = findall(index .> nbasis) - (1:N);
    left        = max.(pointer, degk);
    b           = repeat(hcat(1, zeros(1, degk)), N, 1);

    # recursion directly from de Boor (and James Ramsay).
    for jj = 1:degk
        saved = zeros(N, 1);
        for r = 1:jj
            leftpr    = left .+ r;
            tr        = obj.t[leftpr] - x;
            tl        = x - obj.t[leftpr .- jj];

            term      = b[1:N, r]./(tr+tl);
            b[1:N,r]  = saved + tr.*term;
            saved     = tl.*term;
        end
        b[1:N, jj+1]    = saved;
    end

    # now each end column in b corresponds to B_{i-degk+1,degk}, ..., B_{ik}
    # for each i, where i corresponds to the knot to the left of
    # the interval in which each x falls. So to place them all in
    # the same (comparable matrix, we need to shift the columns
    # about.

    # James Ramsay has a cute MATALAB hack to restructure the matrix quickly
    # (benchmarking in MATLAB demonstrates substantial improvement). However the
    # naive (and simple) version here appears to perform just as quickly in julia.
    # if do_slow
    B = zeros(N, nt - obj.bdknots+1);
    for nn = 1:N
        B[nn, (left[nn]-degk+1):left[nn]+1] = b[nn,:];
    end

    # if reordered elements if they were not monotonically increasing, put back in original
    reordered && (B[reorder_perm,:] = B);
    return B
end


function bsplineM(x::AbstractVector{T}, breaks::AbstractVector, norder::Int=4,
        nderiv::Int=0, sparsewrd::Bool=false) where T
    """
        bsplineM(x, breaks, norder, nderiv, sparsewrd)

    Original James Ramsay function ported from MATLAB upon which the `basis_eval`
    function was based. This has greater functionality (does derivatives too).
    I've changed the nature of the sparse return value but o.w. its ~same.

    Computes values or derivatives of B-spline basis functions.

    Arguments:
    X         ... Argument values for which function values are computed
    BREAKS    ... Increasing knot sequence spanning argument range
    NORDER    ... Order of B-spline (one greater than degree) max = 19
                 Default 4.
    NDERIV    ... Order of derivative required, default 0.
    SPARSEWRD ... if true, return in (a non-standard) sparse form

    Return:
    BSPLINEMAT ... length(X) times number of basis functions matrix
                  of Bspline values

    last modified 11 February 2015

    ported from James Ramsay (et al.?)'s FDA toolbox.
    """

    n = length(x);
    nbreaks = length(breaks);

    @assert nbreaks >= 2 "Number of knots less than 2."
    @assert issorted(breaks) "breaks are not strictly increasing"
    @assert norder >= 1 "Order of basis less than one."
    @argcheck nderiv >= 0
    @assert nderiv < norder "NDERIV cannot be as large as order of B-spline."

    if minimum(diff(x)) < 0
        reorder_perm = sortperm(x)
        x = x[reorder_perm]
        reorder_reverse = sortperm(reorder_perm)
        sortwrd = true
    else
        sortwrd = false
    end

    if x[1] - breaks[1] < -1e-10 || x[n] - breaks[nbreaks] > 1e-10
        display([x[1], x[n], breaks[1], breaks[nbreaks]])
        error("Argument values out of range.")
    end

    if norder == 1
        @argcheck !sparsewrd "norder=1 not yet implemented in sparse form"
        bsplinemat = zeros(T, n, nbreaks-1);
        for ibreaks = 2:nbreaks
            ind = (x >= breaks[ibreaks-1]) & (x <= breaks[ibreaks]);
            bsplinemat[ind, ibreaks-1] = 1;
        end
        return bsplinemat
    end

    # set some abbreviations

    k   = norder;         # order of splines
    km1 = k-1;
    nb  = length(breaks); # number of break points
    nx  = length(x);      # number of argument values
    nd  = nderiv+1;       # ND is order of derivative plus one
    ns  = nb - 2 + k;     # number of splines to compute
    if ns < 1
       @warn "There are no B-splines for the given input."
       return zeros(T, 0, 0)
    end

    #  augment break sequence to get knots by adding a K-fold knot at each end.
    knots  = vcat(repeat(breaks[1:1], km1), breaks, repeat(breaks[nb:nb], km1));
    nbasis = length(knots) - k;

    #  For each  i , determine  left(i)  so that  K <= left(i) < nbasis+1 , and,
    #    within that restriction,
    #        knots(left(i)) <= pts(i) < knots(left(i)+1) .

    knotslower      = knots[1:nbasis];
    index           = sortperm(vcat(knotslower, x));
    pointer         = findall(index .> nbasis) - (1:length(x));
    left            = max.(pointer, k);

    # compute bspline values and derivatives, if needed:
    # (run the recurrence simultaneously for all  x(i).)
    # =====================================================

    # initialize the  b  array.
    b    = repeat(hcat(1, zeros(1, km1)), nd*nx, 1)
    nxs  = nd*(1:nx);

    # First, bring it up to the intended level:
    for j=1:k-nd
       saved = zeros(T, nx, 1);
       for r=1:j
          leftpr   = left .+ r;
          tr       = knots[leftpr] - x;
          tl       = x - knots[leftpr .- j];
          term     = b[nxs, r] ./ (tr + tl);
          b[nxs,r] = saved + tr .* term;
          saved    = tl .* term;
       end
       b[nxs, j+1]  = saved;
    end

    # save the B-spline values in successive blocks in  b .
    for jj=1:nd-1
       j = k - nd + jj;
       saved = zeros(T, nx, 1);
       nxn   = nxs .- 1;
       for r=1:j
          leftpr   = left .+ r;
          tr       = knots[leftpr] - x;
          tl       = x - knots[leftpr .- j];
          term     = b[nxs, r] ./ (tr + tl);
          b[nxn,r] = saved + tr .* term;
          saved    = tl .* term;
       end
       b[nxn,j+1]  = saved;
       nxs = nxn;
    end

    # now use the fact that derivative values can be obtained by differencing:
    for jj=nd-1:-1:1
       j = k - jj;
       nxs = (repeat(jj:nd-1, 1, nx) + repeat(nxs', nd-jj, 1))[:];
       for r=j:-1:1
          leftpr     = left .+ r;
          temp       = repeat(knots[leftpr] - knots[leftpr .- j], nd-jj)' / j;
          b[nxs, r]   = -b[nxs, r] ./ temp[:];
          b[nxs, r+1] =  b[nxs, r+1] - b[nxs, r];
       end
    end

    # Finally, zero out all rows of  b  corresponding to x outside the basic
    # interval,  [breaks[1] .. breaks[nb]] .
    index = findall((x .< breaks[1]) .| (x .> breaks[nb]));
    if length(index) > 0
       temp = repeat(1-nd:0, 1, length(index)) + nd*repeat(index', nd, 1);
       b[temp[:], :] .= 0
    end

    # If sparse return value, return now.
    if sparsewrd
        elements = b[nd*(1:nx),:];
        first_columns = left .- (km1)
        sortwrd && (elements = elements[reorder_reverse,:])
        sortwrd && (first_columns = first_columns[reorder_reverse,:])

        return elements, first_columns
    end

    # set up output matrix bsplinemat
    width = max(ns,nbasis) + km1 + km1;
    cc    = zeros(nx*width, 1);
    index = repeat(1-nx:0, 1, k) + nx*(repeat(left, 1, k) +
                                       repeat((-km1:0)', nx, 1));

    cc[index] = b[nd*(1:nx),:]
    # (This uses the fact that, for a column vector  v  and a matrix  A ,
    #  v(A)(i,j)=v(A(i,j)), all i,j.)  [#AB: no idea what v(A) etc mean...]
    bsplinemat = reshape(cc[repeat(1-nx:0, 1, ns) +
                            nx*repeat((1:ns)', nx, 1)], nx, ns);

    sortwrd && (bsplinemat = bsplinemat[reorder_reverse, :])

    return bsplinemat

end

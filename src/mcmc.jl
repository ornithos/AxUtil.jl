module mcmc

using LinearAlgebra
using Flux
using Flux: Tracker, params


function mala(logdensity, h, M, iters, init_proposal_mean; δ_trunc=Inf)

    function log_llh_and_proposal(z)
        par_z = param(z)
        logp_u = logdensity(par_z)
        Tracker.back!(logp_u)
        grad = Tracker.grad(par_z)

        #  ----- Truncated MALA (T-MALA) ----
        #  ----- (aka norm clipping!) -------
        if δ_trunc < Inf
            grad_norm = sum(x -> x^2, grad)
            if grad_norm > δ_trunc
                grad *= δ_trunc / grad_norm
            end
        end
        return Tracker.data(logp_u), z + 0.5*h*M*grad, grad   # prev (unnorm) logp, gradient step, gradient
    end

    # set up
    d = length(init_proposal_mean)
    ztrace = Array{Float64, 2}(undef, d, iters)
    cholΣ = Matrix(cholesky(h*M).L)   # `Matrix` keeps Flux happy for the time being. For d < 20, it really makes no difference.

    function gauss_logpdf(z)
        u = cholΣ \ z
        exponent = -0.5*sum(x->x*x, u)
#         lognormconst = -0.5*logdet(2π * h * M)
        out = exponent #+ lognormconst
        return out
    end
    gauss_logpdf(z, μ) = gauss_logpdf(z - μ)

    # initialise
    accepted = 0
    z = init_proposal_mean
    cur_u_logp = -Inf
    proposal_mean = init_proposal_mean
    prev_proposal_mean = init_proposal_mean

    # Markov Chain loop
    for i = 1:iters
        # transition r.v.
        ϵ_prop = randn(d)
        prop_transition_logp = -0.5*sum(x->x*x, ϵ_prop)

        # deterministic proposal
        z_prop = proposal_mean + cholΣ * ϵ_prop

        # calculate logdensity(z_prop) and accompanying new proposal mean
        prop_u_logp, ν_proposal_mean, _ = log_llh_and_proposal(z_prop)

        # metropolis-hastings
        r = prop_u_logp - cur_u_logp +
                gauss_logpdf(z, ν_proposal_mean) -
                prop_transition_logp   |> exp

        # if <accept>
        if rand() < r
            z = z_prop                                                       # update z
            prev_proposal_mean = proposal_mean                               # store prev proposal mean
            cur_u_logp, proposal_mean = prop_u_logp, ν_proposal_mean
            accepted += 1
        end

        ztrace[:,i] = z      # store z regardless
    end

    return ztrace, accepted/iters
end

end

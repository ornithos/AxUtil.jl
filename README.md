# AxUtil
My utilities package for julia. This is a collection of useful functions that are shared across projects. None of this work functions as a useful standalone project, and hence does not follow the julia convention of minimal projects which perform one thing well.

This is very much a WIP, and usually includes things that are specific to whatever I'm currently developing. It is also includes ports of things from `pyalexutil`. Briefly, it currently has the following modules:

* `AMIS`: utilities for Adaptive Mixture Importance Sampling, most \*importantly\* including a _weighted_ GMM EM algo.
* `Arr`: array functions, operations that I find useful which are not part of `Base`. This is at present kind of trivial.
* `Flux`: some additions to Flux supported operations for Automatic Differentiation that I have used. The more useful ones (`inv`, `ldiv`, `rdiv`) have been moved into Flux itself.
* `Math`: a few softmax-y logsumexp-y type functions, plus numerical gradient checking.
* `MCDiagnostic`: Some Monte Carlo diagnostics, including a few different Effective Sample Size functions and $\hat{R}$.
* `MCMC`: Currently just an implementation of (Truncated) Metropolis Adjusted Langevin Algorithm (T-MALA).
* `Misc`: Misc functions I miss from python/MATLAB, and faster version of `countmap`.
* `Plot`: A few useful plotting utilities, including pairplot, axis operations and image tiling.
* `Random`: Mostly implementations of Multinomial sampling that are currently unavailable in Distributions.jl. Not entirely clear if there is a wider demand for this.

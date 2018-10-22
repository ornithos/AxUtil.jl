# AxUtil
My utilities package for julia. This is a collection of useful functions that are shared across projects. None of this work functions as a useful standalone project, and hence does not follow the julia convention of minimal projects which perform one thing well.

This is very much a WIP, and usually includes things that are specific to whatever I'm currently developing. It is also includes ports of things from `pyalexutil`. Briefly, it currently has the following modules:

* `AMIS`: [*deprecated, to move out to seq inference pkg*] Adaptive Mixture Importance Sampling, most \*importantly\* including a _weighted_ GMM EM algo.
* `Arr`: array functions, operations that I find useful which are not part of `Base`. This is at present kind of trivial.
* `dpmeans`: A dirichlet process kmeans algorithm, following Kulis & Jordan, 2012. Sadly inherently sequential, so threading is basically nonexistent.
* `Flux`: some additions to Flux supported operations for Automatic Differentiation that I have used. The more useful ones (`inv`, `ldiv`, `rdiv`) have been moved into Flux itself.
* `gmm`: my own version of Gaussian Mixture Models. Various choices in `Distributions.jl` make these a little slow, and furthermore there's no generic MLE fitting routine. This includes a (weighted) EM algorithm for fitting, plus some of the usual utilities for Distributions, such as `rand`, `logpdf`, `show` etc.
* `Logging`: Only one `<: AbstractLogger` at present: this adds a timestamp to `SimpleLogger` and shifts the output around a little. The file `TimeStampLog.sublime-syntax` provides useful highlighting of the log if using Sublime Text 3.
* `Math`: a few softmax-y logsumexp-y type functions, plus numerical gradient checking.
* `MCDiagnostic`: Some Monte Carlo diagnostics, including a few different Effective Sample Size functions and $\hat{R}$.
* `MCMC`: Currently just an implementation of (Truncated) Metropolis Adjusted Langevin Algorithm (T-MALA).
* `Misc`: Misc functions I miss from python/MATLAB, a faster version of `countmap` for `Int`s, and `repelem`-like behaviour for `repeat`.
* `Plot`: A few useful plotting utilities, including pairplot, axis operations and image tiling.
* `Random`: Mostly implementations of Multinomial sampling that are currently unavailable in Distributions.jl. [**Update**: I've subsequently noticed this is just called `Categorical` in `Distributions.jl`, however the linear scan version in particular appears to be substantially faster still than their version.
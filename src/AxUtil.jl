module AxUtil

using Requires

include("arr.jl")
include("dpmeans.jl")
include("logging.jl")
include("math.jl")
include("flux.jl")  # must come after math
include("mcdiag.jl")
include("mcmc.jl")
include("misc.jl")
include("random.jl")

@init @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" include("cuda.jl")

end # module

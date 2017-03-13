"""
  import CMA
  reload("CMA")
  const A = 10
  rastrigin(x) = A*length(x) + sum(x.^2 .- A*cos.(2π*x))
  CMA.cmaes(rastrigin, -ones(2), ones(2), restarts=1)
"""
module CMA
using PyCall
@pyimport cma
function cmaes(f, l::Array, u::Array; restarts=0, o...)
    str =  "np.array($l)+np.random.rand($(length(l)))*np.array($(u-l))"
    options = Dict("boundary_handling"=>"BoundPenalty","bounds"=>[l, u])
    res = cma.fmin(x->f(x), str, 1/4*minimum(u-l),
    options, bipop=true, o...);
    res[1],res[2]
end
function cmaes(f, x0::Array, σ0::Float64=1; restarts=0, o...)
    res = cma.fmin(x->f(x), x0, σ0, o...)
    res[1],res[2]
end
end

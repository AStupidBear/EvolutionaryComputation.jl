"""
    addprocs(1)
    @everywhere begin
    reload("PNCS")
    rastrigin(x) = 10*length(x) + sum(x.^2 .- 10*cos.(2π*x))
    end
    @time PNCS.fmin(rastrigin, -ones(2), ones(2); T=100)
"""

module PNCS
export Particle, init_pop, optimize
const r = 0.99
const λrange= 0.1
const epoch = 100
type Particle{T,N}
    x::Array{T,N} # position
    v::T # value
    vn::T # normalized value
    x′::Array{T,N} # trial position
    v′::T  # trial value
    vn′::T # normalized tial value
    σ::Array{T,N} # diagnol elements of Σ (sampling covariance matrix)
    Corr::T # correlation value
    Corr′::T # tiral correlation value
    flag::T # successful trial in an epoch
    λ::T # λ>0 is a parameter to balance exploration and exploitation.
end
Base.show(p::Particle) = println("x=",p.x," f(x)=",p.v)
function init_pop(f,l,u; N=10)
    g = [Particle(l+(u-l).*rand(size(l)),Inf,Inf,
        l+(u-l).*rand(size(l)),Inf,Inf,
        (u-l)/N,Inf,Inf,0.0,1.0) for i=1:N]
    for p in g
        p.v = f(p.x)
    end
    g
end
function BD(p, q)
    Corr = Corr′ = 0
    @inbounds for d in eachindex(p.x)
        Δx = p.x[d] - q.x[d]
        Δx′ = p.x′[d] - q.x[d]
        σ2 = p.σ[d]^2; σ2′ = q.σ[d]^2
        c = (σ2+σ2′)/2
        Corr += 1/8*Δx^2/c + 1/2*(log(c) - 0.5*(log(σ2) + log(σ2′)))
        Corr′+= 1/8*Δx′^2/c + 1/2*(log(c) - 0.5*(log(σ2) + log(σ2′)))
    end
    Corr,Corr′
end
function inbounds!(x, l, u)
  @inbounds for i in eachindex(x)
    if x[i] < l[i]
      x[i] = -x[i] + 2*l[i]
    elseif x[i] > u[i]
      x[i] = -x[i] +2*u[i]
    end
  end
end

function search!(g, l, u)
    for p in g
        p.x′ .= p.x .+ p.σ.*randn(size(p.x))
        inbounds!(p.x, l, u)
    end
end

function fitness_values!(f, g)
    scores = pmap(p->f(p.x′), g)
    for i in eachindex(scores)
      g[i].v′ = scores[i]
    end
end

function normalize_fitness_values!(g)
    v_min = minimum(min(p.v,p.v′) for p in g)
    for p in g
        p.vn = p.v - v_min;
        p.vn′ = p.v′ - v_min;
        p.vn,p.vn′ = p.vn/(p.vn+p.vn′),p.vn′/(p.vn+p.vn′)
    end
end
function correlation_values!(g)
    for p in g
        p.Corr = p.Corr′ = Inf
        for q in g
            c,c′ = BD(p,q)
            if c < p.Corr
                p.Corr = c
            end
            if c′ < p.Corr′
                p.Corr′ = c′
            end
        end
    end
end
function normalize_correlation_values!(g)
    for p in g
        p.Corr,p.Corr′ = p.Corr/(p.Corr+p.Corr′+1e-20),
        p.Corr′/(p.Corr+p.Corr′+1e-20)
    end
end
function selection!(g)
    for p in g
        if p.λ*p.Corr′ > p.vn′
            p.x = copy(p.x′)
            p.v = p.v′
            p.flag += 1.0
        end
    end
end

function update_parameters!(g,t,T)
    for p in g
        p.λ = 1 + λrange*(1-t/T)*randn()
    end
    if mod(t, epoch) == 0
        for p in g
            if p.flag/epoch > 0.2
                p.σ /= r;
            elseif p.flag/epoch < 0.2
                p.σ *= r;
            end
            p.flag = 0.0
        end
    end
end
function best!(p0,g)
    for p in g
            if p.v < p0.v
            p0.x = copy(p.x)
            p0.v = p.v
        end
    end
end
function optimize!(f, g, l, u; T=10, disp=false)
    p0 = deepcopy(g[1])
    for t = 0:T
        search!(g, l, u)
        fitness_values!(f, g)
        normalize_fitness_values!(g)
        correlation_values!(g)
        normalize_correlation_values!(g)
        selection!(g)
        best!(p0, g)
        update_parameters!(g, t, T)
        if disp == true && mod(t, 100) == 0
            show(p0)
        end
    end
    p0.x,p0.v
end

export fmin
function fmin(f, l, u; N = 50, T = 100, disp = true)
  g = init_pop(f, l, u; N = N)
  x0, y0 = optimize!(f, g, l, u; T = T, disp = disp)
end
end

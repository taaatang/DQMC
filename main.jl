using LinearAlgebra
include("helper.jl")
## parameters
Nx = 6
Ny = 6
periodic = true
β = 1.0/0.06
L = Int64(ceil(β/0.1))
U = 2.0
nwarm = 200
nmeas = 800
N = Nx * Ny
dτ = β/L
λ = acosh(exp(dτ * U / 2.0))
## Initialization
K = constructK(N, periodic)
expK = exp(-dτ * K)
invexpK = exp(dτ * K)
s = rand([-1, 1], (N, L))
σu = 1.0
σd = -1.0
expVu = exp.(σu * λ * s)
expVd = exp.(σd * λ * s) 
lb = 10
lw= 5
gu = Green(expK, expVu, 1, lb)
gd = Green(expK, expVd, 1, lb)
## MC iteration
nu = 0.0
nd = 0.0
doubleocc = 0.0
@time for sweep = 1:(nwarm + nmeas)
    for l = 1:L
        if l % lw == 0
            gu = Green(expK, expVu, l, lb)
            gd = Green(expK, expVd, l, lb) 
        end
        for i = 1:N
            fu = factor(s[i,l], λ, σu)
            fd = factor(s[i,l], λ, σd)
            # ws'/ws
            r = (1.0 + (1.0 - gu[i,i]) * fu) * (1.0 + (1.0 - gd[i,i]) * fd)
            if rand() < r
                updateG!(gu, i, fu)
                updateG!(gd, i, fd) 
                s[i,l] = -s[i,l]
                expVu[i,l] = exp(σu * λ * s[i,l])
                expVd[i,l] = exp(σd * λ * s[i,l])
            end
        end
        wrapG!(gu, l, expK, expVu)
        wrapG!(gd, l, expK, expVd)
    end
    gu = Green(expK, expVu, 1, lb)
    gd = Green(expK, expVd, 1, lb)
    # measurement
    if sweep > nwarm
        for i = 1:N
            nui = 1 - gu[i,i]
            ndi = 1 - gd[i,i]
            nu += nui
            nd += ndi
            doubleocc += nui * ndi
        end
    end
end

nu /= N * nmeas
nd /= N * nmeas
doubleocc /= N * nmeas
mz² = nu + nd - 2.0 * doubleocc

println(nu)
println(nd)
println(doubleocc)
println(mz²)
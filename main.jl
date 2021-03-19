using LinearAlgebra
include("helper.jl")
## parameters
Nx = 6
Ny = 6
periodic = true
β = 5.0
L = 50
U = 4.0
nwarm = 200
nmeas = 800
N = Nx * Ny
dτ = β/L
λ = acosh(exp(dτ * U / 2.0))
##
K = constructK(N, periodic)
expK = exp(-dτ * K)
invexpK = exp(dτ * K)
s = rand([-1, 1], (N, L))
σu = 1.0
σd = -1.0
expVu = exp.(σu * λ * s)
expVd = exp.(σd * λ * s) 
gu = Green(expK, expVu)
gd = Green(expK, expVd)
## MC iteration
nu = 0.0
nd = 0.0
doubleocc = 0.0
for sweep = 1:(nwarm + nmeas)
    for l = 1:L
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
    gu = Green(expK, expVu)
    gd = Green(expK, expVd)
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
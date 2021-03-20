using LinearAlgebra

function constructK(N, periodic=true)
    K = zeros(Float64, (N, N))
    for y in 1:Ny
        for x in 1:Nx
            i = x + (y -1) * Nx
            for dy in [-1, 1]
                for dx in [-1, 1]
                    if periodic
                        x1 = x + dx
                        y1 = y + dy
                        if x1 < 1
                            x1 = x1 % Nx + Nx
                        end
                        if x1 > Nx
                            x1 = x1 % Nx
                        end
                        if y1 < 1
                            y1 = y1 % Ny + Ny
                        end
                        if y1 > Ny
                            y1 = y1 % Ny
                        end
                        j = x1  + (y1 - 1) * Nx
                        K[i,j] = 1.0
                    else
                        j = (x + dx) + (y + dy -1) * Nx
                        if j ≥ 1 && j≤ N
                            K[i,j] = 1.0
                        end
                    end
                end
            end
        end
    end
    return K
end

# Green's function. Naive way
function Green(expK, expV)
    N, L = size(expV)
    g = I
    for i in 1:L  
        g = expK * Diagonal(expV[:, i]) * g
    end
    return inv(I + g)
end

# TODO: optimization, don't need to recalculate B(i+d-1)*...*B(i) each time
"""
@brief Improved Green's function using pivoted QR
@param l0: recalculate Green's function after l0 wraps, g = inv(I + ...B(l0+1) * B(l0))
@param d: group #d B matrixes togother for qr factorization
"""
function Green(expK, expV, l0, d = 10)
    N, L = size(expV)
    Q = I
    D = I
    T = I
    for l in l0 : d : (l0 + L)
        B = I
        for i in l : min(l + d -1, l0 + L)
            idx = (i - 1) % L + 1
            B = expK * Diagonal(expV[:, idx]) * B
        end
        C = B * Q * D
        F = qr(C,Val(true))
        Q = F.Q
        D = Diagonal(F.R)
        T = inv(D) * F.R * F.P' * T
    end
    Db = zeros(N)
    Ds = zeros(N)
    for i in 1 : N
        if abs(D[i, i]) > 1.0
            Db[i] = D[i, i]
            Ds[i] = 1.0
        else
            Db[i] = 1.0
            Ds[i] = D[i, i]
        end
    end
    return inv(Diagonal(1.0./Db) * Q' + Diagonal(Ds) * T) * (Diagonal(1.0./Db) * Q')
end

function factor(s, λ, σ)
    return exp(-2.0 * σ * λ * s) - 1.0
end

function updateG!(g, i, f)
    ui = g[:,i]
    ui[i] -= 1.0
    g[:,:] += ((f/(1.0 + (1.0 - g[i,i]) * f)) * ui) * g[i,:]'
end

function wrapG!(g, l, expK, expV)
    Bl = expK * Diagonal(expV[:, l])
    g[:,:] = Bl * g * inv(Bl)
end
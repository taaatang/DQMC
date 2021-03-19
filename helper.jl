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

## Green's function
function Green(expK, expV)
    N, L = size(expV)
    g = I
    for i in 1:L  
        g = expK * Diagonal(expV[:, i]) * g
    end
    return inv(I + g)
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
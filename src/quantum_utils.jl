#############
#  GENERAL  #
#############

#  Pauli matrices
σI(; type=ComplexF64) = Matrix{type}([1 0; 0 1])
σX(; type=ComplexF64) = Matrix{type}([0 1; 1 0])
σY(; type=ComplexF64) = Matrix{type}([0 -im; im 0])
σZ(; type=ComplexF64) = Matrix{type}([1 0; 0 -1])

# projector onto a direction in the Bloch sphere
function qubit_proj(v::Vector{T}; mes::Bool=false, type=Complex{T}) where {T <: Number}
    if mes
        return [
            (σI(; type=type) - v[1] * σX(; type=type) - v[2] * σY(; type=type) - v[3] * σZ(; type=type)) / 2;;;
            (σI(; type=type) + v[1] * σX(; type=type) + v[2] * σY(; type=type) + v[3] * σZ(; type=type)) / 2
        ]
    else
        return (σI(; type=type) + v[1] * σX(; type=type) + v[2] * σY(; type=type) + v[3] * σZ(; type=type)) / 2
    end
end

function qubit_proj(v::Tuple{T, T, T}) where {T <: Number}
    [
        (σI() - v[1] * σX() - v[2] * σY() - v[3] * σZ()) / 2;;;
        (σI() + v[1] * σX() + v[2] * σY() + v[3] * σZ()) / 2
    ]
end

############
#  STATES  #
############

function rho_singlet(; type=Float64)
    psi = [zero(type), one(type), -one(type), zero(type)]
    return psi * psi' / type(2)
end

function rho_GHZ(N::Int; type=Float64)
    psi = zeros(type, 2^N)
    psi[1] = one(type)
    psi[2^N] = one(type)
    return psi * psi' / type(2)
end

function rho_W(N::Int; type=Float64)
    psi = zeros(type, 2^N)
    for n in 1:N
        psi[2^(n-1)+1] = one(type)
    end
    return psi * psi' / type(N)
end

function ketbra(phi::VecOrMat{T}) where {T <: Number}
    if size(phi, 2) > 1
        if size(phi, 1) == 1
            phi = adjoint(phi)
        else
            throw(DimensionMismatch(string(size(phi))))
        end
    end
    phi * phi'
end

##################
#  MEASUREMENTS  #
##################

# create a set of qubit measurements from an array of Bloch coordinates
function qubit_mes(v::Matrix{T}; type=Complex{T}) where {T <: Number}
    res = zeros(type, 2, 2, 2, size(v, 1))
    for i in 1:size(v, 1)
        res[:, :, :, i] = qubit_proj(v[i, :]; mes=true, type=type)
    end
    return res
end

# platonic solids
cube_vec() = [1 1 1; 1 -1 -1; -1 1 -1; -1 -1 1] / sqrt(3)
octahedron_vec() = [1 0 0; 0 1 0; 0 0 1]
function icosahedron_vec(; type=Float64)
    φ = (1 + sqrt(type(5))) / 2
    return [0 1 φ; 0 1 -φ; 1 φ 0; 1 -φ 0; φ 0 1; φ 0 -1] / sqrt(2 + φ)
end
function dodecahedron_vec(; type=Float64)
    φ = (1 + sqrt(type(5))) / 2
    return [
        1 1 1
        1 -1 -1
        -1 1 -1
        -1 -1 1
        0 1/φ φ
        0 1/φ -φ
        1/φ φ 0
        1/φ -φ 0
        φ 0 1/φ
        φ 0 -1/φ
    ] / sqrt(type(3))
end

# measurements from arXiv:1609.05011 (regular polyhedron in the XY plane)
function polygonXY_vec(m::Int; type=Float64)
    if type <: AbstractFloat
        return collect(hcat([[cos((x - 1) * type(pi) / m), sin((x - 1) * type(pi) / m), zero(type)] for x in 1:m]...)')
    elseif type <: Real
        return collect(
            hcat([[type(cos((x - 1) * big(pi) / m)), type(sin((x - 1) * big(pi) / m)), zero(type)] for x in 1:m]...)',
        )
    else
        return vertices = collect(
            hcat(
                [
                    [(type(E(2m, x - 1)) + type(E(2m, 1 - x))) // 2, -im * (type(E(2m, x - 1)) - type(E(2m, 1 - x))) // 2, 0] for x in 1:m
                ]...,
            )',
        )
    end
end

# measurements from arXiv:1609.06114 (rotated regular polyhedron)
function HQVNB17_vec(n::Int; type=Float64)
    m = isodd(n) ? n^2 : n^2 - n + 1
    res = zeros(type, m, 3)
    if iseven(n)
        tabv = [
            (
                cos(i1 * type(pi) / n) * cos(i2 * type(pi) / n),
                sin(i1 * type(pi) / n) * cos(i2 * type(pi) / n),
                sin(i2 * type(pi) / n),
            ) for i1 in 0:n-1, i2 in 0:n-1 if i2 != n ÷ 2
        ][:]
        push!(tabv, (0.0, 0.0, 1.0))
    else
        tabv = [
            (
                cos(i1 * type(pi) / n) * cos(i2 * type(pi) / n),
                sin(i1 * type(pi) / n) * cos(i2 * type(pi) / n),
                sin(i2 * type(pi) / n),
            ) for i1 in 0:n-1, i2 in 0:n-1
        ][:]
    end
    for i in 1:m
        res[i, 1] = tabv[i][1]
        res[i, 2] = tabv[i][2]
        res[i, 3] = tabv[i][3]
    end
    return res
end

##############
# CONVERSION #
##############

# convert a mx3 Bloch matrix into a mxm correlation matrix
function correlation_matrix(vec::AbstractMatrix{T}) where {T <: Number}
    @assert size(vec, 2) == 3
    -vec * vec'
end

# convert a 2x...x2xmx...xm probability array into a mx...xm correlation array if marg = false (no marginals)
# convert a 2x...x2xmx...xm probability array into a (m+1)x...x(m+1) correlation array if marg = true (marginals)
function correlation_tensor(p::AbstractArray{T, N2}; marg::Bool=false) where {T <: Number} where {N2}
    @assert iseven(N2)
    N = N2 ÷ 2
    m = size(p)[end]
    res = zeros(T, (marg ? m + 1 : m) * ones(Int, N)...)
    cia = CartesianIndices(Tuple(2 * ones(Int, N)))
    cix = CartesianIndices(Tuple((marg ? m + 1 : m) * ones(Int, N)))
    for x in cix
        y = [x[i] ≤ m ? x[i] : Colon() for i in 1:N]
        res[x] =
            sum((-1)^sum(x[i] ≤ m ? a[i] : 0 for i in 1:N) * sum(p[a, y...]) for a in cia) /
            m^sum(x[i] ≤ m ? 0 : 1 for i in 1:N)
        if T <: AbstractFloat
            if abs(res[x]) < Base.rtoldefault(T)
                res[x] = zero(T)
            end
        end
    end
    return res
end

function correlation_tensor(
    vec::AbstractMatrix{T},
    N::Int;
    rho=ketbra(multipartite_W(N)),
    marg::Bool=false,
    type=Complex{T},
) where {T <: Number}
    correlation_tensor(probability_tensor(vec, N; rho=rho, type=type); marg=marg)
end

function correlation_tensor(
    vecs::Vector{TB},
    N::Int;
    rho=ketbra(multipartite_W(N)),
    marg::Bool=false,
    type=Complex{T},
) where {TB <: AbstractMatrix{T}} where {T <: Number}
    correlation_tensor(probability_tensor(vecs, N; rho=rho, type=type); marg=marg)
end

# convert a mx3 Bloch matrix into a 2x...x2xmx...xm probability array
function probability_tensor(vec::AbstractMatrix{T}, N::Int; rho=ketbra(multipartite_W(N)), type=Complex{T}) where {T <: Number}
    @assert size(rho) == (2^N, 2^N)
    m = size(vec, 1)
    Aax = qubit_mes(vec; type=type)
    p = zeros(T, 2 * ones(Int, N)..., m * ones(Int, N)...)
    cia = CartesianIndices(Tuple(2 * ones(Int, N)))
    cix = CartesianIndices(Tuple(m * ones(Int, N)))
    for a in cia, x in cix
        p[a, x] = real(tr(kron([Aax[:, :, a[n], x[n]] for n in 1:N]...) * rho))
    end
    return p
end

# convert a mx3 Bloch matrix into a 2x...x2xmx...xm probability array
function probability_tensor(
    vecs::Vector{TB},
    N::Int;
    rho=ketbra(multipartite_W(N)),
    type=Complex{T},
) where {TB <: AbstractMatrix{T}} where {T <: Number}
    @assert length(vecs) == N
    @assert size(rho) == (2^N, 2^N)
    m = size(vecs[1], 1)
    Aax = [qubit_mes(vec; type=type) for vec in vecs]
    p = zeros(T, 2 * ones(Int, N)..., m * ones(Int, N)...)
    cia = CartesianIndices(Tuple(2 * ones(Int, N)))
    cix = CartesianIndices(Tuple(m * ones(Int, N)))
    for a in cia, x in cix
        p[a, x] = real(tr(kron([Aax[n][:, :, a[n], x[n]] for n in 1:N]...) * rho))
    end
    return p
end

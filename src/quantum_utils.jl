#############
#  GENERAL  #
#############

#  Pauli matrices
σI(; type = ComplexF64) = Matrix{type}([1 0; 0 1])
σX(; type = ComplexF64) = Matrix{type}([0 1; 1 0])
σY(; type = ComplexF64) = Matrix{type}([0 -im; im 0])
σZ(; type = ComplexF64) = Matrix{type}([1 0; 0 -1])

# projector onto a direction in the Bloch sphere
function qubit_proj(v::Vector{T}; mes::Bool = false, type = Complex{T}) where {T <: Number}
    if mes
        return [
            (σI(; type) - v[1] * σX(; type) - v[2] * σY(; type) - v[3] * σZ(; type)) / 2;;;
            (σI(; type) + v[1] * σX(; type) + v[2] * σY(; type) + v[3] * σZ(; type)) / 2
        ]
    else
        return (σI(; type) + v[1] * σX(; type) + v[2] * σY(; type) + v[3] * σZ(; type)) / 2
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

function rho_singlet(; type = Float64)
    psi = [zero(type), one(type), -one(type), zero(type)]
    return psi * psi' / type(2)
end
export rho_singlet

function rho_GHZ(N::Int; d = 2, type = Float64)
    aux = zeros(type, d * ones(Int, N)...)
    for i in 1:d
        aux[i * ones(Int, N)...] = one(type)
    end
    psi = reshape(aux, d^N, 1)
    return psi * psi' / type(d)
end
export rho_GHZ

function rho_W(N::Int; type = Float64)
    psi = zeros(type, 2^N)
    for n in 1:N
        psi[2^(n - 1) + 1] = one(type)
    end
    return psi * psi' / type(N)
end
export rho_W

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
export ketbra

##################
#  MEASUREMENTS  #
##################

# create a set of qubit measurements from an array of Bloch coordinates
function qubit_mes(v::Matrix{T}; type = Complex{T}) where {T <: Number}
    res = zeros(type, 2, 2, 2, size(v, 1))
    for i in 1:size(v, 1)
        res[:, :, :, i] = qubit_proj(v[i, :]; mes = true, type)
    end
    return res
end
export qubit_mes

# platonic solids
cube_vec() = [1 1 1; 1 -1 -1; -1 1 -1; -1 -1 1] / sqrt(3)
export cube_vec

octahedron_vec() = [1 0 0; 0 1 0; 0 0 1]
export octahedron_vec

function icosahedron_vec(; type = Float64)
    φ = (1 + sqrt(type(5))) / 2
    return [0 1 φ; 0 1 -φ; 1 φ 0; 1 -φ 0; φ 0 1; φ 0 -1] / sqrt(2 + φ)
end
export icosahedron_vec

function dodecahedron_vec(; type = Float64)
    φ = (1 + sqrt(type(5))) / 2
    return [
        1 1 1
        1 -1 -1
        -1 1 -1
        -1 -1 1
        0 1 / φ φ
        0 1 / φ -φ
        1 / φ φ 0
        1 / φ -φ 0
        φ 0 1 / φ
        φ 0 -1 / φ
    ] / sqrt(type(3))
end
export dodecahedron_vec

# measurements from arXiv:1609.05011 (regular polyhedron in the XY plane)
function polygonXY_vec(m::Int; type = Float64)
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
export polygonXY_vec

# measurements from arXiv:1609.06114 (rotated regular polyhedron)
function HQVNB17_vec(n::Int; type = Float64)
    m = isodd(n) ? n^2 : n^2 - n + 1
    res = zeros(type, m, 3)
    if iseven(n)
        tabv = [
            (
                    cos(i1 * type(pi) / n) * cos(i2 * type(pi) / n),
                    sin(i1 * type(pi) / n) * cos(i2 * type(pi) / n),
                    sin(i2 * type(pi) / n),
                ) for i1 in 0:(n - 1), i2 in 0:(n - 1) if i2 != n ÷ 2
        ][:]
        push!(tabv, (0.0, 0.0, 1.0))
    else
        tabv = [
            (
                    cos(i1 * type(pi) / n) * cos(i2 * type(pi) / n),
                    sin(i1 * type(pi) / n) * cos(i2 * type(pi) / n),
                    sin(i2 * type(pi) / n),
                ) for i1 in 0:(n - 1), i2 in 0:(n - 1)
        ][:]
    end
    for i in 1:m
        res[i, 1] = tabv[i][1]
        res[i, 2] = tabv[i][2]
        res[i, 3] = tabv[i][3]
    end
    return res
end
export HQVNB17_vec

#  create a set of (projective) POVMs out of a set of bases
function povm(B::Array{T, 3}) where {T <: Number}
    d = size(B, 1)
    n = size(B, 2)
    k = size(B, 3)
    res = zeros(ComplexF64, d, d, n, k)
    for a in 1:n, x in 1:k
        res[:, :, a, x] = B[:, a, x] * B[:, a, x]'
    end
    return res
end
export povm

# construction of the measurements from Eqs. (5) and (6) from quant-ph/0605182
# Barrett Kent Pironio
function BKP_mes(d::Int, N::Int)
    omega = exp(2 * im * pi / d)
    rA = [omega^(q * (r - (k - 1 / 2) / N)) / sqrt(d) for q in 0:(d - 1), r in 0:(d - 1), k in 1:N]
    rB = [omega^(-q * (r - l / N)) / sqrt(d) for q in 0:(d - 1), r in 0:(d - 1), l in 1:N]
    return povm(rA), povm(rB)
end

##############
# CONVERSION #
##############

# convert a 2x...x2xmx...xm probability array into a mx...xm correlation array if marg = false (no marginals)
# convert a 2x...x2xmx...xm probability array into a (m+1)x...x(m+1) correlation array if marg = true (marginals)
function correlation_tensor(p::AbstractArray{T, N2}; marg::Bool = false) where {T <: Number, N2}
    @assert iseven(N2)
    N = N2 ÷ 2
    m = size(p)[end]
    res = zeros(T, (marg ? m + 1 : m) * ones(Int, N)...)
    cia = CartesianIndices(Tuple(2 * ones(Int, N)))
    cix = CartesianIndices(Tuple((marg ? m + 1 : m) * ones(Int, N)))
    for x in cix
        y = [x[i] ≤ m ? x[i] : Colon() for i in 1:N]
        res[x] =
            sum((-1)^sum(x[i] ≤ m ? a[i] - 1 : 0 for i in 1:N) * sum(p[a, y...]) for a in cia) /
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
        rho = rho_W(N; type = T),
        marg::Bool = false,
        type = Complex{T},
    ) where {T <: Number}
    correlation_tensor(probability_tensor(vec, N; rho, type); marg)
end

function correlation_tensor(
        vecs::Vector{TB},
        N::Int;
        rho = rho_W(N; type = TB),
        marg::Bool = false,
        type = Complex{T},
    ) where {TB <: AbstractMatrix{T}} where {T <: Number}
    correlation_tensor(probability_tensor(vecs, N; rho, type); marg)
end
export correlation_tensor

# convert a mx3 Bloch matrix into a 2x...x2xmx...xm probability array
function probability_tensor(vec::AbstractMatrix{T}, N::Int; rho = rho_W(N; type = T), type = Complex{T}) where {T <: Number}
    @assert size(rho) == (2^N, 2^N)
    m = size(vec, 1)
    Aax = qubit_mes(vec; type)
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
        rho = rho_W(N; type = TB),
        type = Complex{T},
    ) where {TB <: AbstractMatrix{T}} where {T <: Number}
    @assert length(vecs) == N
    @assert size(rho) == (2^N, 2^N)
    m = size(vecs[1], 1)
    Aax = [qubit_mes(vec; type) for vec in vecs]
    p = zeros(T, 2 * ones(Int, N)..., m * ones(Int, N)...)
    cia = CartesianIndices(Tuple(2 * ones(Int, N)))
    cix = CartesianIndices(Tuple(m * ones(Int, N)))
    for a in cia, x in cix
        p[a, x] = real(tr(kron([Aax[n][:, :, a[n], x[n]] for n in 1:N]...) * rho))
    end
    return p
end

# convert a N sets of m o-outcome POVMs acting on C^d into a dx...xdxmx...xm probability array
function probability_tensor(
        Aax::Vector{TB},
        N::Int;
        rho = rho_GHZ(N; d = size(Aax[1], 1), type = T),
    ) where {TB <: AbstractArray{Complex{T}, 4}} where {T <: Number}
    d, _, o, m = size(Aax[1])
    @assert length(Aax) == N
    @assert size(rho) == (d^N, d^N)
    p = zeros(T, o * ones(Int, N)..., m * ones(Int, N)...)
    cia = CartesianIndices(Tuple(o * ones(Int, N)))
    cix = CartesianIndices(Tuple(m * ones(Int, N)))
    for a in cia, x in cix
        p[a, x] = real(tr(kron([Aax[n][:, :, a[n], x[n]] for n in 1:N]...) * rho))
    end
    return p
end
export probability_tensor

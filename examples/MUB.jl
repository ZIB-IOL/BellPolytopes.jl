using BellPolytopes
using Combinatorics
using LinearAlgebra
using Nemo
using Primes
using Random
using RandomMatrix
Random.seed!(0)

# Construction of the standard complete set of MUBs
# The dimension d can be any integer greater than two
# The output contains min_i p_i^r_i+1 bases where d = p_1^r_1*...*p_n^r_n
# Reference: arXiv:1004.3348
# Contact sebastien.designolle@gmail.com for questions
function mub(d)
    # Auxiliary function to compute the trace in finite fields
    function tr_int(a::fq_nmod)
        parse(Int, string(tr(a)))
    end
    f = collect(Primes.factor(d))
    p = f[1][1]
    r = f[1][2]
    if length(f) > 1
        B_aux1 = mub(p^r)
        B_aux2 = mub(d÷p^r)
        k = min(size(B_aux1, 3), size(B_aux2, 3))
        B = zeros(Complex{Float64}, d, d, k)
        for j = 1:k
            B[:, :, j] = kron(B_aux1[:, :, j], B_aux2[:, :, j])
        end
    else
        B = zeros(Complex{Float64}, d, d, d+1)
        B[:, :, 1] = Matrix(I, d, d)
        gamma = exp(2*im*pi/p)
        f, x = FiniteField(p, r, "x")
        pow = [x^i for i = 0:r-1]
        el = [sum(digits(i, base=p, pad=r).*pow) for i = 0:d-1]
        if p == 2
            for i = 1:d
                for k = 0:d-1
                    for q = 0:d-1
                        aux = 1
                        q_bin = digits(q, base=2, pad=r)
                        for m = 0:r-1
                            for n = 0:r-1
                                aux = aux*conj(im^tr_int(el[i]*el[q_bin[m+1]*2^m+1]*el[q_bin[n+1]*2^n+1]))
                            end
                        end
                        B[:, k+1, i+1] = B[:, k+1, i+1]+(-1)^tr_int(el[q+1]*el[k+1])*aux*B[:, q+1, 1]/sqrt(d)
                    end
                end
            end
        else
            inv_two = inv(2*one(f))
            for i = 1:d
                for k = 0:d-1
                    for q = 0:d-1
                        B[:, k+1, i+1] = B[:, k+1, i+1]+gamma^tr_int(-el[q+1]*el[k+1])*gamma^tr_int(el[i-1+1]*el[q+1]*el[q+1]*inv_two)*B[:, q+1, 1]/sqrt(d);
                    end
                end
            end
        end
    end
    return B
end

# Select a specific subset with k bases
function mub(d, k, s)
    B = mub(d)
    subs = collect(combinations(1:size(B, 3), k))
    sub = subs[s]
    return B[:, :, sub]
end

# Select the first subset with k bases
function mub(d, k)
    return mub(d, k, 1)
end

# Check whether the input is indeed mutually unbiased
function is_mu(B; tol=1e-10)
    d = size(B, 1)
    k = size(B, 3)
    for x = 1:k, y = x:k, a = 1:d, b = 1:d
        if x == y
            if a == b
                aux = 1
            else
                aux = 0
            end
        else
            aux = 1/sqrt(d)
        end
        if abs(dot(B[:, a, x], B[:, b, y]))-aux > tol
            return false
        end
    end
    return true
end

d = 3
N = 2
bas = mub(d)
k = size(bas, 3)
basA = bas
Aax = povm(basA)
U = randUnitary(d)
basB = bas
for a in 1:k
    basB[:, :, a] = U * bas[:, :, a]
end
Bby = povm(bas)
p = probability_tensor([Aax, Bby], N)
res = bell_frank_wolfe(p; verbose=3, prob=true, mode=1)
println()

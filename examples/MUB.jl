using BellPolytopes
using Ket

# dimension
d = 4

# Alice's measurements
basA = mub(d)
Aax = povm(basA)

# Bob's measurements
U = random_unitary(d)
basB = [U * b for b in basA]
Bby = povm(basB)

# probability tensor
p = tensor_probability(state_phiplus(d), Aax, Bby)

# Frank-Wolfe
res = bell_frank_wolfe(p; verbose = 3, prob = true, mode = 1)
println()

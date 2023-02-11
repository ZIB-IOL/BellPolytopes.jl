var documenterSearchIndex = {"docs":
[{"location":"api/#API-reference","page":"API reference","title":"API reference","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"Modules = [BellPolytopes]","category":"page"},{"location":"api/#BellPolytopes.bell_frank_wolfe-Union{Tuple{Array{T, N}}, Tuple{T}, Tuple{N}} where {N, T<:Number}","page":"API reference","title":"BellPolytopes.bell_frank_wolfe","text":"Calls the lazy pairwise blended conditional gradient algorithm from Frank-Wolfe package.\n\nArguments:\n\np: a correlation tensor of order N.\n\nReturns:\n\nx: a correlation tensor of order N, the output of the Frank-Wolfe algorithm,\nds: a deterministic strategy, the atom returned by the last LMO,\nprimal: ½|x-v₀*p|²\ndual_gap: ⟨x-v₀*p, x-ds⟩\ntraj_data: trajectory of the algorithm,\nactive_set: all deterministic strategies used for the decomposition of the last iterate x, contains fields weights, atoms, and x,\nM: a Bell inequality, meaningful only if the dual gap is small enough\nβ: the local bound of the inequality parametrised by M, reliable only if the last LMO is exact.\n\nOptional arguments:\n\nmarg: a boolean, indicates if p contains marginals\nv0: the visibility used to make a nonlocal p closer to the local polytope,\nepsilon: the tolerance, used as a stopping criterion (when the primal value or the dual gap go below its value), by default 1e-7,\nverbose: an integer, indicates the level of verbosity from 0 to 3,\nshr2: the potential underlying shrinking factor, used to display the lower bound in the callback,\nTD: type of the computations, by default the one of ̀p,\nmode: an integer, 0 is for the heuristic LMO, 1 for the enumeration LMO,\nnb: an integer, number of random tries in the LMO, if heuristic, by default 10^2,\nTL: type of the last call of the LMO,\nmode_last: an integer, mode of the last call of the LMO, -1 for no last call\nnb_last: an integer, number of random tries in the last LMO, if heuristic, by default 10^5,\nsym: a boolean, indicates if the symmetry of the input should be used, by default automatic choice,\nuse_array: a boolean, indicates to store the full deterministic strategies to trade memory for speed in multipartite scenarios,\ncallback_interval: an integer, print interval if verbose = 3,\nseed: an integer, the initial random seed.\n\n\n\n\n\n","category":"method"},{"location":"api/#BellPolytopes.local_bound-Union{Tuple{Array{T, N}}, Tuple{T}, Tuple{N}} where {N, T<:Number}","page":"API reference","title":"BellPolytopes.local_bound","text":"Compute the local bound of a Bell inequality parametrised by M.\n\n\n\n\n\n","category":"method"},{"location":"api/#BellPolytopes.nonlocality_threshold-Union{Tuple{TB}, Tuple{T}, Tuple{Union{Vector{TB}, AbstractMatrix{T} where T, TB}, Int64}} where {T<:Number, TB<:AbstractMatrix{T}}","page":"API reference","title":"BellPolytopes.nonlocality_threshold","text":"Compute the nonlocality threshold of the qubit measurements encoded by the Bloch vectors vec in a Bell scenario with N parties.\n\nArguments:\n\nvec: an m × 3 matrix with Bloch vectors coordinates,\nN: the number of parties.\n\nReturns:\n\nlower_bound_infinite: a lower bound on the nonlocality threshold under all projective measurements (in the subspace spanned by vec in the Bloch sphere),\nlower_bound: a lower bound on the nonlocality threshold under the measurements provided in input,\nupper_bound: a (heuristic) upper bound on the nonlocality threshold under the measurements provided in input, also valid for all projective measurements\nlocal_model: a decomposition of the correlation tensor obtained by applying the measurements encoded by the Bloch vectors vec on all N subsystems of the shared state rho with visibility lower_bound,\nbell_inequality: a (heuristic) Bell inequality corresponding to upper_bound.\n\nOptional arguments:\n\nrho: the shared state, by default the singlet state in the bipartite case and the GHZ state otherwise,\nv0: the initial visibility, which should be an upper bound on the nonlocality threshold, 1.0 by default,\nprecision: number of digits of lower_bound, 4 by default,\nfor the other optional arguments, see bell_frank_wolfe.\n\n\n\n\n\n","category":"method"},{"location":"api/#BellPolytopes.pythagorean_approximation-Union{Tuple{Matrix{T}}, Tuple{T}} where T<:Number","page":"API reference","title":"BellPolytopes.pythagorean_approximation","text":"Compute a rational approximation of a m × 3 Bloch matrix.\n\n\n\n\n\n","category":"method"},{"location":"api/#BellPolytopes.shrinking_squared-Union{Tuple{AbstractMatrix{T}}, Tuple{T}} where T<:Number","page":"API reference","title":"BellPolytopes.shrinking_squared","text":"Compute the shrinking factor of a m × 3 Bloch matrix, symmetrising it to account for antipodal vectors.\n\n\n\n\n\n","category":"method"},{"location":"","page":"Home","title":"Home","text":"EditURL = \"https://github.com/ZIB-IOL/BellPolytopes.jl/blob/master/README.md\"","category":"page"},{"location":"#BellPolytopes.jl","page":"Home","title":"BellPolytopes.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: Stable) (Image: Dev) (Image: Build Status) (Image: Genie Downloads)","category":"page"},{"location":"","page":"Home","title":"Home","text":"This package addresses the membership problem for local polytopes: it constructs Bell inequalities and local models in multipartite Bell scenarios with binary outcomes.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The original article for which it was written can be found here:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Improved local models and new Bell inequalities via Frank-Wolfe algorithms.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The most recent release is available via the julia package manager, e.g., with","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Pkg\nPkg.add(\"BellPolytopes\")","category":"page"},{"location":"","page":"Home","title":"Home","text":"or the main branch:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Pkg.add(url=\"https://github.com/ZIB-IOL/BellPolytopes.jl\", rev=\"main\")","category":"page"},{"location":"#Getting-started","page":"Home","title":"Getting started","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Let's say we want to characterise the nonlocality threshold obtained with the two-qubit maximally entangled state and measurements whose Bloch vectors form an icosahedron. Using BellPolytopes.jl, here is what the code looks like.","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> using BellPolytopes\n\njulia> N = 2; # bipartite scenario\n\njulia> rho = rho_GHZ(N) # two-qubit maximally entangled state\n4×4 Matrix{Float64}:\n 0.5  0.0  0.0  0.5\n 0.0  0.0  0.0  0.0\n 0.0  0.0  0.0  0.0\n 0.5  0.0  0.0  0.5\n\njulia> measurements_vec = icosahedron_vec() # Bloch vectors forming an icosahedron\n6×3 Matrix{Float64}:\n 0.0        0.525731   0.850651\n 0.0        0.525731  -0.850651\n 0.525731   0.850651   0.0\n 0.525731  -0.850651   0.0\n 0.850651   0.0        0.525731\n 0.850651   0.0       -0.525731\n\njulia> _, lower_bound, upper_bound, local_model, bell_inequality, _ =\n       nonlocality_threshold(measurements_vec, N; rho = rho);\n\njulia> println([lower_bound, upper_bound])\n[0.7784, 0.7784]\n\njulia> p = correlation_tensor(measurements_vec, N; rho = rho)\n6×6 Matrix{Float64}:\n  0.447214  -1.0       -0.447214   0.447214   0.447214  -0.447214\n -1.0        0.447214  -0.447214   0.447214  -0.447214   0.447214\n -0.447214  -0.447214  -0.447214   1.0        0.447214   0.447214\n  0.447214   0.447214   1.0       -0.447214   0.447214   0.447214\n  0.447214  -0.447214   0.447214   0.447214   1.0        0.447214\n -0.447214   0.447214   0.447214   0.447214   0.447214   1.0\n\njulia> norm(lower_bound * p - sum(local_model.weights[i] * local_model.atoms[i] for i in 1:length(local_model))) < 1e-3 # checking local model\ntrue\n\njulia> local_bound(bell_inequality)[1] / dot(bell_inequality, p) # checking the Bell inequality\n0.7783914488195466","category":"page"},{"location":"","page":"Home","title":"Home","text":"More examples can be found in the corresponding folder of the package.","category":"page"}]
}

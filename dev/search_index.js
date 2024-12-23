var documenterSearchIndex = {"docs":
[{"location":"api/#API-reference","page":"API reference","title":"API reference","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"Modules = [BellPolytopes]","category":"page"},{"location":"api/#BellPolytopes.bell_frank_wolfe-Union{Tuple{Array{T, N}}, Tuple{N}, Tuple{T}} where {T<:Number, N}","page":"API reference","title":"BellPolytopes.bell_frank_wolfe","text":"Calls the lazy pairwise blended conditional gradient algorithm from Frank-Wolfe package.\n\nArguments:\n\np: a correlation/probability tensor of order N.\n\nReturns:\n\nx: a correlation/probability tensor of order N, the output of the Frank-Wolfe algorithm,\nds: a deterministic strategy, the atom returned by the last LMO,\nprimal: ½|x-v₀*p|²,\ndual_gap: ⟨x-v₀*p, x-ds⟩,\nactive_set: all deterministic strategies used for the decomposition of the last iterate x, contains fields weights, atoms, and x,\nM: a Bell inequality, meaningful only if the dual gap is small enough,\nβ: the local bound of the inequality parametrised by M, reliable only if the last LMO is exact.\n\nOptional arguments:\n\no: same type as p, corresponds to the noise to be added, by default the center of the polytope,\nprob: a boolean, indicates if p is a corelation or probability array,\nmarg: a boolean, indicates if p contains marginals,\nv0: the visibility used to make a nonlocal p closer to the local polytope,\nepsilon: the tolerance, used as a stopping criterion (when the primal value or the dual gap go below its value), by default 1e-7,\nverbose: an integer, indicates the level of verbosity from 0 to 3,\nshr2: the potential underlying shrinking factor, used to display the lower bound in the callback,\nmode: an integer, 0 is for the heuristic LMO, 1 for the enumeration LMO,\nnb: an integer, number of random tries in the LMO, if heuristic, by default 10^2,\nTL: type of the last call of the LMO,\nmode_last: an integer, mode of the last call of the LMO, -1 for no last call,\nnb_last: an integer, number of random tries in the last LMO, if heuristic, by default 10^5,\nsym: a boolean, indicates if the symmetry of the input should be used, by default automatic choice,\nuse_array: a boolean, indicates to store the full deterministic strategies to trade memory for speed in multipartite scenarios,\ncallback_interval: an integer, print interval if verbose = 3,\nseed: an integer, the initial random seed.\n\n\n\n\n\n","category":"method"},{"location":"api/#BellPolytopes.local_bound-Tuple{Any}","page":"API reference","title":"BellPolytopes.local_bound","text":"Compute the local bound of a Bell inequality parametrised by M. No symmetry detection is implemented yet, used mostly for pedagogy and tests.\n\n\n\n\n\n","category":"method"},{"location":"api/#BellPolytopes.nonlocality_threshold-Union{Tuple{TB}, Tuple{T}, Tuple{Union{Vector{TB}, AbstractMatrix{T} where T, TB}, Int64}} where {T<:Number, TB<:AbstractMatrix{T}}","page":"API reference","title":"BellPolytopes.nonlocality_threshold","text":"Compute the nonlocality threshold of the qubit measurements encoded by the Bloch vectors vec in a Bell scenario with N parties.\n\nArguments:\n\nvec: an m × 3 matrix with Bloch vectors coordinates,\nN: the number of parties.\n\nReturns:\n\nlower_bound_infinite: a lower bound on the nonlocality threshold under all projective measurements (in the subspace spanned by vec in the Bloch sphere),\nlower_bound: a lower bound on the nonlocality threshold under the measurements provided in input,\nupper_bound: a (heuristic) upper bound on the nonlocality threshold under the measurements provided in input, also valid for all projective measurements,\nlocal_model: a decomposition of the correlation tensor obtained by applying the measurements encoded by the Bloch vectors vec on all N subsystems of the shared state rho with visibility lower_bound,\nbell_inequality: a (heuristic) Bell inequality corresponding to upper_bound.\n\nOptional arguments:\n\nrho: the shared state, by default the singlet state in the bipartite case and the GHZ state otherwise,\nv0: the initial visibility, which should be an upper bound on the nonlocality threshold, 1.0 by default,\nprecision: number of digits of lower_bound, 4 by default,\nfor the other optional arguments, see bell_frank_wolfe.\n\n\n\n\n\n","category":"method"},{"location":"api/#BellPolytopes.pythagorean_approximation-Union{Tuple{Matrix{T}}, Tuple{T}} where T<:Number","page":"API reference","title":"BellPolytopes.pythagorean_approximation","text":"Compute a rational approximation of a m × 3 Bloch matrix.\n\n\n\n\n\n","category":"method"},{"location":"api/#BellPolytopes.shrinking_squared-Union{Tuple{AbstractMatrix{T}}, Tuple{T}} where T<:Number","page":"API reference","title":"BellPolytopes.shrinking_squared","text":"Compute the shrinking factor of a m × 3 Bloch matrix, symmetrising it to account for antipodal vectors.\n\n\n\n\n\n","category":"method"},{"location":"","page":"Home","title":"Home","text":"EditURL = \"https://github.com/ZIB-IOL/BellPolytopes.jl/blob/master/README.md\"","category":"page"},{"location":"#BellPolytopes.jl","page":"Home","title":"BellPolytopes.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: Dev) (Image: Build Status)","category":"page"},{"location":"","page":"Home","title":"Home","text":"This package addresses the membership problem for local polytopes: it constructs Bell inequalities and local models in multipartite Bell scenarios with arbitrary settings.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The original article for which it was written can be found here:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Improved local models and new Bell inequalities via Frank-Wolfe algorithms.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The most recent release is available via the julia package manager, e.g., with","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Pkg\nPkg.add(\"BellPolytopes\")","category":"page"},{"location":"","page":"Home","title":"Home","text":"or the main branch:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Pkg.add(url=\"https://github.com/ZIB-IOL/BellPolytopes.jl\", rev=\"main\")","category":"page"},{"location":"#Getting-started","page":"Home","title":"Getting started","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Let's say we want to characterise the nonlocality threshold obtained with the two-qubit maximally entangled state and measurements whose Bloch vectors form an icosahedron. Using BellPolytopes.jl, here is what the code looks like.","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> using BellPolytopes, LinearAlgebra\n\njulia> N = 2; # bipartite scenario\n\njulia> rho = rho_GHZ(N) # two-qubit maximally entangled state\n4×4 Matrix{Float64}:\n 0.5  0.0  0.0  0.5\n 0.0  0.0  0.0  0.0\n 0.0  0.0  0.0  0.0\n 0.5  0.0  0.0  0.5\n\njulia> measurements_vec = icosahedron_vec() # Bloch vectors forming an icosahedron\n6×3 Matrix{Float64}:\n 0.0        0.525731   0.850651\n 0.0        0.525731  -0.850651\n 0.525731   0.850651   0.0\n 0.525731  -0.850651   0.0\n 0.850651   0.0        0.525731\n 0.850651   0.0       -0.525731\n\njulia> _, lower_bound, upper_bound, local_model, bell_inequality, _ =\n       nonlocality_threshold(measurements_vec, N; rho = rho);\n\njulia> println([lower_bound, upper_bound])\n[0.7784, 0.7784]\n\njulia> p = correlation_tensor(measurements_vec, N; rho = rho)\n6×6 Matrix{Float64}:\n  0.447214  -1.0       -0.447214   0.447214   0.447214  -0.447214\n -1.0        0.447214  -0.447214   0.447214  -0.447214   0.447214\n -0.447214  -0.447214  -0.447214   1.0        0.447214   0.447214\n  0.447214   0.447214   1.0       -0.447214   0.447214   0.447214\n  0.447214  -0.447214   0.447214   0.447214   1.0        0.447214\n -0.447214   0.447214   0.447214   0.447214   0.447214   1.0\n\njulia> final_iterate = sum(local_model.weights[i] * local_model.atoms[i] for i in 1:length(local_model));\n\njulia> norm(final_iterate - lower_bound * p) < 1e-3 # checking local model\ntrue\n\njulia> local_bound(bell_inequality)[1] / dot(bell_inequality, p) # checking the Bell inequality\n0.7783914488195466","category":"page"},{"location":"#Under-the-hood","page":"Home","title":"Under the hood","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The computation is based on an efficient variant of the Frank-Wolfe algorithm to iteratively find the local point closest to the input correlation tensor. See this recent review for an introduction to the method and the package FrankWolfe.jl for the implementation on which this package relies.","category":"page"},{"location":"","page":"Home","title":"Home","text":"In a nutshell, each step gets closer to the objective point:","category":"page"},{"location":"","page":"Home","title":"Home","text":"either by moving towards a good vertex of the local polytope,\nor by astutely combining the vertices (or atoms) already found and stored in the active set.","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> res = bell_frank_wolfe(p; v0=0.8, verbose=3, callback_interval=10^2, mode_last=-1);\n\nVisibility: 0.8\n Symmetric: true\n   #Inputs: 6\n Dimension: 21\n\nIntervals\n    Print: 100\n   Renorm: 100\n   Reduce: 10000\n    Upper: 1000\nIncrement: 1000\n\n   Iteration        Primal      Dual gap    Time (sec)       #It/sec    #Atoms       #LMO\n         100    8.7570e-03    6.0089e-02    6.9701e-04    1.4347e+05        11         26\n         200    5.9241e-03    5.4948e-02    9.4910e-04    2.1073e+05        16         33\n         300    3.5594e-03    3.4747e-02    1.1942e-03    2.5122e+05        18         40\n         400    1.9068e-03    3.4747e-02    1.3469e-03    2.9697e+05        16         42\n         500    1.8093e-03    5.7632e-06    1.5409e-03    3.2448e+05        14         48\n\n    Primal: 1.81e-03\n  Dual gap: 2.60e-08\n      Time: 1.63e-03\n    It/sec: 3.28e+05\n    #Atoms: 14\n\nv_c ≤ 0.778392","category":"page"},{"location":"#Going-further","page":"Home","title":"Going further","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"More examples can be found in the corresponding folder of the package. They include the construction of a Bell inequality with a higher tolerance to noise as CHSH as well as multipartite instances.","category":"page"}]
}

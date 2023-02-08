var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"EditURL = \"https://github.com/ZIB-IOL/BellPolytopes.jl/blob/master/README.md\"","category":"page"},{"location":"#BellPolytopes","page":"Home","title":"BellPolytopes","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: Stable) (Image: Dev) (Image: Build Status) <!– (Image: Coverage) –>","category":"page"},{"location":"","page":"Home","title":"Home","text":"This package addresses the membership problem for local polytopes: it constructs Bell inequalities and local models in multipartite Bell scenarios with binary outcomes.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The most recent release is available via the julia package manager, e.g., with","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Pkg\nPkg.add(\"BellPolytopes\")","category":"page"},{"location":"","page":"Home","title":"Home","text":"or the main branch:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Pkg.add(url=\"https://github.com/ZIB-IOL/BellPolytopes.jl\", rev=\"main\")","category":"page"},{"location":"#Getting-started","page":"Home","title":"Getting started","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Let's say we want to characterise the nonlocality threshold obtained with the two-qubit maximally entangled state and measurements whose Bloch vectors form an icosahedron. Using BellPolytopes.jl, here is what the code looks like.","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> using BellPolytopes\n\njulia> N = 2; # bipartite scenario\n\njulia> rho = rho_GHZ(N) # two-qubit maximally entangled state\n4×4 Matrix{Float64}:\n 0.5  0.0  0.0  0.5\n 0.0  0.0  0.0  0.0\n 0.0  0.0  0.0  0.0\n 0.5  0.0  0.0  0.5\n\njulia> measurements_vec = icosahedron_vec() # Bloch vectors forming an icosahedron\n6×3 Matrix{Float64}:\n 0.0        0.525731   0.850651\n 0.0        0.525731  -0.850651\n 0.525731   0.850651   0.0\n 0.525731  -0.850651   0.0\n 0.850651   0.0        0.525731\n 0.850651   0.0       -0.525731\n\njulia> _, lower_bound, upper_bound, local_model, bell_inequality, _ =\n       nonlocality_threshold(measurements_vec, N; rho = rho);\n\njulia> println([lower_bound, upper_bound])\n[0.7784, 0.7784]\n\njulia> p = correlation_tensor(measurements_vec, N; rho = rho)\n6×6 Matrix{Float64}:\n  0.447214  -1.0       -0.447214   0.447214   0.447214  -0.447214\n -1.0        0.447214  -0.447214   0.447214  -0.447214   0.447214\n -0.447214  -0.447214  -0.447214   1.0        0.447214   0.447214\n  0.447214   0.447214   1.0       -0.447214   0.447214   0.447214\n  0.447214  -0.447214   0.447214   0.447214   1.0        0.447214\n -0.447214   0.447214   0.447214   0.447214   0.447214   1.0\n\njulia> norm(lower_bound * p - sum(local_model.weights[i] * local_model.atoms[i] for i in 1:length(local_model))) < 1e-3 # checking local model\ntrue\n\njulia> local_bound(bell_inequality)[1] / dot(bell_inequality, p) # checking the Bell inequality\n0.7783914488195466","category":"page"},{"location":"","page":"Home","title":"Home","text":"More examples can be found in the corresponding folder of the package.","category":"page"}]
}

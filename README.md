# BellPolytopes

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://zib-iol.github.io/BellPolytopes.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://zib-iol.github.io/BellPolytopes.jl/dev/)
[![Build Status](https://github.com/zib-iol/BellPolytopes.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/zib-iol/BellPolytopes.jl/actions/workflows/CI.yml?query=branch%3Amain)
<!-- [![Coverage](https://codecov.io/gh/zib-iol/BellPolytopes.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/zib-iol/BellPolytopes.jl) -->

This package addresses the membership problem for local polytopes: it constructs Bell inequalities and local models in multipartite Bell scenarios with binary outcomes.

## Installation

The most recent release is available via the julia package manager, e.g., with

```julia
using Pkg
Pkg.add("BellPolytopes")
```

or the main branch:

```julia
Pkg.add(url="https://github.com/ZIB-IOL/BellPolytopes.jl", rev="main")
```

## Getting started

Let's say we want to characterise the nonlocality threshold obtained with the two-qubit maximally entangled state and measurements whose Bloch vectors form an icosahedron.
Using `BellPolytopes.jl`, here is what the code looks like.

```julia
julia> using BellPolytopes

julia> N = 2; # bipartite scenario

julia> rho = rho_GHZ(N) # two-qubit maximally entangled state
4×4 Matrix{Float64}:
 0.5  0.0  0.0  0.5
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0
 0.5  0.0  0.0  0.5

julia> measurements_vec = icosahedron_vec() # Bloch vectors forming an icosahedron
6×3 Matrix{Float64}:
 0.0        0.525731   0.850651
 0.0        0.525731  -0.850651
 0.525731   0.850651   0.0
 0.525731  -0.850651   0.0
 0.850651   0.0        0.525731
 0.850651   0.0       -0.525731

julia> _, lower_bound, upper_bound, local_model, bell_inequality, _ =
       nonlocality_threshold(measurements_vec, N; rho = rho);

julia> println([lower_bound, upper_bound])
[0.7784, 0.7784]

julia> p = correlation_tensor(measurements_vec, N; rho = rho)
6×6 Matrix{Float64}:
  0.447214  -1.0       -0.447214   0.447214   0.447214  -0.447214
 -1.0        0.447214  -0.447214   0.447214  -0.447214   0.447214
 -0.447214  -0.447214  -0.447214   1.0        0.447214   0.447214
  0.447214   0.447214   1.0       -0.447214   0.447214   0.447214
  0.447214  -0.447214   0.447214   0.447214   1.0        0.447214
 -0.447214   0.447214   0.447214   0.447214   0.447214   1.0

julia> norm(lower_bound * p - sum(local_model.weights[i] * local_model.atoms[i] for i in 1:length(local_model))) < 1e-3 # checking local model
true

julia> local_bound(bell_inequality)[1] / dot(bell_inequality, p) # checking the Bell inequality
0.7783914488195466
```

More examples can be found in the corresponding folder of the package.

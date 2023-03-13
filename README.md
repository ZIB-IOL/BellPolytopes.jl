# BellPolytopes.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://zib-iol.github.io/BellPolytopes.jl/dev/)
[![Build Status](https://github.com/zib-iol/BellPolytopes.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/zib-iol/BellPolytopes.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Genie Downloads](https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/BellPolytopes)](https://pkgs.genieframework.com?packages=BellPolytopes)

This package addresses the membership problem for local polytopes: it constructs Bell inequalities and local models in multipartite Bell scenarios with binary outcomes.

The original article for which it was written can be found here:

> [Improved local models and new Bell inequalities via Frank-Wolfe algorithms](http://arxiv.org/abs/2302.04721).

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
julia> using BellPolytopes, LinearAlgebra

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

julia> final_iterate = sum(local_model.weights[i] * local_model.atoms[i] for i in 1:length(local_model));

julia> norm(final_iterate - lower_bound * p) < 1e-3 # checking local model
true

julia> local_bound(bell_inequality)[1] / dot(bell_inequality, p) # checking the Bell inequality
0.7783914488195466
```

## Under the hood

The computation relies on an efficient variant of the Frank-Wolfe algorithm to iteratively find the local point closest to the input correlation tensor.
See this recent [review](https://arxiv.org/abs/2211.14103) for an introduction to the method.

In a nutshell, each step gets closer to the objective point:
* either by moving towards a *good* vertex of the local polytope,
* or by astutely combining the vertices (or atoms) already found and stored in the *active set*.

```julia
julia> res = bell_frank_wolfe(p; v0=0.8, verbose=3, callback_interval=10^2, mode_last=-1);

Visibility: 0.8
 Symmetric: true
   #Inputs: 6
 Dimension: 21

Intervals
    Print: 100
   Renorm: 100
   Reduce: 10000
    Upper: 1000
Increment: 1000

   Iteration        Primal      Dual gap    Time (sec)       #It/sec    #Atoms       #LMO
         100    8.7570e-03    6.0089e-02    6.9701e-04    1.4347e+05        11         26
         200    5.9241e-03    5.4948e-02    9.4910e-04    2.1073e+05        16         33
         300    3.5594e-03    3.4747e-02    1.1942e-03    2.5122e+05        18         40
         400    1.9068e-03    3.4747e-02    1.3469e-03    2.9697e+05        16         42
         500    1.8093e-03    5.7632e-06    1.5409e-03    3.2448e+05        14         48

    Primal: 1.81e-03
  Dual gap: 2.60e-08
      Time: 1.63e-03
    It/sec: 3.28e+05
    #Atoms: 14

v_c ≤ 0.778392
```

## Going further

More examples can be found in the corresponding folder of the package.
They include the construction of a Bell inequality with a higher tolerance to noise as CHSH as well as multipartite instances.

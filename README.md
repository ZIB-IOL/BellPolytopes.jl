# BellPolytopes.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://zib-iol.github.io/BellPolytopes.jl/dev/)
[![Build Status](https://github.com/zib-iol/BellPolytopes.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/zib-iol/BellPolytopes.jl/actions/workflows/CI.yml?query=branch%3Amain)

This package addresses the membership problem for local polytopes: it constructs Bell inequalities and local models in multipartite Bell scenarios with arbitrary settings.

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
julia> using BellPolytopes, Ket, LinearAlgebra

julia> rho = state_phiplus(Float64) # two-qubit maximally entangled state
4×4 Hermitian{Float64, Matrix{Float64}}:
 0.5  0.0  0.0  0.5
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0
 0.5  0.0  0.0  0.5

julia> φ = (1 + √5) / 2;

julia> v = [0 1 φ; 0 1 -φ; 1 φ 0; 1 -φ 0; φ 0 1; φ 0 -1] / sqrt(2 + φ) # Bloch vectors forming an icosahedron
6×3 Matrix{Float64}:
 0.0        0.525731   0.850651
 0.0        0.525731  -0.850651
 0.525731   0.850651   0.0
 0.525731  -0.850651   0.0
 0.850651   0.0        0.525731
 0.850651   0.0       -0.525731

julia> povm_dichotomic(A) = [A, I - A]

julia> mes = povm_dichotomic.(bloch_operator.(eachrow(v)));

julia> p = tensor_correlation(rho, mes, 2; marg=false)
6×6 Matrix{Float64}:
  0.447214  -1.0       -0.447214   0.447214   0.447214  -0.447214
 -1.0        0.447214  -0.447214   0.447214  -0.447214   0.447214
 -0.447214  -0.447214  -0.447214   1.0        0.447214   0.447214
  0.447214   0.447214   1.0       -0.447214   0.447214   0.447214
  0.447214  -0.447214   0.447214   0.447214   1.0        0.447214
 -0.447214   0.447214   0.447214   0.447214   0.447214   1.0

julia> lower_bound, upper_bound, local_model, bell_inequality = nonlocality_threshold(p);

julia> println([lower_bound, upper_bound])
[0.778, 0.779]

julia> final_iterate = sum(weight * atom for (weight, atom) in local_model);

julia> norm(final_iterate - lower_bound * p) < 1e-3 # checking local model
true

julia> local_bound_correlation(bell_inequality)[1] / dot(bell_inequality, p) # checking the Bell inequality
0.7785490499446976
```

## Under the hood

The computation is based on an efficient variant of the Frank-Wolfe algorithm to iteratively find the local point closest to the input correlation tensor.
See this [review](https://arxiv.org/abs/2211.14103) for an introduction to the method and the package [FrankWolfe.jl](https://github.com/ZIB-IOL/FrankWolfe.jl) for the implementation on which this package relies.

In a nutshell, each step gets closer to the objective point:
* either by moving towards a *good* vertex of the local polytope,
* or by astutely combining the vertices (or atoms) already found and stored in the *active set*.

```julia
julia> res = bell_frank_wolfe(p; v0=0.8, verbose=3, callback_interval=10^2, mode_last=-1);
   #Inputs: 6
 Symmetric: true
 Dimension: 21
Visibility: 0.8
   Iteration        Primal      Dual gap    Time (sec)       #It/sec    #Atoms       #LMO
         100    7.5008e-03    1.2306e-01    7.0849e-03    1.4114e+04        15         23
         200    1.8452e-03    4.2474e-02    9.2252e-03    2.1680e+04        14         27
         300    1.8093e-03    2.2514e-07    1.2580e-02    2.3847e+04        14         36
        Last    1.8093e-03    7.6190e-08    1.2739e-02    2.3786e+04        14         37
        Last    1.8093e-03    7.6190e-08    1.3419e-02    2.2580e+04        14         38
v_c ≤ 0.778392
```

## Going further

More examples can be found in the corresponding folder of the package.
They include the construction of a Bell inequality with a higher tolerance to noise as CHSH as well as multipartite and high-dimensional instances.

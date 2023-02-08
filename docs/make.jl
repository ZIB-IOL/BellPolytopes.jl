using BellPolytopes
using Documenter

DocMeta.setdocmeta!(BellPolytopes, :DocTestSetup, :(using BellPolytopes); recursive=true)

makedocs(;
    modules=[BellPolytopes],
    authors="Mathieu Besançon <mathieu.besancon@gmail.com> and contributors",
    repo="https://github.com/matbesancon/BellPolytopes.jl/blob/{commit}{path}#{line}",
    sitename="BellPolytopes.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://matbesancon.github.io/BellPolytopes.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/matbesancon/BellPolytopes.jl",
    devbranch="main",
)

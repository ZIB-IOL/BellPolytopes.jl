using BellPolytopes
using Documenter

DocMeta.setdocmeta!(BellPolytopes, :DocTestSetup, :(using BellPolytopes); recursive=true)

makedocs(;
    modules=[BellPolytopes],
    authors="ZIB AISST",
    repo="https://github.com/ZIB-IOL/BellPolytopes.jl/blob/{commit}{path}#{line}",
    sitename="BellPolytopes.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://zib-iol.github.io/BellPolytopes.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ZIB-IOL/BellPolytopes.jl",
    devbranch="main",
)

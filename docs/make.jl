using BellPolytopes
using Documenter

DocMeta.setdocmeta!(BellPolytopes, :DocTestSetup, :(using BellPolytopes); recursive = true)

generated_path = joinpath(@__DIR__, "src")
base_url = "https://github.com/ZIB-IOL/BellPolytopes.jl/blob/main/"
isdir(generated_path) || mkdir(generated_path)

open(joinpath(generated_path, "index.md"), "w") do io
    # Point to source license file
    println(
        io,
        """
        ```@meta
        EditURL = "$(base_url)README.md"
        ```
        """,
    )
    # Write the contents out below the meta block
    for line in eachline(joinpath(dirname(@__DIR__), "README.md"))
        println(io, line)
    end
end

makedocs(;
    modules = [BellPolytopes],
    authors = "ZIB AISST",
    repo = "https://github.com/ZIB-IOL/BellPolytopes.jl/blob/{commit}{path}#{line}",
    sitename = "BellPolytopes.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://zib-iol.github.io/BellPolytopes.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = ["Home" => "index.md", "API reference" => "api.md"],
)

deploydocs(; repo = "github.com/ZIB-IOL/BellPolytopes.jl", devbranch = "main", push_preview = true)

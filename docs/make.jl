using DistCtrl4DistMan
using Documenter

makedocs(;
    modules=[DistCtrl4DistMan],
    authors="Martin Gurtner",
    repo="https://github.com/martingurtner/DistCtrl4DistMan.jl/blob/{commit}{path}#L{line}",
    sitename="DistCtrl4DistMan.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://martingurtner.github.io/DistCtrl4DistMan.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/martingurtner/DistCtrl4DistMan.jl",
)

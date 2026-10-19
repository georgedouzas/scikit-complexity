# Proposal

## Why

Thurner et al. (2018 §3.5.2) and Trzaska (2023 §3.5) build the Koch curve, the Cantor set, and the Sierpiński
triangle and carpet by substitution: each generation replaces every segment or polygon by scaled copies of itself.
The inventory records these three models as out of scope, "Geometric substitution, not dynamics". The user directive
puts them in scope, and they are dynamics: a rewriting system is a discrete-time map whose state is a word, a set of
tiles, or a graph, and whose generations are its steps. The same capability covers L-systems for plant morphology
(Prusinkiewicz and Lindenmayer 1990), formal grammars, substitution tilings, and graph rewriting for chemistry and
networks. The framework has no state whose size changes from step to step other than agents, and no law that replaces
one part of a state by several parts.

## What Changes

- Add rewriting systems as a form. A rewriting element holds a structure, a word of symbols, a patch of tiles, or a
  host graph, and productions replace its parts in parallel once per generation. One generation is one step of the
  simulation, so `run(until=n)` gives the generations `0` to `n`.
- Add string rewriting in the style of L-systems: deterministic, stochastic, context-sensitive with ignored symbols,
  bracketed, and parametric productions, with conditions and successor parameters written as Sage expressions and
  lowered to NumPy like every other law.
- Add turtle interpretation of words in two and three dimensions, with branching through brackets, giving a geometry
  of line segments per generation.
- Add geometric substitution of segments and polygons by affine maps, which covers Cantor-like sets, Koch-like curves,
  Sierpiński-like gaskets, and substitution tilings such as the chair and Penrose tilings.
- Add graph rewriting by double-pushout productions, matched by subgraph monomorphism.
- Keep the fixed-size part of the state in the state vector: the count of every symbol or prototile and the size of
  the structure. Record the structure and its geometry per frame next to the vector, as agents already are.
- Add geometric measures as observables: total length, area, number of parts, bounding box, and a raster of the
  geometry that `box_counting_dimension` reads.
- Add a geometry visual that logs curves and polygon outlines as line strips and filled patches as an image.
- Add `shapely` as a runtime dependency for the geometric measures, densification, rasterisation, and unions.
- Add the models `koch_curve`, `cantor_set`, and `sierpinski_substitution` in `mathematics/fractals`, each with a
  gallery example, a Rerun thumbnail, and an analysis plot.

## Capabilities

### New Capabilities

- `rewriting-systems`: rewriting elements, string productions, turtle interpretation, geometric substitution, graph
  productions, the rewriting form and its stepper, and geometric measures.

### Modified Capabilities

- `simulation`: frames and trajectories carry the structure of rewriting elements per generation.
- `rerun-visualization`: a geometry visual draws curves, outlines, and filled patches.
- `model-library`: the Koch curve, the Cantor set, and the Sierpiński substitutions.

## Impact

- `skcomplex.core.model` gains `Word`, `Module`, `Production`, `Turtle`, `Patch`, `Prototile`, `Affine`,
  `Substitution`, `HostGraph`, and `GraphProduction`. `skcomplex.core.base` gains `Form.REWRITING` and
  `RewritingError`, and `skcomplex.core.solvers` gains the `Rewriter` stepper.
- `skcomplex.core.runtime` gains `Frame.structures` and `Trajectory.structure`, and `skcomplex.core.viz` gains
  `GeometryVisual`.
- `pyproject.toml` gains `shapely>=2.1`, which is BSD-3-Clause with wheels for every supported platform.
- The inventory moves `koch_curve`, `cantor_set`, and `sierpinski_substitution` from out of scope to pending this
  change, and the `organize-model-library` design's list of models with no planned change loses geometric
  substitution fractals.
- Stochastic productions draw from the simulation's random stream that `add-stochastic-maps` shares, and use the
  simulation's generator directly until that change lands.

# Design

## Context

The parts of the framework this change builds on:
- A system compiles to a `Problem` with a fixed-length state vector, and the law kinds decide its form. `Update` laws
  give a discrete map, and stepwise forms take an integer `dt`.
- Agent-based models already keep a structure of changing size outside the vector. Each frame carries the agents as
  snapshots, and the trajectory returns them per frame, next to the vector's states.
- Observables reduce a frame to values, and `box_counting_dimension` in `skcomplex.core.analysis` reads a boolean grid.
- Laws are Sage expressions printed to NumPy and Numba, and `Call` runs Python or Numba functions inside laws.
- The catalog runs factories with `create` and `run`, and the gallery needs a Rerun screenshot and an analysis figure
  per model within 300 frames and 20 MB.

A rewriting system does not fit the vector alone. The Koch word has `4^n` symbols at generation `n`, so the state's
size changes every step. It does fit the frame model, since a generation is one application of a map.

## Goals / Non-Goals

**Goals:**

- Run rewriting systems through `Simulation`, `create`, and `run`, with generations as steps, and the same observers,
  experiments, and Rerun streaming as every other form.
- Cover the L-system family of Prusinkiewicz and Lindenmayer (1990): D0L, stochastic, context-sensitive (2L and IL),
  bracketed, and parametric systems.
- Cover affine substitution of segments and polygons, which gives the classical fractals and substitution tilings.
- Cover graph rewriting with double-pushout productions.
- Keep the core domain-agnostic. A word is a plant, a sentence of a grammar, or a polymer, and a host graph is a
  molecule or a network.

**Non-Goals:**

- Interactive plant modelling with surfaces, tropisms, and environmental queries as in L-Py. The turtle draws segments
  only, and environment coupling goes through co-simulation.
- Sequential grammars, such as Chomsky derivations with one rewrite per step. They are a different semantics, and a
  parallel production with a single match covers the examples the references use.
- Graph rewriting in the single-pushout and sesqui-pushout approaches, and attributed graph grammars with typed
  attributes beyond labels.
- Continuous-time rewriting, such as rule-based chemistry in Kappa. It is a jump process over a graph and belongs with
  the stochastic simulation algorithms.

## Decisions

### How a rewriting system fits System, Simulation, and Trajectory

The fit is faithful, and it forces no time axis:
- A rewriting element, `Word`, `Patch`, or `HostGraph`, holds an initial structure. Interactions return production
  laws that target it, exactly as other interactions return `Update` laws.
- `compile_system` infers `Form.REWRITING = 'rewriting system'` from the law kinds. A system that mixes rewriting
  laws with other kinds is a co-simulation, and the rewriting partition reads the other partitions' values as
  parameters held for one generation.
- The state vector holds what has a fixed size: the count of every symbol or prototile and the structure size, under
  `<element>.<symbol>` and `<element>.size`. These are the Parikh vector of formal language theory, and they are
  ordinary states for trajectories, observables, and analyses.
- The structure itself, with its geometry, lives next to the vector in `Frame.structures`, as agent snapshots do. The
  `Rewriter` stepper owns it, and `Trajectory.structure(path, frame)` returns the recorded copy.
- One step is one generation. `dt` is a positive integer number of generations, `time` is the generation, and
  `run(until=n)` records the generations `0` to `n`.
- Parameters of productions, such as an angle or a contraction factor, are ordinary parameters, so `schedule` and
  `update_parameters` change them between generations.

Alternatives considered:
- A fixed-capacity vector of symbol codes padded to the largest generation. It wastes `4^n` slots from the first frame
  and does not hold parameters, tiles, or graphs.
- Keeping only the final structure as the result, like a solver's output. It loses the generations, which are what
  the references show and what the gallery animates.

### Library survey

Checked on 2026-10-19 against PyPI and the projects' repositories:

| Library | Version (date) | Licence | Wheels for Linux and macOS, 3.12–3.13 | Maintenance | Scope | Decision |
|---|---|---|---|---|---|---|
| L-Py (`openalea.lpy`) | conda only | CeCILL, GPL compatible, the README badge says GPL-3.0 | none on PyPI, conda channel `openalea3` | active | full L-system language with Python productions, PlantGL rendering, Qt editor | reject |
| `lpy` on PyPI | 0.9.2.2 | GPL-2 | none | unrelated package of the same name | | reject |
| `lsystem` | 1.0 | GPL | none, sdist only | inactive | D0L strings | reject |
| `lsystems` | 0.1.0 (2014-08-12) | BSD | none, sdist only | inactive | D0L strings and turtle | reject |
| `lfract` | 1.0.0 (2021-08-20) | MIT | pure Python | inactive | command line images of deterministic, context-free systems | reject |
| `LSystemRenderer` | 0.0.9 (2022-10-18) | MIT | pure Python | inactive, pins about 30 dependencies such as `black==22.10.0` | rendering of stochastic systems | reject |
| `ulsys` and other GitHub projects | none | various | none | pre-alpha | stochastic strings | reject |
| Shapely | 2.2.0 (2026-10-07) | BSD-3-Clause | cp312 and cp313 for manylinux x86_64 and aarch64, and macOS x86_64 and arm64 | very active, GEOS based | vectorised geometry arrays, length, area, unions, densification, point-in-polygon | adopt as a runtime dependency |
| NetworkX | 3.7 (2026-09-21) | BSD-3-Clause | pure Python | very active, already a dependency | subgraph monomorphism with `DiGraphMatcher` and `GraphMatcher` | adopt for graph matching |
| ReGraph | 2.0.1 (2020-03-31) | MIT | pure Python | inactive since 2020, requires `neo4j-driver`, `sympy`, and `greenery` | sesqui-pushout rewriting on NetworkX and Neo4j | reject |
| GrandIso | 2.2.0 (2024-02-13) | MIT | sdist only | low activity | queue-based subgraph monomorphism | reject, NetworkX covers it |
| NumPy and Numba | already dependencies | BSD | yes | very active | vectorised gathers and compiled stack walks | adopt for the string rewriting and the turtle |

No Python library implements parallel string rewriting with stochastic, context-sensitive, and parametric productions
and is mature, permissively licensed, and on PyPI with wheels. L-Py is the reference implementation, but it is
copyleft, it ships only on conda, and its productions are Python code, not symbolic laws. As an optional extra it would
not be installable with `pip`, so the rule for copyleft extras cannot apply. The JavaScript `lindenmayer` package
covers the family but is not Python. The string rewriting and the turtle are therefore implemented in the core, as a
small adapter of vectorised NumPy and Numba kernels, and the constitution's record of rejected libraries is the table
above.

Geometry takes Shapely. Its vectorised functions over arrays of geometries give the measures and the raster:
`shapely.length`, `shapely.area`, `shapely.union_all`, `shapely.segmentize` with `shapely.get_coordinates` for
rasterising curves, and `shapely.contains_xy` for rasterising filled patches. Graph matching takes NetworkX, and the
core holds only the double-pushout gluing on top of it. Rerun draws the geometry with `LineStrips2D`, `LineStrips3D`,
and `Image`, which the viewer already batches.

### String rewriting

A word is two arrays: integer symbol codes of length `m`, and a float parameter table of shape `(m, p)` padded with
NaN, where `p` is the largest module arity. The alphabet is declared, either as a string whose characters are symbols
or as `Module` names with arities.

```python
from skcomplex.core.model import Interaction, Module, Production, Symbols, Turtle, Word

x, R = Symbols('x R')
word = Word('plant', axiom=[Module('A', 1.0)], alphabet={'A': 1, 'F': 1, '+': 0, '-': 0, '[': 0, ']': 0},
            turtle=Turtle(angle=25.0, step=1.0))
laws = [Production(Module('A', x), [Module('F', x), '[', '+', Module('A', x * R), ']', Module('A', x * R)],
                   condition=x > 0.1)]
```

One generation runs in four vectorised passes:
1. Match. Each production gives a boolean mask over the word: the predecessor code, the condition lowered from Sage to
   NumPy over the parameter columns, and the context. A context-sensitive production ignores the declared symbols and
   finds a symbol's bracket-aware neighbours with a Numba stack walk that runs once per generation.
2. Choose. Productions are ranked, context and condition first. For stochastic productions of one symbol, one uniform
   draw per symbol picks a production from the cumulative probabilities, which the compiler checks sum to one.
3. Size. The successor length per symbol, a cumulative sum, and the size guard, which raises `RewritingError` before
   allocating a word beyond `max_size`.
4. Write. `np.repeat` and a gather from a table of successor codes write the new codes, and the successor parameter
   expressions, lowered once, fill the new parameter table for the matched rows.

No pass loops over symbols in Python. A benchmark test rewrites a Koch word of 10⁶ symbols and sets the target of
one generation at under 0.1 s on the reference machine.

### Turtle

`Turtle(angle, step, contraction=1.0, symbols=None)` maps symbols to actions: draw forward (`F`), move forward (`f`),
turn (`+`, `-`), turn back (`|`), pitch, roll, and yaw in three dimensions (`&`, `^`, `\`, `/`), and push and pop
(`[`, `]`). `symbols` remaps them for a model's own alphabet. Without brackets the headings are a cumulative sum of
turns and the positions a cumulative sum of steps, all in NumPy. With brackets or in three dimensions a Numba kernel
walks the word with a stack of positions and frames. The step at generation `n` is `step · contraction^n`, so the
Koch curve keeps a unit span. The result is a segment array of shape `(k, 2, d)` with each segment's branch depth.

### Geometric substitution

A `Patch` holds tiles as a prototile code per tile and a stack of affine matrices of shape `(t, d+1, d+1)`. A
`Prototile` is a segment or a polygon given by its vertices in its own frame. A `Substitution(prototile, pieces)` law
lists the pieces as pairs of a prototile and an `Affine` map, built from a scale, a rotation, a reflection, and a
translation, or given as a matrix. One generation composes every tile's matrix with its pieces' matrices by one
batched product, `np.einsum('tij,pjk->tpik', ...)`, and keeps tiles whose prototile has no substitution. The geometry
of a tile is its prototile's vertices under its matrix. This one mechanism gives the Cantor set, the Sierpiński triangle
and carpet, Koch curves by segment substitution, and the chair, sphinx, and Penrose tilings.

### Graph rewriting

A `HostGraph` wraps a NetworkX graph with node and edge labels. A `GraphProduction(left, interface, right)` gives the
three graphs, with the interface's nodes shared by name. Matching uses NetworkX's `GraphMatcher` or `DiGraphMatcher`
for subgraph monomorphisms that respect labels. The stepper orders matches with a permutation from the simulation's
random stream, keeps a maximal set of parallel independent matches greedily, skips matches that fail the dangling
condition, and then applies the deletions and the gluing of the right graph for all kept matches at once. Node counts
by label are states, as symbol counts are for words.

### Observables and analysis

`length`, `area`, `parts`, `bounds`, and `raster(resolution)` are observables of rewriting elements. `raster` turns
segments into points with `shapely.segmentize` and `shapely.get_coordinates` and bins them, or tests pixel centres with
`shapely.contains_xy` for filled polygons. The result is the boolean grid that `box_counting_dimension` already reads,
so no new analysis function is needed for the fractal dimension.

### Compile pipeline

Conditions, successor parameters, probabilities, angles, and contraction factors are Sage expressions or numbers, and
they are lowered with the existing printers. The NumPy backend evaluates them over the parameter columns, and the Numba
backend compiles the stack walks and the context search. A production's laws carry their equations for
`equations_of`, written as `A(x) → F(x) A(x·R) : x > 0.1`, so the field pages show them like other laws.

### Visualization and runtime

`GeometryVisual` is registered as `geometry` and is the default visual of the rewriting elements. It logs one
`LineStrips2D` or `LineStrips3D` batch per frame, coloured by prototile or by branch depth, and for filled patches an
`Image` of the raster. It fixes the view's bounds to the bounding box of the run, so generations grow inside a still
frame. Symbol counts stream as scalars like other states. The default blueprint's primary view is the geometry view,
which the gallery screenshots at the last frame.

### Library models

The three models live in `skcomplex.models.mathematics.fractals`:
- `koch_curve(angle=60.0)` is the L-system `F → F+F--F+F` with a turtle of contraction `1/3`, and
  defaults to `until=5`.
- `cantor_set()` is the unit segment with two pieces of scale `1/3`, and defaults to `until=8`.
- `sierpinski_substitution(shape='triangle')` is the triangle with three half-scale corner pieces, or the square with
  eight third-scale pieces for `shape='carpet'`, and defaults to `until=6` and `until=4`.

Each gallery example runs its model within the budget, `4^6 = 4096` Koch segments at most, takes the thumbnail from
the geometry view at the last frame, and plots the log measure against the generation and the box-counting fit against
the similarity dimension.

## Risks / Trade-offs

- [Exponential growth exhausts memory] → `max_size` defaults to 10⁷ parts, and the size pass raises before allocating.
- [In-house L-system code] → It is kept to vectorised array passes and two Numba kernels, and its tests reproduce the
  worked examples of Prusinkiewicz and Lindenmayer (1990), so the semantics match the reference implementation.
- [Recordings grow with the generations] → The gallery defaults stop at a few thousand segments, well under 20 MB.
- [Graph matching is NP-hard in general] → Productions in the references have small left graphs, and the stepper takes
  a `max_matches` bound per generation.
- [Shapely is a new runtime dependency] → It is BSD-3-Clause, ships wheels for every supported platform, and replaces
  the geometry code the measures would otherwise need.

# Rewriting systems

Grow a structure generation by generation, as an L-system grows a plant or a substitution tiling grows a fractal. A
rewriting system replaces its parts in parallel at every step.

```python
from skcomplex.core.components.elements import WordElement
from skcomplex.core.components.laws import ProductionLaw
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate


class AlgaeElement(WordElement):
    def _get_laws(self, s):
        return [ProductionLaw('A', 'AB'), ProductionLaw('B', 'A')]


algae = System(elements=[AlgaeElement('algae', axiom='A')])
trajectory = simulate(algae, until=5)
trajectory.values('algae.size').tolist(), trajectory.structure('algae', 5).text
```

There are two kinds of rewritten element:

- A `WordElement` is a string of symbols that grows by string productions, in the style of L-systems.
- A `PatchElement` is a set of tiles that grows by affine substitutions.

`simulate` compiles a system of words or patches to the rewriting form, and every step is one generation.
`simulate(system, until=n)` records the generations `0` to `n`. The structure itself lives next to the state vector. The states are the count of every symbol or prototile
that is a word, and the size `<element>.size`. `Trajectory.structure(path, frame)` returns the recorded structure of a
frame, which later generations do not change.

## Words and productions

Write the rules of a word as productions, and every generation applies them all at once. A word or a patch rewrites
itself by the rewriting laws it returns from `_get_laws`, a `ProductionLaw` for every rule of a word. Strings are words
of one-character symbols, and `ModuleSpec` gives a symbol parameters.

```python
from typing import Annotated

from sage.all__sagemath_symbolics import var

from skcomplex.core.components.elements import WordElement
from skcomplex.core.components.laws import ProductionLaw
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ModuleSpec, ParameterSpec

x = var('x')


class ShrinkingElement(WordElement):
    def __init__(self, label, axiom=(), *, alphabet=None, ratio: Annotated[float, ParameterSpec()] = 0.5):
        super().__init__(label, axiom, alphabet=alphabet)
        self.ratio = ratio

    def _get_laws(self, s):
        successor = [ModuleSpec('F', (x,)), ModuleSpec('A', (s.ratio * x,))]
        return [ProductionLaw(ModuleSpec('A', (x,)), successor, condition=x > 0.1)]


plant = System(elements=[ShrinkingElement('plant', axiom=[ModuleSpec('A', (1.0,))], alphabet={'A': 1, 'F': 1})])
simulate(plant, until=5).structure('plant', 5).text
```

A production can take several forms:

- **Stochastic.** Its probabilities sum to one per predecessor and context, and it draws from the random stream of the run.
- **Context-sensitive.** It depends on a left or a right context. The context skips the word's `ignore` symbols and
  treats bracketed branches as Prusinkiewicz and Lindenmayer (1990) do. A context-sensitive production takes
  precedence over context-free productions.
- **Parametric.** It binds the parameters of the predecessor to symbols, such as `x` above. Its condition and the
  parameters of its successor are expressions of these symbols and of the parameters of the rewritten element.
  A `schedule` changes those parameters between generations.

A word element with parameters declares them as annotated arguments of its `__init__`, as `ratio` above.

A generation that would make a word longer than `max_size` raises `RewritingError` before allocating it. The frames
before it stay recorded.

## Word geometries, patches, and measures

Draw a word as a curve, or grow a patch of tiles, and measure the result as a fractal.

```python
from skcomplex.core.analysis import scaling
from skcomplex.core.components.elements import PatchElement, WordElement
from skcomplex.core.components.laws import ProductionLaw, SubstitutionLaw
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import AffineSpec, GeometrySpec, PrototileSpec


class KochElement(WordElement):
    def _get_laws(self, s):
        return [ProductionLaw('F', 'F+F--F+F')]


koch = System(elements=[KochElement('koch', axiom='F', geometry=GeometrySpec(angle=60.0, contraction=1 / 3))])
curve = simulate(koch, until=4).structure('koch', 4)
len(curve.segments()), round(curve.length(), 6)
```

A `GeometrySpec`, from `skcomplex.core.specs`, draws a word as line segments in two or three dimensions, as the turtle
interpretation of L-systems does:

- `F` draws forward, and `f` moves forward without drawing.
- `+` and `-` turn by the angle.
- `[` and `]` push and pop a branch.
- `contraction` scales the step at every generation.

A `PatchElement` holds tiles. Each tile is a `PrototileSpec`, a segment or a polygon, under an affine placement. A
`SubstitutionLaw` replaces every tile of a prototile by pieces, each placed by an `AffineSpec` map relative to it. The
Cantor set below keeps the outer thirds of every interval.

```python continuation
interval = PrototileSpec('interval', [(0, 0), (1, 0)])
thirds = [('interval', AffineSpec(scale=1 / 3)), ('interval', AffineSpec(scale=1 / 3, translation=(2 / 3, 0)))]


class CantorElement(PatchElement):
    def _get_laws(self, s):
        return [SubstitutionLaw('interval', thirds)]


cantor = System(elements=[CantorElement('cantor', prototiles=[interval])])
trajectory = simulate(cantor, until=5)
trajectory.values('cantor.size')[-1], round(trajectory.values('cantor.length')[-1], 6)
grid = trajectory.structure('cantor', 5).raster(729)
round(scaling(grid, kind='box-counting', scales=[3, 9, 27]).exponent, 2)
```

Structures with a geometry report `<element>.length` and `<element>.area` at every frame. They also give `bounds()`, and
`raster(resolution)`, a boolean grid that `scaling(grid, kind='box-counting')` reads, as above.

The `geometry` visual logs segments and outlines as line strips, and a patch with `filled=True` as an image of its
raster. It draws within the bounds the element declares, or else within the bounds of its first tiles.

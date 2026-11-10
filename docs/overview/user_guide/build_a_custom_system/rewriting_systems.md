# Rewriting systems

This page grows a structure generation by generation, as an L-system grows a plant or a substitution tiling grows a
fractal. A rewriting system replaces all its parts in parallel at every step, so one generation rewrites every symbol
of a word or every tile of a patch at once.

## A first rewriting system

This section grows the algae of Lindenmayer, a word over the symbols `A` and `B` where every `A` becomes `AB` and every
`B` becomes `A`. A [`WordElement`](../core_components/elements.md#wordelement) holds the word, and its `_get_laws`
returns one [`ProductionLaw`](../core_components/laws.md#productionlaw) per rule. The
[`CompositeSystem`](../core_components/systems.md#compositesystem) runs five generations from the axiom `A`, the word of
generation 0.

```python
from skcomplex.core.components.elements import WordElement
from skcomplex.core.components.laws import ProductionLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate


class AlgaeElement(WordElement):
    def _get_laws(self, s):
        return [ProductionLaw('A', 'AB'), ProductionLaw('B', 'A')]


algae = CompositeSystem(elements=[AlgaeElement('algae', axiom='A')])
trajectory = simulate(algae, until=5)
trajectory.get_values('algae.size').tolist(), trajectory.get_structure('algae', 5).text
```

The lengths of the words are 1, 2, 3, 5, 8, and 13, the Fibonacci numbers, and the word of generation 5 is
`ABAABABAABAAB`.

There are two kinds of rewritten element.

- A `WordElement` is a string of symbols that grows by string productions, in the style of L-systems.
- A [`PatchElement`](../core_components/elements.md#patchelement) is a set of tiles that grows by affine
  substitutions.

`simulate` compiles a system of words or patches to the rewriting form, and every step is one generation.
`simulate(system, until=n)` records the generations `0` to `n`. The structure itself lives next to the state vector.
The states are the count of every symbol or prototile that is a word, and the size `<element>.size`.
`Trajectory.get_structure(path, frame)` returns the recorded structure of a frame, which later generations do not
change.

## Words and productions

This section writes the rules of a word as productions with parameters and conditions. A word or a patch rewrites
itself by the rewriting laws it returns from `_get_laws`, a `ProductionLaw` for every rule of a word. Strings are words
of one-character symbols, and a `ModuleSpec` gives a symbol parameters, such as the length of a segment. Here an apex
`A(x)` lays down a segment `F(x)` and a new apex of half the length, as long as `x` exceeds 0.1. The ratio one half is
a parameter of the element.

```python
from typing import Annotated

from sage.all__sagemath_symbolics import var

from skcomplex.core.components.elements import WordElement
from skcomplex.core.components.laws import ProductionLaw
from skcomplex.core.components.systems import CompositeSystem
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


plant = CompositeSystem(elements=[ShrinkingElement('plant', axiom=[ModuleSpec('A', (1.0,))], alphabet={'A': 1, 'F': 1})])
simulate(plant, until=5).get_structure('plant', 5).text
```

After five generations the word is `F(1) F(0.5) F(0.25) F(0.125) A(0.0625)`. The apex stops at 0.0625, since the
condition `x > 0.1` fails there.

A production can take several forms.

- **Stochastic.** Its probabilities sum to one per predecessor and context, and it draws from the random stream of
  the run.
- **Context-sensitive.** It depends on a left or a right context. The context skips the `ignore` symbols of the word
  and treats bracketed branches as Prusinkiewicz and Lindenmayer (1990) do. A context-sensitive production takes
  precedence over context-free productions.
- **Parametric.** It binds the parameters of the predecessor to symbols, such as `x` above. Its condition and the
  parameters of its successor are expressions of these symbols and of the parameters of the rewritten element. A
  `schedule` changes those parameters between generations.

A word element with parameters declares them as annotated arguments of its `__init__`, as `ratio` above. A generation
that would make a word longer than `max_size` raises `RewritingError` before allocating it. The frames before it stay
recorded.

## Word geometries

This section draws a word as a curve and measures it. A `GeometrySpec` of `skcomplex.core.specs` draws a word as line
segments in two or three dimensions, as the turtle interpretation of L-systems does. The Koch curve replaces every
segment by four segments of a third of its length, turning by 60 degrees.

```python
from skcomplex.core.analysis import scaling
from skcomplex.core.components.elements import PatchElement, WordElement
from skcomplex.core.components.laws import ProductionLaw, SubstitutionLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import AffineSpec, GeometrySpec, PrototileSpec


class KochElement(WordElement):
    def _get_laws(self, s):
        return [ProductionLaw('F', 'F+F--F+F')]


koch = CompositeSystem(elements=[KochElement('koch', axiom='F', geometry=GeometrySpec(angle=60.0, contraction=1 / 3))])
curve = simulate(koch, until=4).get_structure('koch', 4)
len(curve.get_segments()), round(curve.get_length(), 6)
```

After four generations the curve has 256 segments and the length 3.160494, which is `(4/3)^4`. The symbols of the
turtle have fixed meanings.

- `F` draws forward, and `f` moves forward without drawing.
- `+` and `-` turn by the angle.
- `[` and `]` push and pop a branch.
- `contraction` scales the step at every generation.

## Patches and measures

This section grows a patch of tiles and measures it as a fractal. A `PatchElement` holds tiles. Each tile is a
`PrototileSpec`, a segment or a polygon, under an affine placement. A
[`SubstitutionLaw`](../core_components/laws.md#substitutionlaw) replaces every tile of a prototile by pieces, each
placed by an `AffineSpec` map relative to it. The Cantor set below keeps the outer thirds of every interval.

```python continuation
interval = PrototileSpec('interval', [(0, 0), (1, 0)])
thirds = [('interval', AffineSpec(scale=1 / 3)), ('interval', AffineSpec(scale=1 / 3, translation=(2 / 3, 0)))]


class CantorElement(PatchElement):
    def _get_laws(self, s):
        return [SubstitutionLaw('interval', thirds)]


cantor = CompositeSystem(elements=[CantorElement('cantor', prototiles=[interval])])
trajectory = simulate(cantor, until=5)
trajectory.get_values('cantor.size')[-1], round(trajectory.get_values('cantor.length')[-1], 6)
```

After five generations the patch holds 32 intervals of total length 0.131687, which is `(2/3)^5`. Structures with a
geometry report `<element>.length` and `<element>.area` at every frame. They also give `get_bounds()`, and
`get_raster(resolution)`, a boolean grid that the analysis `scaling` reads. The box-counting dimension counts the boxes
of each size that the set touches, and its slope against the size on a log scale estimates the fractal dimension.

```python continuation
grid = trajectory.get_structure('cantor', 5).get_raster(729)
round(scaling(grid, kind='box-counting', scales=[3, 9, 27]).exponent, 2)
```

The estimate is 0.65, close to the dimension `log 2 / log 3`, about 0.63, of the Cantor set.

The `geometry` visual logs segments and outlines as line strips, and a patch with `filled=True` as an image of its
raster. It draws within the bounds the element declares, or else within the bounds of its first tiles.

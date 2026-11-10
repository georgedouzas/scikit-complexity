# Cell complexes and link fields

This page puts states on the links and plaquettes of a lattice, or on the edges and faces of a network, as lattice
gauge theories and discrete exterior calculus do. A cell complex is a set of cells of rising degree, sites of degree 0,
links of degree 1, and plaquettes or faces of degree 2, where each cell is bounded by cells of the degree below. A
[`CellComplexSpace`](../core_components/spaces.md#cellcomplexspace) holds these cells with the oriented incidence
between them. It is a [`NetworkSpace`](../core_components/spaces.md#networkspace), so groups on its `'edges'` and
`'hyperedges'` hold states on links and plaquettes.

## Cells of lattices and networks

This section builds the complex of a torus and of a tetrahedron and reads how their cells connect. The matrix
`incidences_[k]` has one row per cell of degree `k + 1` and one column per cell of degree `k`. An entry is plus or
minus one when the lower cell lies on the boundary of the higher one, with the sign of their orientations. The torus is
a periodic [`LatticeSpace`](../core_components/spaces.md#latticespace) of 8 by 8 sites.

```python
import networkx as nx

from skcomplex.core.components.spaces import CellComplexSpace, LatticeSpace, NetworkSpace

torus = CellComplexSpace(LatticeSpace((8, 8))).prepare()
[matrix.shape for matrix in torus.incidences_], (torus.incidences_[1] @ torus.incidences_[0]).count_nonzero()
```

The torus has 64 sites, 128 links, and 64 plaquettes, so the matrices have the shapes `(128, 64)` and `(64, 128)`.
Their product has no nonzero entry, since the boundary of a boundary is empty. The complex of a network takes its faces
from one of several sources, and the cliques of the complete graph on four nodes fill a tetrahedron.

```python continuation
tetrahedron = NetworkSpace(nx.complete_graph(4))
cliques = CellComplexSpace(tetrahedron, faces='cliques', dimension=3).prepare()
cliques.incidences_[2].shape
```

The shape `(1, 4)` holds the one solid tetrahedron and its four triangles. The cells follow these rules.

- The links of a lattice point along the positive direction of their axis. Fixed and reflecting edges drop the cells
  that leave the box.
- A network takes `faces=None` for a minimum cycle basis, `'hyperedges'` for its hyperedges, `'cliques'` for its
  cliques up to `dimension`, or a sequence of node cycles.

## Exterior calculus

This section writes field equations on a complex with the operators of discrete exterior calculus. The exterior
derivative `d` maps a state on cells of one degree to the next, as the gradient maps a potential to a field on links.
The codifferential maps back down, as the divergence does. Here a potential `A` on the links of the torus follows the
lattice Maxwell equations `dA/dt = -E` and `dE/dt = codifferential(d A)`. Two
[`RateInteraction`](../core_components/interactions.md#rateinteraction)s write them on a
[`GroupElement`](../core_components/elements.md#groupelement) on the edges of the complex.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.laws.utils import codifferential, exterior_derivative
from skcomplex.core.components.spaces import CellComplexSpace, LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class FieldLinkElement(Element):
    def __init__(
        self,
        label,
        visual=None,
        *,
        A: Annotated[float, StateSpec()] = 0.0,
        E: Annotated[float, StateSpec()] = 0.0,
    ):
        super().__init__(label, visual=visual)
        self.A = A
        self.E = E


space = CellComplexSpace(LatticeSpace((8, 8))).prepare()
potential = np.random.default_rng(0).normal(size=128)
links = GroupElement('links', element=FieldLinkElement, space=space, on='edges', values={'A': potential})
interactions = [
    RateInteraction('potential', target='links', state='A', rate=lambda n: -n.target.E),
    RateInteraction('field', target='links', state='E', rate=lambda n: codifferential(exterior_derivative(n.target.A))),
]
system = CompositeSystem(elements=[links], interactions=interactions)
trajectory = simulate(system, until=2.0, dt=0.1)
bool(np.abs(space.incidences_[0].T @ trajectory.get_values('links.E').T).max() < 1e-10)
```

The block returns `True`. The divergence of the electric field at every site stays below `1e-10` over the run, so the
Gauss law holds to round-off.

The law utilities `exterior_derivative`, `codifferential`, and `hodge_star` act on the states of groups on the cells of
a complex. The degree of their argument is the part of the complex its group lives on.

- `exterior_derivative` maps to the next degree, and applied twice it is exactly 0.
- `codifferential` is `(-1)^k star^-1 d^T star` on a state of degree `k`. On an edge state it is the outflow minus the
  inflow at every site, and on a face state it is the curl on every edge. Applied twice it is 0 up to round-off.
- `hodge_star` is a diagonal factor. On a lattice it is the ratio of dual to primal cell volumes, and on a network it
  is the parameter `weight` of the group on that part of the complex.

With this sign, `codifferential(exterior_derivative(x))` is the graph Laplacian of a site state, and the curl of the
curl of an edge state. A rate of `RateInteraction` reads another group on the complex through `sources`, and the value
must have the degree of the state it changes.

## Coface reductions

This section sums over the faces around an edge, or the edges around a site. A coface of a cell is a cell of the next
degree whose boundary holds it. `neighbors(expression, over=CofaceNeighborhood())` reduces over the cofaces, with a
[`CofaceNeighborhood`](../core_components/spaces.md#cofaceneighborhood). Inside the reduction, `exterior_derivative` is
the oriented sum around the one coface, which is the holonomy of an abelian connection, and every coface counts with
weight one. The next section uses it for the Wilson action.

## Sampling links

This section samples a lattice gauge field at equilibrium, link by link. In compact U(1) gauge theory every link holds
an angle, and the Wilson action sums the cosine of the oriented sum of the angles around every plaquette. A
[`SamplingInteraction`](../core_components/interactions.md#samplinginteraction) draws new angles by Metropolis moves
against this objective, in a [`DiscreteTime`](../core_components/spaces.md#discretetime) of 200 sweeps.

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import cos, pi

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import SamplingInteraction
from skcomplex.core.components.laws.utils import exterior_derivative, neighbors
from skcomplex.core.components.spaces import CellComplexSpace, CofaceNeighborhood, DiscreteTime, LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class AngleLinkElement(Element):
    def __init__(self, label, visual=None, *, theta: Annotated[float, StateSpec(period=2 * pi)] = 0.0):
        super().__init__(label, visual=visual)
        self.theta = theta


space = CellComplexSpace(LatticeSpace((8, 8))).prepare()
links = GroupElement('links', element=AngleLinkElement, space=space, on='edges')
sampling = SamplingInteraction(
    'wilson',
    target='links',
    state='theta',
    objective=lambda n: -2.0 * neighbors(cos(exterior_derivative(n.target.theta)), over=CofaceNeighborhood()),
    move='rotate',
    width=1.5,
    schedule='colored',
)
system = CompositeSystem(elements=[links], interactions=[sampling], time=DiscreteTime())
states = simulate(system, until=200, random_state=0).get_values('links.theta')
round(float(np.cos(space.incidences_[1] @ states[-50:].T).mean()), 1)
```

The mean plaquette over the last 50 sweeps, the average cosine of the oriented sum of the angles around every face, is
0.7. The sampler follows these rules.

- `SamplingInteraction` samples a group on the edges of a cell complex with a local objective over cofaces.
- Its `'colored'` schedule updates together only links that share no face, and every draw comes from the generator of
  the simulation.
- The move is `'rotate'` for U(1), and `'random-value'` for Z_N with the `'heat-bath'` rule.

## Matrix links

This section holds a matrix of a compact group on every link, as the SU(2) and SU(3) gauge fields of the weak and
strong forces do. A `MatrixGroupSpec` names `SU(n)` or `U(n)`, and a state declared with `StateSpec(group=...)` stores
the `2 n^2` real components of every matrix. The spec turns components into matrices and back, draws matrices from the
Haar measure, the uniform distribution on the group, or near the identity, projects rounded products back onto the
group, and multiplies matrices along paths. Here 32 random SU(3) matrices sit on the links of a torus of 4 by 4 sites.

```python
import numpy as np

from skcomplex.core.components.spaces import CellComplexSpace, LatticeSpace
from skcomplex.core.specs import MatrixGroupSpec

group = MatrixGroupSpec('SU', 3)
space = CellComplexSpace(LatticeSpace((4, 4))).prepare()
links = group.get_uniform_of(np.random.default_rng(0), 32)
plaquettes = group.get_holonomy_of(links, space.boundaries_, space.boundary_signs_)
plaquettes.shape, bool(np.allclose(np.linalg.det(plaquettes), 1.0))
```

The torus has 16 plaquettes, so the holonomies have the shape `(16, 3, 3)`, and every one has the determinant 1, as an
element of SU(3) must.

- `boundaries_` lists the links around every face in the cyclic order of its nodes, and `boundary_signs_` the sign of
  their traversal. So `get_holonomy_of` gives the holonomy of every face, the ordered product of the link matrices with
  the inverse of a link traversed against its orientation. Any other path, such as a Wilson or a Polyakov loop, is a
  row of links and signs, padded with the sign 0.
- The matrices do not commute, so the holonomy depends on the order of the links. Its trace depends on the start of
  the loop only up to a cyclic shift, which leaves it unchanged.

Sampling matrix links by a class function of the holonomy of every face, such as the real part of the trace for the
Wilson action, is a recipe of physics. `WilsonSamplingInteraction` subclasses `SamplingInteraction` and reads the
holonomies of the faces around a link with `get_holonomy_of`. It proposes Metropolis or Glauber moves that rotate a
link by a random matrix of the group near the identity, or draw a new one from the Haar measure.
`LatticeYangMillsSystem` and the analysis `gauge_observables` with a `group`, both of `skcomplex.systems.physics.waves`,
realize SU(2) and SU(3) lattice gauge theory with its average plaquette, Wilson loops, and Polyakov loop.

## Analysing complexes

This section splits a link field into its gradient, curl, and harmonic parts, counts the holes of a complex, and
measures how curved a gauge field is. The Hodge decomposition writes a link state as the gradient of a site potential,
plus the codifferential of a face state, plus a harmonic part that neither produces. The Betti numbers count the
connected components, the independent loops, and the enclosed cavities. A uniform flow along the first axis of the
torus serves as the state.

```python
import numpy as np

from skcomplex.core.analysis import cell_complex
from skcomplex.core.components.spaces import CellComplexSpace, LatticeSpace

torus = CellComplexSpace(LatticeSpace((8, 8))).prepare()
uniform = np.where(torus.link_axes_ == 0, 1.0, 0.0)
hodge = cell_complex(uniform, kind='hodge', space=torus)
cell_complex(None, kind='betti', space=torus).betti, bool(np.allclose(hodge.harmonic[0], uniform))
```

The torus has the Betti numbers `(1, 2, 1)`, one component, two independent loops, and one cavity. The harmonic part
of the uniform flow equals the flow, so the flow is purely harmonic. `cell_complex` reads a state of a group on a
complex from a trajectory, or an array with the complex as `space`.

- The kind `'hodge'` gives `gradient`, `curl`, and `harmonic`, which are orthogonal and sum to the state, and the site
  potential `potential`.
- The kind `'betti'` gives `betti`, and reads the complex alone, with `None` as the source.
- The kind `'curvature'` gives `share`, the share of faces whose oriented sum of an angle state is not a multiple of
  its period, and `energy`, the sum of one minus the cosine of the sums.

`CellVisual`, named `VisualSpec('cell')`, draws edge states as coloured segments. It draws their face sums and
hyperedge states as an image on a lattice complex, or as polygons on a network complex, with a cyclic palette for
angle states.

# Cell complexes and link fields

Put states on the links and plaquettes of a lattice, or on the edges and faces of a network, as lattice gauge theories
and discrete exterior calculus do. A `CellComplexSpace` holds the sites, links, and plaquettes of a lattice, or the
nodes, edges, and faces of a network, with the oriented incidence between them. It is a `NetworkSpace`, so groups on
its `'edges'` and `'hyperedges'` hold states on links and plaquettes.

## Cells of lattices and networks

Build the complex of a torus and of a tetrahedron, and read how their cells connect:

```python
import networkx as nx

from skcomplex.core.components.spaces import CellComplexSpace, LatticeSpace, NetworkSpace

torus = CellComplexSpace(LatticeSpace((8, 8))).prepare()
[matrix.shape for matrix in torus.incidences_], (torus.incidences_[1] @ torus.incidences_[0]).count_nonzero()

tetrahedron = NetworkSpace(nx.complete_graph(4))
cliques = CellComplexSpace(tetrahedron, faces='cliques', dimension=3).prepare()
cliques.incidences_[2].shape
```

- The links of a lattice point along the positive direction of their axis. Fixed and reflecting edges drop the cells
  that leave the box.
- A network takes `faces=None` for a minimum cycle basis, `'hyperedges'` for its hyperedges, `'cliques'` for its
  cliques up to `dimension`, or a sequence of node cycles.
- `incidences_[k]` has one row per cell of degree `k + 1` and one column per cell of degree `k`, so the product of two
  consecutive ones is 0.

## Exterior calculus

Write field equations on a complex with the operators of discrete exterior calculus. Here a potential on links follows
lattice Maxwell equations, and their Gauss law holds to round-off:

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.laws.utils import codifferential, exterior_derivative
from skcomplex.core.components.spaces import CellComplexSpace, LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
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
system = System(elements=[links], interactions=interactions)
trajectory = Simulation(system, dt=0.1).prepare().run(until=2.0).trajectory_
bool(np.abs(space.incidences_[0].T @ trajectory.values('links.E').T).max() < 1e-10)
```

The law utilities `exterior_derivative`, `codifferential`, and `hodge_star` act on the states of groups on the cells of
a complex. The degree of their argument is the part its group lives on.

- `exterior_derivative` maps to the next degree, and applied twice it is exactly 0.
- `codifferential` is `(-1)^k star^-1 d^T star` on a state of degree `k`. On an edge state it is the outflow minus the
  inflow at every site, and on a face state it is the curl on every edge. Applied twice it is 0 up to round-off.
- `hodge_star` is a diagonal factor. On a lattice it is the ratio of dual to primal cell volumes, and on a network it is
  the parameter `weight` of the group on that part of the complex.

With this sign, `codifferential(exterior_derivative(x))` is the graph Laplacian of a site state, and the curl of the
curl of an edge state. A rate of `RateInteraction` reads another group on the complex through `sources`, and the value
must have the degree of the state it changes.

## Coface reductions

Sum over the faces around an edge, or the edges around a site. `neighbors(expression, over=CofaceNeighborhood())`
reduces over the faces that contain an edge, or the edges that contain a site. Inside the reduction,
`exterior_derivative` is the oriented sum around the one coface, which is the holonomy of an abelian connection, and
every coface counts with weight one. The next example uses it for the Wilson action.

## Sampling links

Sample a lattice gauge field at equilibrium, such as compact U(1) with the Wilson action, link by link:

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import cos, pi

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import SamplingInteraction
from skcomplex.core.components.laws.utils import exterior_derivative, neighbors
from skcomplex.core.components.spaces import CellComplexSpace, CofaceNeighborhood, DiscreteTime, LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
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
system = System(elements=[links], interactions=[sampling], time=DiscreteTime())
states = Simulation(system, random_state=0).prepare().run(until=200).trajectory_.values('links.theta')
round(float(np.cos(space.incidences_[1] @ states[-50:].T).mean()), 1)
```

The last line is the mean plaquette, the average cosine of the oriented sum of the angles around every face, which
lies between 0.6 and 0.8 here.

- `SamplingInteraction` samples a group on the edges of a cell complex with a local objective over cofaces.
- Its `'colored'` schedule updates together only links that share no face, and every draw comes from the generator of
  the simulation.
- The move is `'rotate'` for U(1), and `'random-value'` for Z_N with the `'heat-bath'` rule.

## Analysing complexes

Split a link field into its gradient, curl, and harmonic parts, count the holes of a complex, or measure how curved a
gauge field is:

```python
import numpy as np

from skcomplex.core.analysis import CellComplexAnalysis
from skcomplex.core.components.spaces import CellComplexSpace, LatticeSpace

torus = CellComplexSpace(LatticeSpace((8, 8))).prepare()
uniform = np.where(torus.link_axes_ == 0, 1.0, 0.0)
hodge = CellComplexAnalysis('hodge', space=torus).prepare(uniform).run()
CellComplexAnalysis('betti', space=torus).prepare().run().betti_, bool(np.allclose(hodge.harmonic_[0], uniform))
```

A uniform flow around the torus is purely harmonic, and the torus has one component, two independent loops, and one
cavity. `CellComplexAnalysis` reads a state of a group on a complex from a trajectory, or an array with the complex as
`space`.

- The kind `'hodge'` sets `gradient_`, `curl_`, and `harmonic_`, which are orthogonal and sum to the state, and the site
  potential `potential_`.
- The kind `'betti'` sets `betti_`.
- The kind `'curvature'` sets `share_`, the share of faces whose oriented sum of an angle state is not a multiple of its
  period, and `energy_`, the sum of one minus the cosine of the sums.

`CellVisual`, named `VisualSpec('cell')`, draws edge states as colored segments. It draws their face sums and
hyperedge states as an image on a lattice complex, or as polygons on a network complex, with a cyclic palette for angle
states.

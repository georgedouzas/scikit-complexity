# Phase-space fields

This page evolves a density over position and over classes such as velocity or orientation, the way a kinetic
equation does. Particles stream through position at their velocity, forces push them through velocity, and collisions
redistribute them over velocity at a fixed position.

A phase space is a lattice whose axes are the position axes followed by the class axes. A
[`FieldElement`](../core_components/elements.md#fieldelement) holds one value per point of it, and its laws are written
for one point, like those of any field.

No existing library covers this case. `py-pde` stops at three dimensions and has no class axes, `pylbm` and `lbmpy`
hold a few discrete velocities, Dedalus is GPL with no wheels, and Struphy is specific to plasmas. No maintained Python
library holds the fast spectral collision operator. So the numerics wrap `scipy.fft` and `scipy.ndimage`, and the core
holds the adapters and the assembly of the collision weights.

## Phase spaces and coordinates

This section builds a phase space from a position lattice and a velocity axis and lets a gas stream freely through it.
A [`PhaseSpace`](../core_components/spaces.md#phasespace) takes a position
[`LatticeSpace`](../core_components/spaces.md#latticespace) and a list of class axes. Each class axis is an `AxisSpec`
with a name, a number of cells, and the two ends of its interval. The gas starts as a Maxwell distribution in velocity
with a density that varies as a sine in position, and a
[`ConservationInteraction`](../core_components/interactions.md#conservationinteraction) moves it along the flux
`v f` in position.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.interactions import ConservationInteraction
from skcomplex.core.components.laws.utils import coordinate
from skcomplex.core.components.spaces import LatticeSpace, PhaseSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.core.specs import AxisSpec, StateSpec


class DensityElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.x = x


space = PhaseSpace(LatticeSpace((32,), spacing=2 * np.pi / 32), [AxisSpec('v', 64, -6.0, 6.0)]).prepare()
x, v = np.arange(32) * 2 * np.pi / 32, space.coordinates_['v']
start = np.outer(1.0 + 0.5 * np.sin(x), np.exp(-(v**2) / 2) / np.sqrt(2 * np.pi))
gas = FieldElement('gas', element=DensityElement, space=space, values={'x': start}, discretization='finite-volume')
stream = ConservationInteraction('stream', target='gas', state='x', flux=lambda s: [coordinate('v') * s.target.x, 0])
system = CompositeSystem(elements=[gas], interactions=[stream])
final = simulate(system, until=0.5, dt=0.1, solver=RK4Solver(dt=0.01)).get_values('gas.x')[-1]
abs(final.sum() / start.sum() - 1.0) < 1e-12
```

The block returns `True`. The total mass after half a time unit differs from the initial mass by less than one part in
a trillion, so streaming conserves it to round-off.

The element `DensityElement` is a subclass of [`Element`](../core_components/elements.md#element) with one state `x`,
and the system is a [`CompositeSystem`](../core_components/systems.md#compositesystem). The parts of the space follow
these rules.

- `PhaseSpace(position, axes, limit=2**26)` places the position axes first, and they keep the boundary of the lattice.
- An interval axis lets no flux through its ends. An axis built with `periodic=True`, such as an angle, joins them.
- `simulate` rejects a space above its limit with the issue `space-size`, which names the size it would take.

The function `coordinate(name)` returns the coordinate, along a named axis, of the point a law updates. Streaming is
the drift `[coordinate('v'), 0]`, and acceleration by a force `F` is the drift `[0, F]`. A law that reads a coordinate
of a group whose space lacks that axis fails `simulate` with the issue `law-space`.

Finite-volume fields stream on a phase space, and so do finite-difference and spectral fields built with
`riemann=None`. Spectral derivatives need the axis to be periodic.

## Moments and mean fields

This section reads the density, the mean velocity, or the temperature at every position and feeds them back into the
dynamics. A moment of order `n` is the integral of the density times the `n`th power of the class coordinate, taken
over the class axes at one position. A Vlasov-Poisson drift accelerates the gas by the electric field of its own
charge density, which is a moment of order zero.

```python
from skcomplex.core.components.laws.utils import coordinate, gradient, inverse_helmholtz, moment


def poisson_drift(s):
    return [coordinate('v'), -gradient(inverse_helmholtz(1 - moment(s.target.x, 0, axes=('v',))), 1)[0]]


poisson_drift.__name__
```

The block returns `'poisson_drift'`, a drift that a `ConservationInteraction` takes as its flux in place of the free
streaming above. The functions it calls follow these rules.

- `moment(expression, order, axes=names)` reduces over the named class axes at every position, with the cell widths
  as weights.
- The result is a field over position, which laws read broadcast along the reduced axes. With `normalized=True` and
  `central=True` it gives the mean velocity and the temperature at every position.
- `reduce_axes(expression, names)` is the integral behind it, and it gives the moments of `cos` and `sin` of an
  orientation.
- A drift can read a moment of its own field, and the operators of a field over position act along the position axes.

Fields over position and fields on a phase space read each other. A flux reads another field over the position lattice
through the `sources` of a `ConservationInteraction`, broadcast along the class axes. A
[`RateInteraction`](../core_components/interactions.md#rateinteraction) over the position lattice reads a moment of a
field on a phase space through its `sources`.

## Integrals over class axes

This section integrates a density against a kernel of two points of its class axes. Groups with a fraction `z` of
punishers win pairwise conflicts against groups with a fraction `u` at the probability `rho(z, u)`. So the density
changes at the rate `f(z)` times the integral of `rho(z, u) - rho(u, z)` against `f(u)`. The phase space has a single
position and one class axis `z` with 50 cells.

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import tanh

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.laws.utils import coordinate, integral_transform, integration_variable
from skcomplex.core.components.spaces import LatticeSpace, PhaseSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.core.specs import AxisSpec, StateSpec


class DensityElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.x = x


z, u = coordinate('z'), integration_variable('z')
rho = (1 + tanh(2 * (z - u))) / 2
gain = rho - rho.subs({z: u, u: z}, hold=False)
space = PhaseSpace(LatticeSpace((1,)), [AxisSpec('z', 50, 0.0, 1.0)])
groups = FieldElement('groups', element=DensityElement, space=space, values={'x': np.ones((1, 50))})
conflict = RateInteraction(
    'conflict', target='groups', state='x', rate=lambda s: s.target.x * integral_transform(s.target.x, gain, ('z',))
)
system = CompositeSystem(elements=[groups], interactions=[conflict])
trajectory = simulate(system, until=5.0, dt=1.0, solver=RK4Solver(dt=0.05))
final = trajectory.get_values('groups.x')[-1]
round(float(final.sum() / 50), 10), bool(final[-1] > final[0])
```

The block returns `(1.0, True)`. The mean density stays one, since the kernel is antisymmetric and every gain of one
group is a loss of another. The groups with the most punishers end denser than those with the fewest.

The functions of the block follow these rules.

- `integral_transform(expression, kernel, axes)` is the integral over the named class axes of `K(c, c') u(c')`, at the
  same position and the same coordinates of the other class axes. The kernel reads `coordinate(name)` for the point
  `c` that a law updates, `integration_variable(name)` for the point `c'` integrated over, and the parameters, so it
  need not be a function of `c - c'`. The compiler samples it at the cell centres, and the integral is a product with
  the dense matrix of the kernel.
- `cumulative_integral(expression, axis, reverse=False)` integrates along one axis of a lattice, from the start of the
  domain to every site, or from every site to the end with `reverse=True`, counting half of the cell of the site. The
  light under a canopy is `exp(-gamma * cumulative_integral(f * coordinate('h')**2, 1, reverse=True))`.
- `convolve` acts along the position axes of a phase space at every class, with a kernel of the displacements of the
  position axes, and it follows the boundary of the position lattice. Seeds disperse over position at every height
  this way.
- `simulate` rejects a kernel that reads a state with the issue `kernel-state`, and an integral over class axes of a
  field without a phase space with the issue `law-space`.

## Simplex domains

This section holds a density over the frequencies of three strategies, which sum to one. A simplex is the set of
points whose coordinates are nonnegative and sum to at most one. A
[`SimplexSpace`](../core_components/spaces.md#simplexspace) is a phase space whose class axes are the free frequencies
of a simplex, each on `[0, 1]` with the same number of cells. A uniform density of two on the 2-simplex, a triangle of
area one half, has the integral one.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.laws.utils import moment
from skcomplex.core.components.spaces import LatticeSpace, SimplexSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import AxisSpec, ObservableSpec, StateSpec


class DensityElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.x = x


def total(s):
    return moment(s.x, 0, axes=('x', 'y'))


axes = [AxisSpec('x', 20, 0.0, 1.0), AxisSpec('y', 20, 0.0, 1.0)]
simplex = SimplexSpace(LatticeSpace((1,)), axes).prepare()
groups = FieldElement('groups', element=DensityElement, space=simplex, values={'x': 2.0 * (simplex.volumes_ > 0)})
observables = [ObservableSpec('total', total, group='groups')]
trajectory = simulate(CompositeSystem(elements=[groups]), until=0.1, dt=0.1, observables=observables)
round(float(np.asarray(trajectory.get_values('total'))[-1]), 10)
```

The observable `total` returns `1.0`, the integral of the density over the triangle. The space weights each cell by
the part of it that lies inside the simplex.

- `volumes_` holds the fraction of every cell inside the simplex. Cells below the slanted face are whole, cells along
  it are partial, which is half a cell in two dimensions, and cells beyond it are empty. `apertures_` holds the
  fraction of every face inside the simplex.
- Finite-volume fluxes pass only through the open part of every face and change a cell by its volume. So the total is
  conserved and no density enters the empty cells. A density on a simplex starts at zero in the empty cells.
- `moment`, `reduce_axes`, `integral_transform`, and `cumulative_integral` weight every cell by its volume.
- `simulate` rejects a local operator of a field on a simplex, such as `laplacian` or a finite-difference advection,
  with the issue `law-space`. `SemiLagrangianSolver` refuses partial cells, and `SSPRKSolver` divides its wave speed
  by the smallest cell volume.

## Encounters and equilibria

This section moves particles between classes at one position and relaxes them towards a local equilibrium. The
function `maximum_entropy(expression, axes, order=2)` returns the density `exp(a + b c + d |c|^2)` whose discrete
moments equal those of the expression at every position. Newton steps find the coefficients, so a relaxation towards
it conserves the moments to round-off. This relaxation is the collision term of the BGK equation.

```python
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.laws.utils import maximum_entropy

relax = RateInteraction('r', target='gas', state='x', rate=lambda s: maximum_entropy(s.target.x, axes=('v',)) - s.target.x)
relax.state
```

The block returns `'x'`, the state of the gas that the relaxation changes. The encounters of the classes at one
position are the recipe [`EncounterInteraction`](../core_components/general.md#encounterinteraction) of the general
realizations, which writes them as bilinear laws of the core.

## Collisions

This section adds binary collisions between particles of different velocities, as in the Boltzmann equation. The
function `collision_integral(f, kernel)` of `skcomplex.core.components.laws.utils` is the binary collision integral
over two interval class axes, and a `RateInteraction` adds it to the rate of a density. The general realization
`EncounterInteraction` with a `cross_section` and a `factor` writes the same rate. The pages of the catalog listed at
the end build these collisions in full.

- The cross-section is the kernel `B(speed, cosine)` of the relative speed and of the cosine of the deflection angle.
  It returns an expression of the two and numbers, written with Sage functions.
- The rate is a [`RateLaw`](../core_components/laws.md#ratelaw) of `f` and `collision_integral(f, kernel)`, and the
  call carries its kernel. A kernel that reads a state or a parameter raises a `ValidationError`.
- The integral enters the fast spectral method of Mouhot and Pareschi through a singular value decomposition, with a
  single term for Maxwell molecules.
- The method uses products of fast Fourier transforms on a grid padded by `(3 + sqrt(2)) / 2`. It conserves the
  density exactly, and the momentum and the energy within its spectral accuracy.
- A factor that multiplies the integral scales the collision frequency, such as a function of the density moment for
  the local Enskog equation.
- `simulate` rejects a collision without two interval class axes with the issue `law-space`.

## Solvers

This section steps transport and collisions each with the method that suits it and combines the two. Transport is
fast and linear, so a semi-Lagrangian step follows it exactly. Collisions are local and stiff, so a Runge-Kutta step
with a smaller step integrates them.

```python
from skcomplex.core.engine.solvers import RK4Solver, SemiLagrangianSolver, StrangSplittingSolver

solver = StrangSplittingSolver(SemiLagrangianSolver(dt=0.1), RK4Solver(dt=0.01), dt=0.1)
solver.dt
```

The block returns `0.1`, the step of the split method. Each solver plays its own part.

- `SemiLagrangianSolver(dt)` advances fields whose fluxes are linear drifts `a u`, with `a` free of the state and
  constant along its own axis. It shifts each slice along its characteristic, in the splitting of Cheng and Knorr. It
  is stable at any Courant number and conserves the total on periodic axes. Other fluxes raise a named error.
- `StrangSplittingSolver(transport, collision, dt)` composes a half transport step, a full collision step, and a half
  transport step. The compiled problem keeps the source rates, `problem.sources`, apart from the flux differences.
- `RK4Solver`, `ScipyODESolver`, and the other method-of-lines solvers accept phase-space fields as well.

## Visualizing

`ImageVisual(slices=..., moments=...)` logs the plane that fixes every axis but two, such as `slices={'vy': 2}`. It
also logs the moments over the class axes, as a bar series over a one-dimensional position lattice and as an image
over a two-dimensional one.

## Systems of the catalog

`BoltzmannEquationSystem`, `BGKEquationSystem`, `VlasovEquationSystem`, and `EnskogSystem` in `physics.fluids`, and
`BertinDrozGregoireSystem` in `physics.active_matter`, are the kinetic equations of the library.
`NonlinearPairwiseGroupConflictSystem` and `TrimorphicMultilevelSelectionSystem` in `social.cooperation` integrate
against two-point kernels, the second on the simplex. `ForestSizeStructureSystem` in `biology.ecology` shades trees by
a cumulative integral over height.

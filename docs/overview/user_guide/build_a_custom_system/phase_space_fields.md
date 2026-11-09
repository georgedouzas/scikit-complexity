# Phase-space fields

Evolve a density over position and over classes such as velocity or orientation, the way a kinetic equation does.
Particles stream through position at their velocity, forces push them through velocity, and collisions redistribute
them over velocity at a fixed position.

A phase space is a lattice whose axes are the product of position and the class axes. So `FieldElement` holds one value
per point of it, and its laws are written for one point, like those of any field.

No existing library fits. `py-pde` stops at three dimensions and has no class axes, `pylbm` and `lbmpy` hold a few
discrete velocities, Dedalus is GPL with no wheels, and Struphy is specific to plasmas. No maintained Python library
holds the fast spectral collision operator. So the numerics wrap `scipy.fft` and `scipy.ndimage`, and the core holds
the adapters and the assembly of the collision weights.

## Phase spaces and coordinates

Build a phase space from a position lattice and a velocity axis, and let a gas stream freely through it:

```python
import numpy as np

from skcomplex.core.components.elements import FieldElement, NonnegativeElement
from skcomplex.core.components.interactions import AdvectionInteraction
from skcomplex.core.components.laws.utils import coordinate
from skcomplex.core.components.spaces import LatticeSpace, PhaseSpace
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.core.specs import AxisSpec

space = PhaseSpace(LatticeSpace((32,), spacing=2 * np.pi / 32), [AxisSpec('v', 64, -6.0, 6.0)]).prepare()
x, v = np.arange(32) * 2 * np.pi / 32, space.coordinates_['v']
start = np.outer(1.0 + 0.5 * np.sin(x), np.exp(-(v**2) / 2) / np.sqrt(2 * np.pi))
gas = FieldElement('gas', element=NonnegativeElement, space=space, values={'x': start}, discretization='finite-volume')
stream = AdvectionInteraction('stream', target='gas', state='x', drift=lambda s: [coordinate('v'), 0])
system = System(elements=[gas], interactions=[stream])
final = simulate(system, until=0.5, dt=0.1, solver=RK4Solver(dt=0.01)).values('gas.x')[-1]
abs(final.sum() / start.sum() - 1.0) < 1e-12
```

The total mass is conserved to round-off.

- `PhaseSpace(position, axes, limit=2**26)` takes a position `LatticeSpace` and the class axes.
- Each class axis is an `AxisSpec(name, size, low, high, periodic=False)` with its own cells. An interval has no flux
  through its ends, and a periodic axis, such as an angle, joins them.
- The position axes come first, and they keep the boundary of the lattice.
- `simulate` rejects a space above its limit with the issue `space-size`, which names the size it would take.

`coordinate(name)` is the coordinate, along a named axis, of the point a law updates. Streaming is the drift
`[coordinate('v'), 0]`, and acceleration by a force `F` is the drift `[0, F]`. `simulate` rejects a coordinate in a law
whose group is not on a phase space with that axis with the issue `law-space`.

Finite-volume fields, finite-difference fields with `riemann=None`, and spectral fields with `riemann=None` all stream
on a phase space. Spectral derivatives need the axis to be periodic.

## Moments and mean fields

Read the density, mean velocity, or temperature at every position, and feed them back into the dynamics. A
Vlasov-Poisson drift accelerates the gas by the field of its own charge density:

```python
from skcomplex.core.components.laws.utils import coordinate, gradient, inverse_laplacian, moment


def poisson_drift(s):
    return [coordinate('v'), -gradient(inverse_laplacian(1 - moment(s.target.x, 0, axes=('v',))), 1)[0]]


poisson_drift.__name__
```

- `moment(expression, order, axes=names)` reduces over class axes at every position, with the cell widths as weights.
- It is a field over position, which laws read broadcast along the reduced axes. So `normalized=True` and
  `central=True` give the mean velocity and the temperature at every position.
- `reduce_axes(expression, names)` is the integral behind it, for the moments of `cos` and `sin` of an orientation.
- A drift can read a moment of its own field, and the operators of a field over position act along the position axes.

Fields over position and fields on a phase space read each other:

- A drift reads another field over the position lattice through the `sources` of `AdvectionInteraction` and
  `ConservationInteraction`, broadcast along the class axes.
- A `RateInteraction` over the position lattice reads a moment of a field on a phase space through its `sources`.

## Integrals over class axes

Integrate a density against a kernel of two points of its class axes, or along one axis up to every point. Groups with
a fraction `z` of punishers win pairwise conflicts against groups with a fraction `u` at the probability `rho(z, u)`, so
the density changes at the rate `f(z)` times the integral of `rho(z, u) - rho(u, z)` against `f(u)`:

```python
import numpy as np
from sage.all__sagemath_symbolics import tanh

from skcomplex.core.components.elements import FieldElement, NonnegativeElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.laws.utils import coordinate, integral_transform, integration_variable
from skcomplex.core.components.spaces import LatticeSpace, PhaseSpace
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.core.specs import AxisSpec

z, u = coordinate('z'), integration_variable('z')
rho = (1 + tanh(2 * (z - u))) / 2
gain = rho - rho.subs({z: u, u: z}, hold=False)
space = PhaseSpace(LatticeSpace((1,)), [AxisSpec('z', 50, 0.0, 1.0)])
groups = FieldElement('groups', element=NonnegativeElement, space=space, values={'x': np.ones((1, 50))})
conflict = RateInteraction(
    'conflict', target='groups', state='x', rate=lambda s: s.target.x * integral_transform(s.target.x, gain, ('z',))
)
trajectory = simulate(System(elements=[groups], interactions=[conflict]), until=5.0, dt=1.0, solver=RK4Solver(dt=0.05))
final = trajectory.values('groups.x')[-1]
round(float(final.sum() / 50), 10), bool(final[-1] > final[0])
```

The groups of more punishers win, and the total stays one, since the kernel is antisymmetric.

- `integral_transform(expression, kernel, axes)` is the integral over the named class axes of `K(c, c') u(c')`, at the
  same position and the same coordinates of the other class axes. The kernel reads `coordinate(name)` for the point `c`
  a law updates, `integration_variable(name)` for the point `c'` integrated over, and the parameters, so it need not be
  a function of `c - c'`. It is sampled at the cell centres when the problem is compiled, and the integral is a product
  with the dense matrix of the kernel.
- `cumulative_integral(expression, axis, reverse=False)` integrates along one axis of a lattice, from the start of the
  domain to every site, or from every site to the end with `reverse=True`, counting half of the cell of the site. The
  light under a canopy is `exp(-gamma * cumulative_integral(f * coordinate('h')**2, 1, reverse=True))`.
- `convolve` acts along the position axes of a phase space at every class, with a kernel of the displacements of the
  position axes, and it follows the boundary of the position lattice. Seeds disperse over position at every height.
- `simulate` rejects a kernel that reads a state with the issue `kernel-state`, and an integral over class axes of a
  field without a phase space with the issue `law-space`.

## Simplex domains

Hold a density over the frequencies of three strategies, which sum to one. `SimplexSpace` is a phase space whose class
axes are the free frequencies of a simplex, each on `[0, 1]` with the same number of cells. A uniform density of two
on the 2-simplex has the integral one:

```python
import numpy as np

from skcomplex.core.components.elements import FieldElement, NonnegativeElement
from skcomplex.core.components.laws.utils import moment
from skcomplex.core.components.spaces import LatticeSpace, SimplexSpace
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import AxisSpec, ObservableSpec


def total(s):
    return moment(s.x, 0, axes=('x', 'y'))


axes = [AxisSpec('x', 20, 0.0, 1.0), AxisSpec('y', 20, 0.0, 1.0)]
simplex = SimplexSpace(LatticeSpace((1,)), axes).prepare()
groups = FieldElement('groups', element=NonnegativeElement, space=simplex, values={'x': 2.0 * (simplex.volumes_ > 0)})
observables = [ObservableSpec('total', total, group='groups')]
trajectory = simulate(System(elements=[groups]), until=0.1, dt=0.1, observables=observables)
round(float(np.asarray(trajectory.values('total'))[-1]), 10)
```

- `volumes_` holds the fraction of every cell inside the simplex: whole cells below the slanted face, partial cells
  along it, half cells in two dimensions, and empty cells beyond it. `apertures_` holds the fraction of every face
  inside the simplex.
- Finite-volume fluxes pass only through the open part of every face and change a cell by its volume, so the total is
  conserved and no density enters the empty cells. A density on a simplex starts at zero in the empty cells.
- `moment`, `reduce_axes`, `integral_transform`, and `cumulative_integral` weight every cell by its volume.
- `simulate` rejects a local operator, such as `laplacian` or a finite-difference advection, of a field on a simplex
  with the issue `law-space`. `SemiLagrangianSolver` refuses partial cells, and `SSPRKSolver` divides its wave
  speed by the smallest cell volume.

## Encounters and equilibria

Move particles between classes at one position, and relax them towards a local equilibrium:

```python
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.laws.utils import maximum_entropy

relax = RateInteraction('r', target='gas', state='x', rate=lambda s: maximum_entropy(s.target.x, axes=('v',)) - s.target.x)
relax.state
```

`EncounterInteraction` on a phase space pairs classes at the same position. Without `field` it is a linear
redistribution: a candidate at class `h` moves to class `i` at the rate `eta[h]` with the probability `kernel[h, i]`. A
rate that is one expression may read the `coordinate` of the class of the candidate.

`maximum_entropy(expression, axes, order=2)` is the density `exp(a + b c + d |c|^2)` whose discrete moments equal those
of the expression at every position. Newton steps find it, so a relaxation towards it conserves the moments to
round-off.

## Collisions

Add binary collisions between particles of different velocities, as in the Boltzmann equation.
`EncounterInteraction(label, candidate=..., state='x', cross_section=..., factor=None)` is the binary collision
integral over two interval class axes.

- The cross-section is the kernel `B(speed, cosine)` of the relative speed and of the cosine of the deflection angle.
  It returns an expression of the two and numbers, written with Sage functions.
- The interaction returns `RateLaw(f, collision_integral(f, kernel))`. A law of your own can read the same integral,
  since the call carries its kernel. A kernel that reads a state or a parameter raises a `ValidationError`.
- It enters the fast spectral method of Mouhot and Pareschi through a singular value decomposition, a single term for
  Maxwell molecules.
- The method uses FFT products on a grid padded by `(3 + sqrt(2)) / 2`. It conserves the density exactly, and the
  momentum and the energy within its spectral accuracy.
- `factor` scales the collision frequency, such as a function of the density moment for the local Enskog equation.
- `simulate` rejects a collision without two interval class axes with the issue `law-space`.

## Solvers

Step transport and collisions each with the method that suits it, and combine them:

```python
from skcomplex.core.engine.solvers import RK4Solver, SemiLagrangianSolver, StrangSplittingSolver

solver = StrangSplittingSolver(SemiLagrangianSolver(dt=0.1), RK4Solver(dt=0.01), dt=0.1)
solver.dt
```

- `SemiLagrangianSolver(dt)` advances fields whose fluxes are linear drifts `a u`, with `a` free of the state and
  constant along its own axis. It shifts each slice along its characteristic, in the splitting of Cheng and Knorr. It
  is stable at any Courant number and conserves the total on periodic axes. Other fluxes raise a named error.
- `StrangSplittingSolver(transport, collision, dt)` composes a half transport step, a full collision step, and a half
  transport step. The problem keeps the source rates, `problem.sources`, apart from the flux differences.
- `RK4Solver`, `ScipyODESolver`, and the other method-of-lines solvers accept phase-space fields as well.

## Visualizing

`ImageVisual(slices=..., moments=...)` logs the plane that fixes every axis but two, such as `slices={'vy': 2}`. It
also logs the moments over the class axes, as a bar series over a one-dimensional position lattice and as an image over
a two-dimensional one.

`BoltzmannEquationSystem`, `BGKEquationSystem`, `VlasovEquationSystem`, and `EnskogSystem` in `physics.fluids`, and
`BertinDrozGregoireSystem` in `physics.active_matter`, are the kinetic equations of the library.
`NonlinearPairwiseGroupConflictSystem` and `TrimorphicMultilevelSelectionSystem` in `social.cooperation` integrate
against two-point kernels, the second on the simplex, and `ForestSizeStructureSystem` in `biology.ecology` shades trees
by a cumulative integral over height.

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

from skcomplex.core.components.elements import DensityElement, FieldElement
from skcomplex.core.components.interactions import AdvectionInteraction
from skcomplex.core.components.laws.utils import coordinate
from skcomplex.core.components.spaces import LatticeSpace, PhaseSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.core.specs import AxisSpec

space = PhaseSpace(LatticeSpace((32,), spacing=2 * np.pi / 32), [AxisSpec('v', 64, -6.0, 6.0)]).prepare()
x, v = np.arange(32) * 2 * np.pi / 32, space.coordinates_['v']
start = np.outer(1.0 + 0.5 * np.sin(x), np.exp(-(v**2) / 2) / np.sqrt(2 * np.pi))
gas = FieldElement('gas', element=DensityElement, space=space, values={'f': start}, discretization='finite-volume')
stream = AdvectionInteraction('stream', target='gas', state='f', drift=lambda s: [coordinate('v'), 0])
system = System(elements=[gas], interactions=[stream])
simulation = Simulation(system, dt=0.1, solver=RK4Solver(dt=0.01)).prepare()
final = simulation.run(until=0.5).trajectory_.values('gas.f')[-1]
abs(final.sum() / start.sum() - 1.0) < 1e-12
```

The total mass is conserved to round-off.

- `PhaseSpace(position, axes, limit=2**26)` takes a position `LatticeSpace` and the class axes.
- Each class axis is an `AxisSpec(name, size, low, high, periodic=False)` with its own cells. An interval has no flux
  through its ends, and a periodic axis, such as an angle, joins them.
- The position axes come first, and they keep the boundary of the lattice.
- The compile report holds a `space-size` issue for a space above its limit, with the size it would take.

`coordinate(name)` is the coordinate, along a named axis, of the point a law updates. Streaming is the drift
`[coordinate('v'), 0]`, and acceleration by a force `F` is the drift `[0, F]`. The compile report holds a `law-space`
issue for a coordinate in a law whose group is not on a phase space with that axis.

Finite-volume fields, finite-difference fields with `riemann=None`, and spectral fields with `riemann=None` all stream
on a phase space. Spectral derivatives need the axis to be periodic.

## Moments and mean fields

Read the density, mean velocity, or temperature at every position, and feed them back into the dynamics. A
Vlasov-Poisson drift accelerates the gas by the field of its own charge density:

```python
from skcomplex.core.components.laws.utils import coordinate, gradient, inverse_laplacian, moment


def poisson_drift(s):
    return [coordinate('v'), -gradient(inverse_laplacian(1 - moment(s.target.f, 0, axes=('v',))), 1)[0]]


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

## Encounters and equilibria

Move particles between classes at one position, and relax them towards a local equilibrium:

```python
from skcomplex.core.components.interactions import RelaxationInteraction
from skcomplex.core.components.laws.utils import maximum_entropy

relax = RelaxationInteraction('r', target='gas', state='f', goal=lambda s: maximum_entropy(s.target.f, axes=('v',)))
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
`EncounterInteraction(label, candidate=..., state='f', cross_section=..., factor=None)` is the binary collision
integral over two interval class axes.

- The cross-section is a function of arrays of the relative speed and of the cosine of the deflection angle.
- It enters the fast spectral method of Mouhot and Pareschi through a singular value decomposition, a single term for
  Maxwell molecules.
- The method uses FFT products on a grid padded by `(3 + sqrt(2)) / 2`. It conserves the density exactly, and the
  momentum and the energy within its spectral accuracy.
- `factor` scales the collision frequency, such as a function of the density moment for the local Enskog equation.
- The compile report holds a `law-space` issue for a collision without two interval class axes.

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

`boltzmann_equation`, `bgk_equation`, `vlasov_equation`, and `enskog` in `physics/fluids`, and `bertin_droz_gregoire`
in `physics/active_matter`, are the kinetic equations of the library.

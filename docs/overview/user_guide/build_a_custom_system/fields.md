# Fields

This page builds quantities spread over space, such as heat in a rod or a chemical in a dish. You write the law of one
site, and the library applies it at every site of a lattice, a network, or a mesh. A field is this set of values, one
per site.

## A field on a lattice

This section spreads heat from a hot spot along a rod. A
[`FieldElement`](../core_components/elements.md#fieldelement) places one member of an element class on every site of
a space and gives the members their initial values. Here the class `HeatElement` returns a
[`RateLaw`](../core_components/laws.md#ratelaw) whose rate is `D` times the Laplacian of the temperature `u`, the
heat equation. The rod is a [`LatticeSpace`](../core_components/spaces.md#latticespace) of 400 sites, 0.05 apart, with
insulated ends, and it starts at zero except for 20 units of heat at its middle site.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import laplacian
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ParameterSpec, StateSpec


class HeatElement(Element):
    def __init__(
        self,
        label,
        *,
        u: Annotated[float, StateSpec()] = 0.0,
        D: Annotated[float, ParameterSpec()] = 1.0,
    ):
        super().__init__(label)
        self.u = u
        self.D = D

    def _get_laws(self, s):
        return [RateLaw(s.u, s.D * laplacian(s.u))]


initial = np.zeros(400)
initial[200] = 20.0
space = LatticeSpace(400, spacing=0.05)
rod = FieldElement('rod', element=HeatElement, space=space, boundary={'u': 'zero-flux'}, values={'u': initial})
trajectory = simulate(CompositeSystem(elements=[rod]), until=0.5, dt=0.1)
heat = trajectory.get_values('rod.u')
round(float(heat[-1].max()), 3), heat.sum(axis=1)[[0, -1]].round(6).tolist()
```

The peak falls from 20 to 0.399 after half a unit of time, and the total heat stays at 20.0, because no heat leaves
through insulated ends.

The laws of the class are written for one site and read the neighbouring sites through differential operators of
`skcomplex.core.components.laws.utils`. The method of lines turns the field into one ordinary differential equation per
site, so every ODE [solver](../glossary.md#solver) and every backend runs it.

- `laplacian(u)` becomes the stencil of `2d + 1` points divided by the squared spacing on a lattice of `d` axes, and the
  graph Laplacian on a [`NetworkSpace`](../core_components/spaces.md#networkspace).
- `gradient(u, n_dim)` and `divergence(vector)` use central differences, and `advection(u, velocity)` uses upwind
  differences.
- An operator applies to any expression of the states and parameters of the field, and operators nest.

The argument `boundary` sets the edges of each state, as `'periodic'`, `'zero-flux'`, or `('fixed', value)`. Zero flux
mirrors the field at the edge and conserves its total, as the run above shows. A fixed edge reads the given value. A
state without an entry takes the `boundary` of the lattice. The equations keep the operators, such as
`diff(u(t), t) == D*laplacian(u(t))`, and the backends evaluate the stencils. On Numba, the stencils of a lattice fuse
into one compiled loop.

Explicit steps stay stable only below the diffusive limit `h^2 / (2 d D)`, where `h` is the spacing. A trajectory
holds this limit as `trajectory.problem.stable_step`, and a solver of fixed step whose `dt` exceeds it warns before the
run and names the limit.

Agents on a grid of the same shape couple to a field by co-simulation. A rule reads the field at a cell with
`system.get_field('ground/pheromone.level', cell)` and adds to it with `system.deposit(name, cell, amount)`. The
deposits reach the field at the next point of synchronisation. Lattice fields stream to Rerun as images.

The catalog holds `HeatDiffusionSystem`, `FisherKPPSystem`, `GrayScottSystem`, `BrusselatorSystem`,
`FitzhughNagumoSystem`, `SchnakenbergSystem`, and `PheromoneForagingSystem`. Each is imported from its subfield, such
as `from skcomplex.systems.chemistry.pattern_formation import GrayScottSystem`.

## Discretizations and stiff solvers

This section runs the same laws on another discretization and takes long steps through stiff equations. The
Kuramoto-Sivashinsky equation `u_t = -u u_x - u_xx - u_xxxx` grows waves at long wavelengths and damps them at short
ones, and its fourth derivative makes it stiff. Below it runs spectrally on a periodic line of 128 sites, and
`ETDRK4Solver` steps it by 0.25, far beyond the explicit limit.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import divergence, laplacian
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import ETDRK4Solver
from skcomplex.core.specs import StateSpec


class ChaosElement(Element):
    def __init__(self, label, *, u: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.u = u

    def _get_laws(self, s):
        return [RateLaw(s.u, -divergence([s.u**2 / 2]) - laplacian(s.u) - laplacian(laplacian(s.u)))]


positions = np.arange(1, 129) * 32 * np.pi / 128
line = FieldElement(
    'line',
    element=ChaosElement,
    space=LatticeSpace(128, spacing=32 * np.pi / 128),
    values={'u': np.cos(positions / 16) * (1 + np.sin(positions / 16))},
    discretization='spectral',
)
trajectory = simulate(CompositeSystem(elements=[line]), until=100, dt=10, solver=ETDRK4Solver(dt=0.25))
round(float(np.abs(trajectory.get_values('line.u')[-1]).max()), 1)
```

After 100 units of time the largest amplitude of the chaotic field is 2.4.

The argument `discretization` of `FieldElement` chooses how space is discretized.

- `'spectral'` fields on periodic lattices multiply Fourier modes, and the two-thirds rule removes aliasing from every
  operator, so `laplacian(u)` is exact for smooth fields. `inverse_helmholtz(u)` solves Poisson equations.
  `symmetry={'u': (None, 'odd')}` keeps a state odd or even along an axis, which represents walls on a doubled lattice.
  `LatticeSpace(spacing=(dx, dy))` sets a spacing per axis.
- `'finite-volume'` fields take a [`FluxLaw`](../core_components/laws.md#fluxlaw) per state, the conservation law
  `du/dt + divergence(flux) = 0`. They update with MUSCL reconstruction, minmod or van Leer limiters, and Rusanov or
  exact Godunov fluxes. `SSPRKSolver` steps them by the CFL condition and by the stability limit of their Laplacian
  rates at the current parameters, and it conserves their totals.
- `'finite-element'` fields live on a [`MeshSpace`](../core_components/spaces.md#meshspace) of the points and cells of
  a scikit-fem mesh, such as `MeshSpace(grid.p, grid.t)` for `grid = skfem.MeshTri.init_lshaped().refined(3)`. Their
  stiffness and mass matrices come from scikit-fem, and their boundary values can be functions of the coordinates. They
  stream to Rerun as colored meshes.

A stiff equation has rates that decay on very different time scales, so explicit steps must follow the fastest one.
Implicit and exponential steps do not. The compiler splits the rate of every field into a linear part with constant
coefficients, such as `D * laplacian(u)` or `-laplacian(laplacian(u))`, and a nonlinear remainder.

- `IMEXEulerSolver`, `IMEX2Solver`, and `CrankNicolsonSolver` treat the linear part implicitly. They solve sparse
  systems for finite differences and finite elements, and one equation per Fourier mode for spectral fields.
- `ETDRK4Solver` integrates the linear part exactly per mode, coupled states included, which is why the run above took
  steps far beyond the explicit limit.

For finite-difference and finite-element fields, the implicit methods of `ScipyODESolver` are the stiff path. Unlike
the split steppers of fixed step, they control the error and adapt the step. Below, BDF runs the Fisher-KPP equation of
a spreading population on 400 sites.

```python
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.systems.biology.ecology import FisherKPPSystem

stiff = simulate(FisherKPPSystem(length=400, D=1.0, spacing=0.05), until=5, dt=1.0, solver=ScipyODESolver(method='BDF'))
pattern = ScipyODESolver(method='BDF').prepare(stiff.problem).jacobian_pattern_
pattern.nnz, stiff.get_values('line.u')[-1].max().round(3)
```

The Jacobian has 1198 entries that are not zero, three per site less the two missing neighbours at the ends. The
population, scaled so that its capacity is 1, reaches 0.954 at its highest site.

The compiler derives this sparsity pattern of the Jacobian from the laws. The pattern covers the reads of members and
shared values, the neighbour tables of stencils, the sparse matrices of finite elements, and the reconstruction
stencils of finite volumes. It holds dense blocks where couplings are known only at run time, such as pairs, spectral
operators, and callables.

- `ScipyODESolver(method='BDF')` and `ScipyODESolver(method='Radau')` receive the pattern, which a solver prepared on
  the `problem` of a trajectory holds as `jacobian_pattern_`, as above. SciPy then estimates the Jacobian with one
  evaluation of the drift per group of independent columns, and factors it with sparse LU.
- `LSODA` receives the half bandwidths of a pattern narrower than a tenth of the size.
- The option `jacobian` selects the path: `'auto'` by default, `'exact'` for the sparse Jacobian of the laws
  differentiated by Sage, which laws without a symbolic form cannot give, and `'dense'` for the dense estimate.
- Explicit methods ignore the pattern, and `RK45` warns when one call spans more than 1000 stable steps of the diffusion
  of a field.

The pattern decides the cost of a stiff run. On Fisher-KPP with 4000 points, whose largest eigenvalue is about 1600,
run to `t = 10`, BDF with the dense estimate took 35 to 50 s. With the pattern it took about 0.2 s, with the exact
Jacobian 0.3 s, Radau with the pattern 0.7 s, and `CrankNicolsonSolver(dt=0.01)` 0.8 s. These times come from one
machine and change with it. Spectral fields keep `ETDRK4Solver` and the IMEX steppers, because their linear part is
diagonal in Fourier space while its pattern is dense.

The catalog holds `DecayingTurbulenceSystem`, two-dimensional incompressible flow in the form of vorticity and
streamfunction, and `RayleighBenardSystem`, which adds Boussinesq convection with the numbers `Ra` and `Pr`. It also
holds `KuramotoSivashinskySystem`, `ComplexGinzburgLandauSystem`, `SwiftHohenbergSystem`, `HeatLShapeSystem`,
`BurgersSystem`, and `ShallowWaterSystem`.

## Nonlocal operators

This section reads a field beyond the nearest sites. A rate can smooth the field against a kernel, solve for a
potential, or look ahead by a distance. Each rate below changes a Gaussian bump on a line of 128 sites, through a
[`RateInteraction`](../core_components/interactions.md#rateinteraction) on the field, and the block returns the total of
the field after 0.2 units of time.

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import exp, pi, sqrt

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.laws.utils import convolve, displacement, inverse_helmholtz, partial, shifted
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ParameterSpec, StateSpec


class SiteElement(Element):
    def __init__(
        self,
        label,
        *,
        u: Annotated[float, StateSpec()] = 0.0,
        k: Annotated[float, ParameterSpec()] = 2.0,
    ):
        super().__init__(label)
        self.u = u
        self.k = k


def rate_of(rate):
    centers = np.arange(128) * 0.1
    values = {'u': np.exp(-((centers - 6.4) ** 2))}
    line = FieldElement('line', element=SiteElement, space=LatticeSpace(128, spacing=0.1), values=values)
    interaction = RateInteraction('rate', target='line', state='u', rate=rate)
    return simulate(CompositeSystem(elements=[line], interactions=[interaction]), until=0.2, dt=0.05).get_values('line.u')


def smoothing(s):
    kernel = exp(-displacement(0) ** 2) / sqrt(pi)
    return convolve(s.target.u, kernel) - s.target.u


def screened_drift(s):
    return partial(inverse_helmholtz(s.target.u, s.target.k), 0)


def look_ahead(s):
    return shifted(s.target.u, [0.25 + s.target.u]) - s.target.u


{rate.__name__: rate_of(rate)[-1].sum().round(3) for rate in (smoothing, screened_drift, look_ahead)}
```

The bump starts with a total of 17.725, the integral `sqrt(pi)` divided by the spacing 0.1. Smoothing and the screened
drift keep this total, since each rate sums to zero over the periodic line. The look ahead lowers it to 16.691,
because its distance depends on the field, so the reads ahead no longer balance the values they replace.

These operators of `skcomplex.core.components.laws.utils` read beyond the nearest sites, and a law writes them as a
reference writes them.

- `inverse_helmholtz(expression, screening=0, boundary=None)` is the solution `psi` of
  `(laplacian - screening**2) psi = expression`, and `inverse_helmholtz(u)` solves the Poisson equation. It runs on
  spectral fields, and on finite-difference lattices with periodic, zero-flux, or fixed edges, where the fast transforms
  of SciPy invert the stencil that `laplacian` uses. On networks and meshes it uses a sparse factorization. An operand
  that mixes states of different boundaries needs `boundary=`.
- `convolve(expression, kernel, radius=None)` sums an expression over the sites of a lattice against a kernel written
  in the symbols `displacement(axis)` and the parameters. The kernel is sampled when the problem compiles. It wraps
  around periodic edges and is cut at fixed and zero-flux edges, where the operand is padded with the fixed value or
  its mirror image.
- `shifted(expression, offset, order=1)` reads an expression at a physical displacement, one component per axis, which
  may depend on the states. A displacement of whole sites reads the neighbour tables and stays compilable to Numba. Any
  other displacement interpolates with `order=1` or `order=3`, following the boundary of the field.
- `cumulative_integral(expression, axis, reverse=False)` integrates an expression along one axis of a lattice, from
  the start of the domain to every site, or from every site to the end with `reverse=True`.

On a [`PhaseSpace`](../core_components/spaces.md#phasespace), a convolution acts along the position axes at every
class, and `integral_transform` integrates over the class axes against a kernel of two points that need not be a
function of their displacement, as [phase-space fields](phase_space_fields.md) describes.

The stencil operators read a nonlocal result at the neighbouring sites, so `partial` and `gradient` apply to an
inverted streamfunction. Laws with these operators run on the NumPy backend, except reads of whole sites. The sparsity
pattern of the Jacobian marks a convolution as coupling the sites within its radius, and an inversion as dense. Implicit
solvers therefore stay correct but slow, and explicit or exponential solvers suit these laws best. An observable may
use the operators too, which streams a streamfunction as an image.

`simulate` rejects four misuses with a `ValidationError` that names each issue with the path of the law and a fix.

- `law-space` for a convolution or a shifted read on a network or a mesh, and for a spectral operator on a lattice
  without periodic edges.
- `kernel-state` for a kernel that reads a state.
- `screening` for a screening that reads a state or whose square can be negative.
- `operator-spaces` for an operand that reads fields of more than one space.

## Coupling through a function of both ends

This section couples the sites of a field without an operator. A
[`CouplingInteraction`](../core_components/interactions.md#couplinginteraction) on a field sums a function of each site
and one neighbour over the neighbours of every site, and reads a fixed boundary value as `laplacian` does. Here each
cell of a periodic ring of 32 exchanges a fifth of its difference with each neighbour, starting from one unit at cell
16.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.interactions import CouplingInteraction
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class CellElement(Element):
    def __init__(self, label, *, u: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.u = u


cells = FieldElement('cells', element=CellElement, space=LatticeSpace(32), values={'u': np.eye(32)[16]})
exchange = CouplingInteraction(
    'exchange',
    target='cells',
    read='u',
    write='u',
    function=lambda s: 0.2 * (s.other.u - s.member.u),
)
system = CompositeSystem(elements=[cells], interactions=[exchange])
trajectory = simulate(system, until=2.0, dt=0.5)
trajectory.get_values('cells.u')[-1].sum().round(6)
```

The total stays at 1.0 after two units of time, since each exchange moves as much into one cell as out of the other.

A constant times `s.other.u - s.member.u` becomes `laplacian`, so spectral and finite-element fields keep their own
operator.

The catalog uses these operators in its models. `AmariNeuralFieldSystem` holds a bump of neural activity with a
convolution, `DoubleGyreSystem` streams the streamfunction of a closed basin, `GktTrafficSystem` brakes on the traffic
read ahead, and `ActiveViscoelasticGelSystem` solves its force balance at every step.

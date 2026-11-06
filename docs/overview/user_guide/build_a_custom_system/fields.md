# Fields

Describe a quantity spread over space, such as heat in a rod or a chemical in a dish, and write its law once, for one
site. Heat spreads from a hot spot in the middle of a rod with insulated ends:

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import laplacian
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
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
trajectory = Simulation(System(elements=[rod]), dt=0.1).prepare().run(until=0.5).trajectory_
heat = trajectory.values('rod.u')
round(float(heat[-1].max()), 3), heat.sum(axis=1)[[0, -1]].round(6).tolist()
```

A field holds one value per site of a lattice or per node of a network. `FieldElement(label, element=..., space=...,
boundary=..., values=...)` places a member of an element type on every site. The laws of the type are written for one
site, and they read the neighbouring sites through differential operators. The method of lines turns a field into
ordinary differential equations, so every ODE solver and backend runs it.

- `laplacian(u)` lowers to the `2d + 1`-point stencil divided by the squared spacing, and to the graph Laplacian on a
  `NetworkSpace`.
- `gradient(u, n_dim)` and `divergence(vector)` use central differences, and `advection(u, velocity)` upwind
  differences.
- Operators apply to any expression of the field's states and parameters, and they nest.
- `boundary={'u': 'periodic' | 'zero-flux' | ('fixed', value)}` sets the edges of each state. Zero flux mirrors the
  field and conserves its total, and fixed edges read the given value. The lattice's own `boundary` is the default.
- The equations keep the operators, `diff(u(t), t) == D*laplacian(u(t))`, while the backends evaluate the stencils.
  On Numba, the stencils of a lattice are fused into one compiled loop.
- `problem.stable_step` is the diffusive limit `h^2 / (2 d D)` of explicit steps. Fixed-step solvers whose `dt` exceeds
  it warn before the run, with the limit.

Agents on a grid of the same shape couple to a field in co-simulation. A rule reads the field at a cell with
`system.field('ground/pheromone.level', cell)` and adds to it with `system.deposit(name, cell, amount)`. The deposits
reach the field at the next synchronisation point. Lattice fields stream to Rerun as images.

The library has `heat_diffusion`, `fisher_kpp`, `gray_scott`, `brusselator`, `fitzhugh_nagumo`, `schnakenberg`, and
`pheromone_foraging`, which you select by name, such as `create_system('gray_scott')`.

## Discretizations and stiff solvers

Run the same laws with another discretization, and take long steps through stiff equations. The Kuramoto-Sivashinsky
equation, spectral in space and stepped by `ETDRK4Solver` far beyond the explicit limit:

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import divergence, laplacian
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
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
state = Simulation(System(elements=[line]), dt=10, solver=ETDRK4Solver(dt=0.25)).prepare().run(until=100).state_
round(float(np.abs(state).max()), 1)
```

The argument `FieldElement(..., discretization=...)` chooses how space is discretized:

- `'spectral'` fields on periodic lattices multiply Fourier modes, and the two-thirds rule removes aliasing from every
  operator, so `laplacian(u)` is exact for smooth fields. `inverse_laplacian(u)` solves Poisson equations.
  `symmetry={'u': (None, 'odd')}` keeps a state odd or even along an axis, which represents walls on a doubled lattice.
  `LatticeSpace(spacing=(dx, dy))` sets a spacing per axis.
- `'finite-volume'` fields take `FluxLaw(state, flux)`s, `du/dt + divergence(flux) = 0`. They are updated with MUSCL
  reconstruction, minmod or van Leer limiters, and Rusanov or exact Godunov fluxes. `SSPRKSolver` steps them by the CFL
  condition and by the stability limit of their Laplacian rates at the current parameters, and conserves their totals.
- `'finite-element'` fields live on a `MeshSpace` of the points and cells of a scikit-fem mesh, such as
  `MeshSpace(grid.p, grid.t)` for `grid = skfem.MeshTri.init_lshaped().refined(3)`. Their stiffness and mass matrices
  come from scikit-fem, and their boundary values can be functions of the coordinates. They stream to Rerun as colored
  meshes.

Stiff equations need implicit or exponential steps. The compiler splits the rate of every field into a linear part with
constant coefficients, such as `D * laplacian(u)` or `-laplacian(laplacian(u))`, and a nonlinear remainder.

- `IMEXEulerSolver`, `IMEX2Solver`, and `CrankNicolsonSolver` treat the linear part implicitly. They use sparse solves
  for finite differences and finite elements, and solves per Fourier mode for spectral fields.
- `ETDRK4Solver` integrates the linear part exactly per mode, coupled states included, so stiff equations take steps
  far beyond the explicit limit.

The implicit methods of `ScipyODESolver` are the stiff path for finite-difference and finite-element fields, with the
adaptive error control that the fixed-step split steppers lack:

```python
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.systems import create_system

stiff = create_system('fisher_kpp', length=400, D=1.0, spacing=0.05)
simulation = Simulation(stiff, dt=1.0, solver=ScipyODESolver(method='BDF')).prepare().run(until=5)
simulation.stepper_.jacobian_pattern_.nnz, simulation.state_.max().round(3)
```

The compiler derives the sparsity pattern of the Jacobian from the laws. The pattern covers member and shared reads,
the neighbour tables of stencils, the sparse matrices of finite elements, and the reconstruction stencils of finite
volumes. It holds dense blocks where couplings are known only at run time, such as pairs, spectral operators, and
callables.

- `ScipyODESolver(method='BDF')` and `ScipyODESolver(method='Radau')` receive the pattern. SciPy then estimates the
  Jacobian with one drift evaluation per group of independent columns, and factors it with sparse LU.
- `LSODA` receives the half bandwidths of a pattern narrower than a tenth of the size.
- The option `jacobian` selects the path: `'auto'` by default, `'exact'` for the sparse Jacobian of the laws
  differentiated by Sage, which laws without a symbolic form cannot give, and `'dense'` for the dense estimate.
- Explicit methods ignore it, and `RK45` warns when one call spans more than 1000 stable steps of the diffusion of a
  field.

On Fisher-KPP on 4000 points, whose largest eigenvalue is about 1600, run to `t = 10`, BDF with the dense estimate took
35 to 50 s. With the pattern it took about 0.2 s, with the exact Jacobian 0.3 s, Radau with the pattern 0.7 s, and
`CrankNicolsonSolver(dt=0.01)` 0.8 s. Spectral fields keep `ETDRK4Solver` and the IMEX steppers. These win there,
because the linear part is diagonal in Fourier space while its pattern is dense.

The system `decaying_turbulence` is two-dimensional incompressible flow in vorticity-streamfunction form, and
`rayleigh_benard` adds Boussinesq convection, with `Ra` and `Pr`. The library also has `kuramoto_sivashinsky`,
`complex_ginzburg_landau`, `swift_hohenberg`, `heat_l_shape`, `burgers`, and `shallow_water`.

## Nonlocal operators

Read a field beyond the nearest sites: smooth it against a kernel, solve for a potential, or look ahead by a distance.
Each rate below changes a Gaussian bump on a line in its own way:

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import exp, pi, sqrt

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.laws.utils import convolve, displacement, inverse_helmholtz, partial, shifted
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
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
    simulation = Simulation(System(elements=[line], interactions=[interaction]), dt=0.05)
    return simulation.prepare().run(until=0.2).trajectory_.values('line.u')


def smoothing(s):
    kernel = exp(-displacement(0) ** 2) / sqrt(pi)
    return convolve(s.target.u, kernel) - s.target.u


def screened_drift(s):
    return partial(inverse_helmholtz(s.target.u, s.target.k), 0)


def look_ahead(s):
    return shifted(s.target.u, [0.25 + s.target.u]) - s.target.u


{rate.__name__: rate_of(rate)[-1].sum().round(3) for rate in (smoothing, screened_drift, look_ahead)}
```

Three operators of `skcomplex.core.components.laws.utils` read beyond the nearest sites, and a law writes them as a
reference writes them:

- `inverse_helmholtz(expression, screening, boundary=None)` is the solution of
  `(laplacian - screening**2) psi = expression`, and `inverse_laplacian(u)` is its case without screening. It runs on
  spectral fields, and on finite-difference lattices with periodic, zero-flux, or fixed edges, where the fast transforms
  of SciPy invert the stencil that `laplacian` uses. On networks and meshes it uses a sparse factorization. An operand
  that mixes states of different boundaries needs `boundary=`.
- `convolve(expression, kernel, radius=None)` sums an expression over the sites of a lattice against a kernel written
  in the symbols `displacement(axis)` and the parameters. The kernel is sampled when the problem is compiled. It wraps
  around periodic edges and is cut at fixed and zero-flux edges, where the operand is padded with the fixed value or
  its mirror image.
- `shifted(expression, offset, order=1)` reads an expression at a physical displacement, one component per axis, which
  may depend on the states. A displacement of whole sites reads the neighbour tables and stays compilable to Numba. Any
  other one interpolates with `order=1` or `order=3`, following the boundary of the field.

The stencil operators read a nonlocal result at the neighbouring sites, so `partial` and `gradient` apply to an
inverted streamfunction. Laws with these operators run on the NumPy backend, except whole-site reads. The sparsity
pattern of the Jacobian marks a convolution as coupling the sites within its radius, and an inversion as dense. So
implicit solvers stay correct but slow, and explicit or exponential solvers suit these laws best. An observable may use
the operators too, which streams a streamfunction as an image.

The misuses are issues of the compiler's report, each with the path of the law and a fix:

- `law-space` for a convolution or a shifted read on a network or a mesh, and for a spectral operator on a lattice
  without periodic edges,
- `kernel-state` for a kernel that reads a state,
- `screening` for a screening that reads a state or whose square can be negative,
- `operator-spaces` for an operand that reads fields of more than one space.

A field can also couple its sites through a function of both ends, without an operator. A `CouplingInteraction` on a
field sums the function over the neighbours of every site, and reads a fixed boundary value as `laplacian` does:

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.interactions import CouplingInteraction
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
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
system = System(elements=[cells], interactions=[exchange])
trajectory = Simulation(system, dt=0.5).prepare().run(until=2.0).trajectory_
trajectory.values('cells.u')[-1].sum().round(6)
```

A constant times `s.other.u - s.member.u` lowers to `laplacian`, so spectral and finite-element fields keep their own
operator.

`amari_neural_field` holds a bump with a convolution, `double_gyre` streams the streamfunction of a closed basin,
`gkt_traffic` brakes on the traffic read ahead, and `active_viscoelastic_gel` solves its force balance at every step.

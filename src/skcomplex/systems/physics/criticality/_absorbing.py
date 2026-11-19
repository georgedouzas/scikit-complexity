"""Absorbing-state transitions.

The module holds the Langevin field theory of directed percolation coupled to an energy field.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import sqrt

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import NoiseLaw, RateLaw, UpdateLaw
from ....core.components.laws.utils import laplacian
from ....core.components.spaces import BaseSpace, LatticeSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, EulerMaruyamaSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role


class ActiveFieldElement(Element):
    """Sites of the Langevin equation of directed percolation with a conserved field.

    Every site has an activity `rho` and a field `nu`. `ActivePercolationInteraction` grows, saturates, spreads, and
    shakes the activity, and feeds and drains the field.

    Args:
        label:
            The label of the site or the field.

        rho:
            The initial activity, or one value per site.

        nu:
            The initial field, or one value per site.

        space:
            The lattice of the sites, or `None` for one site.

    Examples:
        >>> from skcomplex.systems.physics.criticality import ActiveFieldElement
        >>> ActiveFieldElement('site').nu
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        rho: object = 0.5,
        nu: object = 1.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.rho = rho
        self.nu = nu


class ActivePercolationInteraction(Interaction):
    """Directed percolation of an activity fed by a conserved field.

    The activity follows `rho_t = (a + b nu) rho - c rho^2 + d laplacian(rho) + e sqrt(rho) eta`, and the field
    follows `nu_t = laplacian(nu) + f - g rho`. Both stay nonnegative.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the sites, whose values are `rho` and `nu`.

        a:
            The linear growth rate of the activity.

        b:
            The growth of the activity per unit of the field.

        c:
            The saturation of the activity.

        d:
            The diffusion coefficient of the activity.

        e:
            The amplitude of the noise.

        f:
            The input of the field.

        g:
            The consumption of the field by the activity.

    Examples:
        >>> from skcomplex.systems.physics.criticality import ActivePercolationInteraction
        >>> ActivePercolationInteraction('percolation', target='neurons').e
        0.2
    """

    law_classes = (UpdateLaw, NoiseLaw, RateLaw)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        a: float = -1.0,
        b: float = 1.0,
        c: float = 1.0,
        d: float = 1.0,
        e: float = 0.2,
        f: float = 0.0,
        g: float = 0.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f
        self.g = g

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rates, the noise, and the bounds of the activity and the field.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s, t = context, context.t
            rho, nu = s.target.rho, s.target.nu
            self.laws_ = [
                RateLaw(rho.diff(t), (s.a + s.b * nu) * rho - s.c * rho**2 + s.d * laplacian(rho)),
                NoiseLaw(rho, s.e * sqrt(rho)),
                RateLaw(nu.diff(t), laplacian(nu) + s.f - s.g * rho),
                UpdateLaw(rho, (rho + abs(rho)) / 2),
                UpdateLaw(nu, (nu + abs(nu)) / 2),
            ]


class DirectedPercolationLangevinSystem(BaseSystem):
    """The Langevin field theory of an absorbing-state transition.

    An active density is fed by an energy field.

    The active density `rho` grows at the local rate `a + b nu`, saturates through `-c rho^2`, spreads with the
    diffusivity `d`, and fluctuates with the multiplicative noise `e sqrt(rho)`, which vanishes in the absorbing state
    `rho = 0`. The energy field `nu` diffuses, gains the drive `f`, and loses the dissipation `g rho`, so `rho_t = (a +
    b nu) rho - c rho^2 + d laplacian(rho) + e sqrt(rho) xi` and `nu_t = laplacian(nu) + f - g rho` (Tian et al., Eqs.
    3-4). Both fields stay nonnegative. Without drive and dissipation the energy is a fixed control parameter: below the
    threshold `nu_c = -a / b` activity dies out, and above it the density settles near `(a + b nu) / c`. The system has
    the activity `neurons.rho` and the energy `neurons.nu`.

    Args:
        shape:
            The number of sites along every axis of the periodic lattice.

        a:
            The bare growth rate of the activity.

        b:
            The gain of the growth rate per unit of energy.

        c:
            The saturation of the activity.

        d:
            The diffusivity of the activity.

        e:
            The amplitude of the multiplicative noise.

        f:
            The drive of the energy.

        g:
            The dissipation of the energy per unit of activity.

        energy:
            The initial energy at every site.

        density:
            The mean initial activity, drawn uniformly between 0 and twice this value.

        random_state:
            The seed or generator of the initial activity.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.criticality import DirectedPercolationLangevinSystem
        >>> system = DirectedPercolationLangevinSystem((8, 8), energy=0.5).prepare().system_
        >>> float(system.elements[0].nu.mean())
        0.5
    """

    field = 'physics/criticality'
    topics = ('field', 'absorbing state', 'directed percolation', 'multiplicative noise', 'spde')
    references = ('Tian et al. 2022', 'Henkel, Hinrichsen, and Lübeck 2008', 'Bonachela and Muñoz 2009')
    until = 100.0
    dt = 1.0

    def __init__(
        self,
        shape: Sequence[int] = (64, 64),
        *,
        a: float = -1.0,
        b: float = 1.0,
        c: float = 1.0,
        d: float = 1.0,
        e: float = 0.2,
        f: float = 0.0,
        g: float = 0.0,
        energy: float = 1.2,
        density: float = 0.5,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.shape = shape
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f
        self.g = g
        self.energy = energy
        self.density = density
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the activity and the energy on a periodic lattice.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        shape = tuple(self.shape)
        rng = np.random.default_rng(self.random_state)
        neurons = ActiveFieldElement(
            'neurons',
            rho=rng.uniform(0.0, 2 * self.density, shape).ravel(),
            nu=np.full(int(np.prod(shape)), self.energy),
            space=LatticeSpace(shape),
        )
        percolation = ActivePercolationInteraction(
            'percolation',
            target='neurons',
            a=self.a,
            b=self.b,
            c=self.c,
            d=self.d,
            e=self.e,
            f=self.f,
            g=self.g,
        )
        self.system_ = System(
            self.label,
            elements=[neurons],
            interactions=[percolation],
            solver=EulerMaruyamaSolver(dt=0.01),
        )

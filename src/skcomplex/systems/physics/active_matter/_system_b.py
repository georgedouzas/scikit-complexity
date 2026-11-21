"""Active System B+.

It is the conserved scalar field theory of motility-induced phase separation with thermal noise.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo, sqrt
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import NoiseLaw, RateLaw
from ....core.components.laws.utils import divergence, gradient, laplacian
from ....core.components.spaces import BaseSpace, LatticeSpace
from ....core.engine.backends import BaseBackend, NumbaBackend
from ....core.engine.solvers import BaseSolver, EulerMaruyamaSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role


class ActiveScalarElement(Element):
    """Sites of a conserved scalar field `phi`, such as the density of active particles.

    Args:
        label:
            The label of the field.

        phi:
            The initial field, one value per site or shared.

        space:
            The lattice of the sites, or `None` for one site.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import ActiveScalarElement
        >>> ActiveScalarElement('site', phi=0.5).phi
        0.5
    """

    def __init__(self, label: str, *, phi: object = 0.0, space: BaseSpace | None = None) -> None:
        super().__init__(label, space=space)
        self.phi = phi


class ActiveCahnHilliardInteraction(Interaction):
    """Cahn-Hilliard dynamics of a conserved field with active currents, Active System B+.

    The field follows `phi_t = laplacian(mu) - div(zeta laplacian(phi) grad phi) + div(sqrt(2 D) eta)`, with the
    chemical potential `mu = -phi + phi^3 - kappa laplacian(phi) + lam |grad phi|^2`. The active terms `lam` and `zeta`
    break the detailed balance of the Cahn-Hilliard equation, and a large `zeta` reverses Ostwald ripening. It is the
    density of swimming bacteria or of self-propelled colloids that cluster.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the field `phi`.

        kappa:
            The stiffness of the interfaces.

        lam:
            The active term of the chemical potential.

        zeta:
            The active current.

        D:
            The strength of the conserved noise.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import ActiveCahnHilliardInteraction
        >>> ActiveCahnHilliardInteraction('currents', target='fluid').zeta
        4.0
    """

    law_classes = (NoiseLaw, RateLaw)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        kappa: Annotated[float, RealSet.open(0, oo)] = 1.0,
        lam: float = 1.0,
        zeta: float = 4.0,
        D: Annotated[float, RealSet.closed_open(0, oo)] = 0.05,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.kappa = kappa
        self.lam = lam
        self.zeta = zeta
        self.D = D

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the conserved rate and the conserved noise of the field of the target.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        s = context
        phi = s.target.phi
        slope = gradient(phi)
        potential = -phi + phi**3 - s.kappa * laplacian(phi) + s.lam * sum(item**2 for item in slope)
        active = [s.zeta * laplacian(phi) * item for item in slope]
        self.laws_ = [
            RateLaw(phi.diff(s.t), laplacian(potential) - divergence(active)),
            NoiseLaw(phi, sqrt(2 * s.D), conserved=True),
        ]


class ActiveSystemBSystem(BaseSystem):
    """Active System B+, the conserved field theory of the density of active particles.

    The density `phi` follows `phi_t = laplacian(mu) - div(zeta laplacian(phi) grad(phi)) + div(sqrt(2 D) eta)`, with
    the chemical potential `mu = -phi + phi^3 - kappa laplacian(phi) + lam |grad(phi)|^2` and a conserved white noise
    `eta`, Equation 4 of Fodor, Jack, and Cates with unit mobility. The terms `lam` and `zeta` are the lowest-order
    active currents that no free energy gives. With `zeta = 0` it is Active System B, where `lam` shifts the coexisting
    densities and the phases still coarsen. With `lam = 1`, `zeta = 4`, and a majority of the dense phase, the Ostwald
    ripening of the dilute bubbles reverses, so they stop growing at a finite size, a microphase separation. The noise
    conserves the total of `phi`, so its mean stays fixed. The lattice has unit spacing, and an explicit step must stay
    below about 1/32. The system has the field `fluid.phi`.

    Args:
        shape:
            The number of sites along every axis of the periodic lattice.

        mean:
            The mean of `phi`, which the dynamics conserve.

        kappa:
            The stiffness of the interfaces.

        lam:
            The active term `lam |grad(phi)|^2` of the chemical potential.

        zeta:
            The active current `zeta laplacian(phi) grad(phi)`.

        D:
            The temperature of the conserved noise.

        random_state:
            The seed or generator of the initial noise of amplitude 0.05 around the mean.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import ActiveSystemBSystem
        >>> ActiveSystemBSystem((8, 8), zeta=0.0).prepare().system_.interactions[0].zeta
        0.0
    """

    field = 'physics/active_matter'
    topics = (
        'field',
        'active matter',
        'phase separation',
        'microphase separation',
        'conserved noise',
        'system b',
        'spde',
    )
    references = ('Fodor, Jack, and Cates 2022', 'Wittkowski et al. 2014', 'Tjhung, Nardini, and Cates 2018')
    until = 1500.0
    dt = 30.0

    def __init__(
        self,
        shape: Sequence[int] = (64, 64),
        *,
        mean: float = 0.4,
        kappa: float = 1.0,
        lam: float = 1.0,
        zeta: float = 4.0,
        D: float = 0.05,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.shape = shape
        self.mean = mean
        self.kappa = kappa
        self.lam = lam
        self.zeta = zeta
        self.D = D
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of Active System B+, the conserved field theory of the density of active particles.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        shape = tuple(self.shape)
        rng = np.random.default_rng(self.random_state)
        phi = self.mean + 0.05 * rng.uniform(-1.0, 1.0, shape)
        currents = ActiveCahnHilliardInteraction(
            'currents',
            target='fluid',
            kappa=self.kappa,
            lam=self.lam,
            zeta=self.zeta,
            D=self.D,
        )
        self.system_ = System(
            self.label,
            elements=[ActiveScalarElement('fluid', phi=phi.ravel(), space=LatticeSpace(shape))],
            interactions=[currents],
            solver=EulerMaruyamaSolver(dt=0.02),
            backend=NumbaBackend(),
        )

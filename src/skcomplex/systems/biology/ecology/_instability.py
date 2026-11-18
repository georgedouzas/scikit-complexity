"""Jensen's coupled populations that lose the stability of their steady state.

They lose it from within.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import exp, oo, sqrt
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.laws.utils import bernoulli, clip, normal, where
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.specs import Role
from ....core.systems import BaseSystem, System


class UnstablePopulationsElement(Element):
    """Resident population `x` and invading population `y`.

    Args:
        label:
            The label of the populations.

        x:
            The initial resident population.

        y:
            The initial invading population.

    Examples:
        >>> from skcomplex.systems.biology.ecology import UnstablePopulationsElement
        >>> UnstablePopulationsElement('populations').x
        1.1
    """

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[float, RealSet.closed_open(0, oo)] = 1.1,
        y: Annotated[float, RealSet.closed_open(0, oo)] = 0.0,
    ) -> None:
        super().__init__(label)
        self.x = x
        self.y = y


class ResidentMapInteraction(Interaction):
    """Jensen's map of a resident population damped by an invader, equation (13.3).

    At every step `x ← [x - eta (x - X)] exp(-alpha y) + √x sigma chi`, an Ornstein-Uhlenbeck step damped by the
    invader `y`. The noise `chi` is a standard normal value, scaled by the square root of the population size as in
    birth and death processes, and the size is cut at 0.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the populations, with the resident `x` and the invader `y`.

        eta:
            The restoring rate `eta` of the resident population.

        mean:
            The level `X` the resident population returns to.

        alpha:
            The suppression `alpha` of the resident population by the invader.

        sigma:
            The amplitude `sigma` of the noise.

    Examples:
        >>> from skcomplex.systems.biology.ecology import ResidentMapInteraction
        >>> ResidentMapInteraction('resident', target='populations').eta
        0.1
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        eta: Annotated[float, RealSet.closed_open(0, oo)] = 0.1,
        mean: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        alpha: Annotated[float, RealSet.closed_open(0, oo)] = 0.01,
        sigma: Annotated[float, RealSet.closed_open(0, oo)] = 0.01,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.eta = eta
        self.mean = mean
        self.alpha = alpha
        self.sigma = sigma

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the map of the resident population.

        Args:
            context:
                The namespace of the populations, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            x, y = s.target.x, s.target.y
            resident = (x - s.eta * (x - s.mean)) * exp(-s.alpha * y) + sqrt(x) * s.sigma * normal('chi_x')
            self.laws_ = [UpdateLaw(x, clip(resident, low=0))]


class InvaderMapInteraction(Interaction):
    """Jensen's map of an invading population fed by mutations of a resident, equation (13.4).

    With probability `p` a mutation from the resident `x` adds `nu x` to the invader `y`, and otherwise
    `y ← y (1 + beta) + √y sigma chi'`. The noise `chi'` is a standard normal value, scaled by the square root of the
    population size, and the size is cut at 0.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the populations, with the resident `x` and the invader `y`.

        beta:
            The growth rate `beta` of the invader.

        nu:
            The share `nu` of the resident population that a mutation moves to the invader.

        p:
            The probability of a mutation per step.

        sigma:
            The amplitude `sigma` of the noise.

    Examples:
        >>> from skcomplex.systems.biology.ecology import InvaderMapInteraction
        >>> InvaderMapInteraction('invader', target='populations').p
        0.001
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        beta: Annotated[float, RealSet.closed_open(0, oo)] = 0.01,
        nu: Annotated[float, RealSet.closed_open(0, oo)] = 0.01,
        p: Annotated[float, RealSet.closed_open(0, oo)] = 0.001,
        sigma: Annotated[float, RealSet.closed_open(0, oo)] = 0.01,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.beta = beta
        self.nu = nu
        self.p = p
        self.sigma = sigma

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the map of the invading population.

        Args:
            context:
                The namespace of the populations, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            x, y = s.target.x, s.target.y
            grown = y * (1 + s.beta) + sqrt(y) * s.sigma * normal('chi_y')
            self.laws_ = [UpdateLaw(y, clip(where(bernoulli('mutation', s.p) > 0, y + s.nu * x, grown), low=0))]


class IntrinsicInstabilitySystem(BaseSystem):
    """Jensen's coupled populations, whose steady state becomes unstable.

    It becomes unstable without a change of the surroundings.

    The resident population fluctuates about its level `X = 1` until a rare mutation seeds the invader. The invader then
    grows by the factor `1 + beta` per step and suppresses the resident, so the trajectory leaves the fixed point `(X,
    0)` along its unstable direction, with eigenvalue `beta`, and the resident declines although no parameter has
    changed. The system has the populations `populations.x` and `populations.y`.

    Args:
        x:
            The initial resident population.

        eta:
            The restoring rate of the resident population.

        alpha:
            The suppression of the resident population by the invader.

        beta:
            The growth rate of the invader.

        nu:
            The share of the resident population that a mutation moves to the invader.

        p:
            The probability of a mutation per step.

        sigma:
            The amplitude of the noise.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.ecology import IntrinsicInstabilitySystem
        >>> IntrinsicInstabilitySystem().prepare().system_.interactions[1].beta
        0.01
    """

    draws = True

    field = 'biology/ecology'
    topics = ('tipping point', 'stochastic', 'discrete map', 'early warning')
    references = ('Jensen 2023',)
    until = 2000
    dt = 10

    def __init__(
        self,
        *,
        x: float = 1.1,
        eta: float = 0.1,
        alpha: float = 0.01,
        beta: float = 0.01,
        nu: float = 0.01,
        p: float = 0.001,
        sigma: float = 0.01,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.eta = eta
        self.alpha = alpha
        self.beta = beta
        self.nu = nu
        self.p = p
        self.sigma = sigma

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of Jensen's coupled populations.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[UnstablePopulationsElement('populations', x=self.x)],
            interactions=[
                ResidentMapInteraction(
                    'resident',
                    target='populations',
                    eta=self.eta,
                    alpha=self.alpha,
                    sigma=self.sigma,
                ),
                InvaderMapInteraction(
                    'invader',
                    target='populations',
                    beta=self.beta,
                    nu=self.nu,
                    p=self.p,
                    sigma=self.sigma,
                ),
            ],
        )

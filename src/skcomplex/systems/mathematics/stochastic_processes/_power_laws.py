"""Stochastic process systems of power laws.

The power laws come from exponential growth observed at random times.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.laws import BaseLaw, JumpLaw, RateLaw
from ....core.components.laws.utils import draw
from ....core.components.system import BaseSystem, System
from ....core.engine.solvers import RK4Solver
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class ObservedGrowthElement(Element):
    """Quantity that grows exponentially.

    The quantity is observed and restarted at exponentially distributed times.

    The quantity follows `x' = mu x` from `x0`, and after a waiting time drawn from the exponential distribution of
    rate `lam` it is observed and starts again from `x0`.

    Args:
        label:
            The label of the element or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial value.

        x0:
            The value every growth starts from.

        mu:
            The growth rate.

        lam:
            The rate of the observation times.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import ObservedGrowthElement
        >>> ObservedGrowthElement('growth', lam=2.0).lam
        2.0
    """

    law_classes = (
        JumpLaw,
        RateLaw,
    )

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 1.0,
        x0: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        mu: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        lam: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x0 = x0
        self.mu = mu
        self.lam = lam

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the growth and the restart at the observation times.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        return [
            RateLaw(s.x, s.mu * s.x),
            JumpLaw(waiting=draw('wait', 'expon', scale=1 / s.lam), reset={s.x: s.x0}, label='observations'),
        ]


class ExponentialGrowthAtRandomTimesSystem(BaseSystem):
    """Exponential growth observed at exponentially distributed times.

    The observed values follow a power law.

    Every sample grows as `x0 exp(mu t)` and is observed and restarted after a waiting time drawn from the exponential
    distribution of rate `lam`. The observed values, and the values of the samples at any later time, follow the power
    law `p(x) ~ x^(-lam / mu - 1)` (Thurner et al. 2018, section 3.3.6.1, after Reed and Hughes 2002). The system has
    the group `samples` of state `x` and the observations `samples.observations`.

    Args:
        n_samples:
            The number of independent samples.

        mu:
            The growth rate.

        lam:
            The rate of the observation times.

        x0:
            The value every growth starts from.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import ExponentialGrowthAtRandomTimesSystem
        >>> ExponentialGrowthAtRandomTimesSystem(10).prepare().system_.elements[0].prepare().size_
        10
    """

    field = 'mathematics/stochastic_processes'
    topics = ('power law', 'renewal', 'exponential growth', 'random observation')
    references = ('Reed and Hughes 2002', 'Thurner et al. 2018')
    until = 20.0
    dt = 1.0
    solver = RK4Solver(dt=0.01)

    def __init__(
        self,
        n_samples: int = 2000,
        *,
        mu: float = 1.0,
        lam: float = 1.5,
        x0: float = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_samples = n_samples
        self.mu = mu
        self.lam = lam
        self.x0 = x0

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        return System(
            self.label,
            elements=[
                GroupElement(
                    'samples',
                    element=ObservedGrowthElement,
                    size=self.n_samples,
                    values={'x': self.x0, 'x0': self.x0, 'mu': self.mu, 'lam': self.lam},
                ),
            ],
        )

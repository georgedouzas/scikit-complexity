"""Epidemiology systems: recurrent outbreaks in discrete time.

The immunity lasts a fixed period.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import NN, Expression, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import where
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System

MAX_RECOVERY = 12


class EpidemicCenterElement(Element):
    """City in a discrete-time SIRS epidemic.

    An infection lasts one step and immunity `recovery` steps.

    The state keeps the infected fractions of the last 12 steps, so the recovery time is at most 12.

    Args:
        label:
            The label of the center.

        x:
            The initial fraction of infected individuals.

        a:
            The rate of transmission per contact.

        recovery:
            The number of steps a recovered individual stays immune, at most 12.

        e:
            The fraction of the infected of the center who visit another center.

        values:
            Further values by name, such as the past fractions `'x[-1]'` that the immunity reads, or `None`.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import EpidemicCenterElement
        >>> EpidemicCenterElement('city', x=0.01).recovery
        6
    """

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[float, RealSet.closed_open(0, oo)] = 0.0,
        a: Annotated[float, RealSet.closed_open(0, oo)] = 2.2,
        recovery: Annotated[int, NN] = 6,
        e: Annotated[float, RealSet.closed_open(0, oo)] = 0.005,
        values: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(label, values=values)
        self.x = x
        self.a = a
        self.recovery = recovery
        self.e = e


class DelayedSIRSSystem(BaseSystem):
    """Two cities with recurrent outbreaks of a disease.

    The disease has fixed infection and immunity periods.

    With one infected step and `recovery` immune steps, each city follows `x' = a x (1 - x - x(t-1) - ... -
    x(t-recovery))`, which reduces to the logistic map without immunity. Outbreaks recur like relaxation oscillations,
    and the visits of a fraction `e` of the infected couple the two cities, whose outbreaks lock in or out of phase. The
    system has the centers `first` and `second`.

    Args:
        first:
            The initial infected fraction of the first city.

        second:
            The initial infected fraction of the second city.

        a:
            The rate of transmission per contact.

        recovery:
            The number of steps a recovered individual stays immune, at most 12.

        e:
            The fraction of the infected who visit the other city.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import DelayedSIRSSystem
        >>> DelayedSIRSSystem().prepare().system_.elements[0].x
        0.01
    """

    field = 'biology/epidemiology'
    topics = ('compartments', 'discrete map', 'delay', 'synchronization', 'relaxation oscillation')
    references = ('Gros 2015', 'He and Stone 2003')
    until = 200
    dt = 1

    def __init__(
        self,
        first: float = 0.01,
        second: float = 0.0,
        *,
        a: float = 2.0,
        recovery: int = 6,
        e: float = 0.005,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.first = first
        self.second = second
        self.a = a
        self.recovery = recovery
        self.e = e

    @staticmethod
    def _get_infected_or_immune(center: SimpleNamespace, t: Expression) -> Expression:
        """Return the fraction of a center that cannot be infected.

        The fraction is the one infected now or in the last `recovery` steps.
        """
        past = sum(where(step <= center.recovery, center.x(t=t - step), 0) for step in range(1, MAX_RECOVERY + 1))
        return center.x + past

    def _get_next_infected(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the update `a (x + e y) s` of the infected fraction of a center.

        The center is visited by the infected of another one.

        The center is the target, `y` and `e` are the infected fraction of the other center and the fraction of them who
        visit, and `s = 1 - x - x(t - 1) - ... - x(t - recovery)` is the susceptible fraction of the center.
        """
        home, away = s.target, s.other
        infected = home.a * (home.x + away.e * away.x) * (1 - self._get_infected_or_immune(home, s.t))
        return [UpdateLaw(home.x, infected)]

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of two cities.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        past = {f'x[-{step}]': 0.0 for step in range(1, MAX_RECOVERY + 1)}
        self.system_ = System(
            self.label,
            elements=[
                EpidemicCenterElement('first', x=self.first, a=self.a, recovery=self.recovery, e=self.e, values=past),
                EpidemicCenterElement('second', x=self.second, a=self.a, recovery=self.recovery, e=self.e, values=past),
            ],
            interactions=[
                Interaction(
                    'visits_first',
                    roles={'target': 'first', 'other': 'second'},
                    laws=self._get_next_infected,
                ),
                Interaction(
                    'visits_second',
                    roles={'target': 'second', 'other': 'first'},
                    laws=self._get_next_infected,
                ),
            ],
        )

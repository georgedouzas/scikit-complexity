"""Epidemiology models: recurrent outbreaks in discrete time with a fixed period of immunity."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import NN, Expression, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.interactions import Interaction
from ....core.laws import BaseLaw, UpdateLaw
from ....core.laws.utils import lag, where
from ....core.system import System
from ....specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_model

MAX_RECOVERY = 12


def _infected_or_immune(center: SimpleNamespace) -> Expression:
    """Return the fraction of a center infected now or in the last `recovery` steps, who cannot be infected."""
    past = sum(where(step <= center.recovery, lag(center.x, step), 0) for step in range(1, MAX_RECOVERY + 1))
    return center.x + past


class EpidemicCenterElement(Element):
    """City in a discrete-time SIRS epidemic, where an infection lasts one step and immunity `recovery` steps.

    The state keeps the infected fractions of the last 12 steps, so the recovery time is at most 12.

    Args:
        label:
            The label of the center.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial fraction of infected individuals.

        a:
            The rate of transmission per contact.

        recovery:
            The number of steps a recovered individual stays immune, at most 12.

    Examples:
        >>> from skcomplex.models.biology.epidemiology import EpidemicCenterElement
        >>> EpidemicCenterElement('city', x=0.01).recovery
        6
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[
            float,
            StateSpec(domain=RealSet.closed_open(0, oo), lags=MAX_RECOVERY, initial=(0.0,) * MAX_RECOVERY),
        ] = 0.0,
        a: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.2,
        recovery: Annotated[int, ParameterSpec(domain=NN)] = 6,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.a = a
        self.recovery = recovery


class VisitsInteraction(Interaction):
    """Infection of a center by its own infected and by a fraction of the infected of another center who visit it.

    The next infected fraction is `x' = a (x + e y) s`, where `y` is the infected fraction of the other center and
    `s = 1 - x - x(t - 1) - ... - x(t - recovery)` the susceptible fraction.

    Args:
        label:
            The label of the interaction.

        home:
            The path of the center the infections happen in.

        away:
            The path of the center whose infected visit.

        e:
            The fraction of the infected of the other center who visit.

    Examples:
        >>> from skcomplex.models.biology.epidemiology import VisitsInteraction
        >>> VisitsInteraction('visits', home='first', away='second', e=0.01).e
        0.01
    """

    def __init__(
        self,
        label: str,
        *,
        home: Annotated[str, RoleSpec()],
        away: Annotated[str, RoleSpec()],
        e: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.005,
    ) -> None:
        super().__init__(label)
        self.home = home
        self.away = away
        self.e = e

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the next infected fraction of the home center.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        home = s.home
        susceptible = 1 - _infected_or_immune(home)
        return [UpdateLaw(home.x, home.a * (home.x + s.e * s.away.x) * susceptible)]


@register_model(
    field='biology/epidemiology',
    tags=('compartments', 'discrete map', 'delay', 'synchronization', 'relaxation oscillation'),
    references=('Gros 2015', 'He and Stone 2003'),
    defaults={'until': 200, 'dt': 1},
)
def delayed_sirs(
    first: float = 0.01,
    second: float = 0.0,
    *,
    a: float = 2.0,
    recovery: int = 6,
    e: float = 0.005,
) -> System:
    """Return two cities with recurrent outbreaks of a disease of fixed infection and immunity periods.

    With one infected step and `recovery` immune steps, each city follows `x' = a x (1 - x - x(t-1) - ... -
    x(t-recovery))`, which reduces to the logistic map without immunity. Outbreaks recur like relaxation oscillations,
    and the visits of a fraction `e` of the infected couple the two cities, whose outbreaks lock in or out of phase.

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

    Returns:
        system:
            The model, with the centers `first` and `second`.

    Examples:
        >>> from skcomplex.models.biology.epidemiology import delayed_sirs
        >>> delayed_sirs().elements[0].x
        0.01
    """
    return System(
        elements=[
            EpidemicCenterElement('first', x=first, a=a, recovery=recovery),
            EpidemicCenterElement('second', x=second, a=a, recovery=recovery),
        ],
        interactions=[
            VisitsInteraction('visits_first', home='first', away='second', e=e),
            VisitsInteraction('visits_second', home='second', away='first', e=e),
        ],
    )

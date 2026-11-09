"""Econophysics systems: the kinetic theory of wealth classes.

Welfare policies act on the encounters.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from functools import partial
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import GroupElement, NonnegativeElement
from ....core.components.interactions import BaseInteraction, EncounterInteraction
from ....core.components.laws.utils import moment, total
from ....core.components.spaces import ContinuousTime, LatticeSpace
from ....core.components.systems import BaseSystem, CompositeSystem, RunSettings
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec


class WealthDensityElement(NonnegativeElement):
    """Value `x` of the density of agents at one wealth class.

    A wealth distribution is a group of these elements on the sites of a lattice of wealth classes, one member per
    class, and the exchanges and the welfare policy change it. The element has no laws of its own.

    Args:
        label:
            The label of the element, unique among the parts of its system.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial density, a nonnegative number.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.core.components.spaces import LatticeSpace
        >>> from skcomplex.systems.economics.econophysics import WealthDensityElement
        >>> classes = LatticeSpace(5, boundary='fixed')
        >>> values = {'x': 0.2}
        >>> group = GroupElement('wealth', element=WealthDensityElement, space=classes, on='sites', values=values)
        >>> group.prepare().size_
        5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual, x=x)


def exchange_table(classes: int, cooperation: object, competition: object) -> npt.NDArray[np.object_]:
    """Return the transition table of encounters between wealth classes, `A[h, k, i]`.

    A candidate of class `h` that meets a richer field particle moves one class up with probability `cooperation`
    and one class down with probability `competition`. One that meets a poorer field particle moves one class up with
    probability `competition` and one class down with probability `cooperation`. So competition makes the rich richer
    and the poor poorer, and cooperation does the opposite. A move past the poorest or the richest class is not made.

    Args:
        classes:
            The number of wealth classes.

        cooperation:
            The probability of the move that narrows the gap between the two classes, a number or a parameter.

        competition:
            The probability of the move that widens it, a number or a parameter.

    Returns:
        table:
            The transition probabilities, which sum to one over the outcomes.

    Examples:
        >>> from skcomplex.systems.economics.econophysics import exchange_table
        >>> exchange_table(3, 0.2, 0.3)[0, 2].astype(float).tolist()
        [0.8, 0.2, 0.0]
    """
    table = np.zeros((classes, classes, classes), dtype=object)
    for candidate in range(classes):
        for field in range(classes):
            gap = int(np.sign(field - candidate))
            moves = {candidate + gap: cooperation, candidate - gap: competition} if gap else {}
            for outcome, probability in moves.items():
                if 0 <= outcome < classes:
                    table[candidate, field, outcome] += probability
            table[candidate, field, candidate] += 1.0 - table[candidate, field].sum()
    return table


class WealthExchangeInteraction(EncounterInteraction):
    """Conservative encounters between wealth classes.

    They include cooperation and competition. A candidate of class `h` meets a field particle of class `k` at the rate
    `eta` and moves by the transition table of `exchange_table`.

    Args:
        label:
            The label of the interaction.

        candidate:
            The path of the distribution over wealth classes whose particles move.

        field:
            The path of the distribution over wealth classes they meet, the candidate again.

        classes:
            The number of wealth classes.

        eta:
            The encounter rate.

        cooperation:
            The probability of the move that narrows the gap between the classes of an encounter.

        competition:
            The probability of the move that widens it.

    Examples:
        >>> from skcomplex.systems.economics.econophysics import WealthExchangeInteraction
        >>> WealthExchangeInteraction('exchange', candidate='wealth', field='wealth').competition
        0.3
    """

    def __init__(
        self,
        label: str,
        *,
        candidate: str,
        field: str,
        classes: Annotated[int, ParameterSpec(domain=RealSet.open(0, oo))] = 9,
        eta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        cooperation: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        competition: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
    ) -> None:
        super().__init__(label, candidate=candidate, field=field, rate=self._get_rate, kernel=self._get_kernel)
        self.classes = classes
        self.eta = eta
        self.cooperation = cooperation
        self.competition = competition

    def _get_rate(self, s: SimpleNamespace) -> Expression:
        """Return the encounter rate `eta`."""
        return s.eta

    def _get_kernel(self, s: SimpleNamespace) -> npt.NDArray[np.object_]:
        """Return the transition table of the encounters.

        It uses the cooperation and competition parameters.
        """
        return exchange_table(int(self.classes), s.cooperation, s.competition)


def policy_table(classes: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the rate pattern and the transition table of a welfare policy.

    The policy taxes the rich and helps the poor. The policy moves a candidate of the richest third of the classes one
    class down, and a candidate of the poorest third one class up, and leaves the middle classes alone.

    Args:
        classes:
            The number of wealth classes.

    Returns:
        pattern:
            The rate pattern `eta[h, 0]`, 1 for the classes the policy acts on and 0 for the others.

        table:
            The transition table `A[h, 0, i]` of the policy.

    Examples:
        >>> from skcomplex.systems.economics.econophysics import policy_table
        >>> policy_table(3)[0].ravel().tolist()
        [1.0, 0.0, 1.0]
    """
    pattern = np.zeros((classes, 1))
    table = np.zeros((classes, 1, classes))
    third = max(1, classes // 3)
    for candidate in range(classes):
        outcome = candidate + 1 if candidate < third else candidate - 1 if candidate >= classes - third else candidate
        pattern[candidate, 0] = float(outcome != candidate)
        table[candidate, 0, outcome] = 1.0
    return pattern, table


class WelfarePolicyInteraction(EncounterInteraction):
    """Welfare policy that taxes the richest classes and helps the poorest.

    It is a micro-scale external action. The policy is an encounter whose field role binds a group with a fixed density,
    so it acts on the transitions of the population at the rate `tax` times that density, by the pattern and the table
    of `policy_table`.

    Args:
        label:
            The label of the interaction.

        candidate:
            The path of the distribution over wealth classes.

        field:
            The path of the group of one member whose fixed density is the strength of the policy.

        classes:
            The number of wealth classes.

        tax:
            The rate at which the policy moves the richest classes down and the poorest up.

    Examples:
        >>> from skcomplex.systems.economics.econophysics import WelfarePolicyInteraction
        >>> WelfarePolicyInteraction('welfare', candidate='wealth', field='state').tax
        0.1
    """

    def __init__(
        self,
        label: str,
        *,
        candidate: str,
        field: str,
        classes: Annotated[int, ParameterSpec(domain=RealSet.open(0, oo))] = 9,
        tax: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, candidate=candidate, field=field, rate=self._get_rate, kernel=self._get_kernel)
        self.classes = classes
        self.tax = tax

    def _get_rate(self, s: SimpleNamespace) -> npt.NDArray[np.object_]:
        """Return the rate pattern of the policy times the tax."""
        pattern, _ = policy_table(int(self.classes))
        return pattern * s.tax

    def _get_kernel(self, s: SimpleNamespace) -> npt.NDArray[np.float64]:
        """Return the transition table of the policy."""
        _, table = policy_table(int(self.classes))
        return table


WELFARE_REFERENCES = ('Bellomo et al. 2017', 'Dolfin and Lachowicz 2014', 'Bertotti and Delitala 2008')


def wealth_classes(classes: int) -> npt.NDArray[np.float64]:
    """Return the wealth of every class.

    The wealth is evenly spaced from 0 for the poorest to 1 for the richest.

    Args:
        classes:
            The number of wealth classes.

    Returns:
        wealth:
            The wealth of every class.

    Examples:
        >>> from skcomplex.systems.economics.econophysics import wealth_classes
        >>> wealth_classes(3).tolist()
        [0.0, 0.5, 1.0]
    """
    return np.linspace(0.0, 1.0, classes)


def gini_coefficient(s: SimpleNamespace, wealth: npt.NDArray[np.float64]) -> Expression:
    """Return the Gini coefficient of a distribution over wealth classes.

    Every member of its group reads it. The coefficient is `sum_ij f_i f_j |w_i - w_j| / (2 F M)`, with the density `F`
    and the total wealth `M`.

    Args:
        s:
            The namespace of one member of the distribution.

        wealth:
            The wealth of every class.

    Returns:
        gini:
            The coefficient as an expression of reductions.
    """
    spread = total(s.x, weights=np.abs(wealth[:, None] - wealth[None, :]))
    return total(s.x * spread) / (2 * moment(s.x, 0, wealth) * moment(s.x, 1, wealth))


def wealth_density(s: SimpleNamespace, wealth: npt.NDArray[np.float64]) -> Expression:
    """Return the density of a distribution over wealth classes, its order-0 moment.

    Args:
        s:
            The namespace of one member of the distribution.

        wealth:
            The wealth of every class.

    Returns:
        density:
            The order-0 moment.
    """
    return moment(s.x, 0, wealth)


def mean_wealth(s: SimpleNamespace, wealth: npt.NDArray[np.float64]) -> Expression:
    """Return the mean wealth of a distribution over wealth classes.

    Args:
        s:
            The namespace of one member of the distribution.

        wealth:
            The wealth of every class.

    Returns:
        mean:
            The normalized first moment.
    """
    return moment(s.x, 1, wealth, normalized=True)


def _classes_of(system: CompositeSystem) -> int:
    """Return the number of wealth classes of a system, the size of its distribution."""
    return int(next(item.size or 0 for item in system.elements if item.label == 'wealth'))


def wealth_observables(classes: int) -> list[ObservableSpec]:
    """Return the density, the mean wealth, and the Gini coefficient of a distribution.

    The distribution is `wealth`, over wealth classes.

    Args:
        classes:
            The number of wealth classes.

    Returns:
        observables:
            The observables, each reading the moments of the distribution.

    Examples:
        >>> from skcomplex.systems.economics.econophysics import wealth_observables
        >>> [item.label for item in wealth_observables(9)]
        ['density', 'mean_wealth', 'gini']
    """
    wealth = wealth_classes(classes)
    return [
        ObservableSpec('density', partial(wealth_density, wealth=wealth), group='wealth'),
        ObservableSpec('mean_wealth', partial(mean_wealth, wealth=wealth), group='wealth'),
        ObservableSpec('gini', partial(gini_coefficient, wealth=wealth), group='wealth'),
    ]


class KineticWealthWelfareSystem(BaseSystem):
    """The kinetic theory of wealth classes.

    A positive tax adds a welfare policy. The population is a distribution over wealth classes, a site group on a
    lattice of the classes. Encounters between classes move particles one class up or down, with cooperation narrowing
    the gaps and competition widening them, so the total density stays constant. When competition wins, wealth polarises
    toward the poorest and the richest classes, and the Gini coefficient grows. A welfare policy is an encounter with a
    group of fixed density, which moves the richest third of the classes down and the poorest third up and lowers the
    stationary Gini coefficient.

    The system has the distribution `wealth`, the encounters `exchange`, and with a tax the group `state` and the policy
    `welfare`. Its standard observables are the density, the mean wealth, and the Gini coefficient of `wealth`.

    Args:
        classes:
            The number of wealth classes.

        cooperation:
            The probability of the move that narrows the gap between the classes of an encounter.

        competition:
            The probability of the move that widens it.

        eta:
            The encounter rate.

        tax:
            The rate of the welfare policy, 0 for none.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.economics.econophysics import KineticWealthWelfareSystem
        >>> [item.label for item in KineticWealthWelfareSystem(tax=0.2).prepare().system_.interactions]
        ['exchange', 'welfare']
    """

    field = 'economics/econophysics'
    topics = ('wealth', 'kinetic', 'distribution', 'welfare')
    references = WELFARE_REFERENCES
    until = 100
    dt = 1.0

    def __init__(
        self,
        classes: int = 9,
        *,
        cooperation: float = 0.2,
        competition: float = 0.3,
        eta: float = 1.0,
        tax: float = 0.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.classes = classes
        self.cooperation = cooperation
        self.competition = competition
        self.eta = eta
        self.tax = tax

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        initial = np.exp(-((wealth_classes(self.classes) - 0.5) ** 2) / 0.05)
        components: list[GroupElement] = [
            GroupElement(
                'wealth',
                element=WealthDensityElement,
                size=self.classes,
                on='sites',
                values={'x': initial / initial.sum()},
            ),
        ]
        interactions: list[BaseInteraction] = [
            WealthExchangeInteraction(
                'exchange',
                candidate='wealth',
                field='wealth',
                classes=self.classes,
                eta=self.eta,
                cooperation=self.cooperation,
                competition=self.competition,
            ),
        ]
        if self.tax > 0:
            components.append(GroupElement('state', element=WealthDensityElement, size=1, values={'x': 1.0}))
            interactions.append(
                WelfarePolicyInteraction(
                    'welfare',
                    candidate='wealth',
                    field='state',
                    classes=self.classes,
                    tax=self.tax,
                ),
            )
        return CompositeSystem(
            self.label,
            elements=components,
            interactions=interactions,
            space=LatticeSpace(self.classes, boundary='fixed'),
            time=ContinuousTime(),
        )

    def get_run_settings(self) -> RunSettings:
        """Return the tuned run settings.

        The observables of the parts are built from the model parameters.
        """
        settings = super().get_run_settings()
        system = self._pass_parameters()
        settings['observables'] = wealth_observables(_classes_of(system))
        return settings

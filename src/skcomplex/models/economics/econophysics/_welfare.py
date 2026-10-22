"""Econophysics models: the kinetic theory of wealth classes, with welfare policies that act on the encounters."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from functools import partial

import numpy as np
import numpy.typing as npt

from ....core.base import Domain, Expression, parameter, role, state
from ....core.components import Element, Group
from ....core.interactions import Interaction
from ....core.laws import Encounter, Law, Symbols, moment, total
from ....core.runtime import register_model
from ....core.space import ContinuousTime, LatticeSpace
from ....core.system import Observable, System

WELFARE_REFERENCES = ('Bellomo et al. 2017', 'Dolfin and Lachowicz 2014', 'Bertotti and Delitala 2008')


def wealth_classes(classes: int) -> npt.NDArray[np.float64]:
    """Return the wealth of every class, evenly spaced from 0 for the poorest to 1 for the richest.

    Args:
        classes:
            The number of wealth classes.

    Returns:
        wealth:
            The wealth of every class.

    Examples:
        >>> from skcomplex.models.economics.econophysics import wealth_classes
        >>> wealth_classes(3).tolist()
        [0.0, 0.5, 1.0]
    """
    return np.linspace(0.0, 1.0, classes)


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
        >>> from skcomplex.models.economics.econophysics import exchange_table
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


def policy_table(classes: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the rate pattern and the transition table of a welfare policy that taxes the rich and helps the poor.

    The policy moves a candidate of the richest third of the classes one class down, and a candidate of the poorest
    third one class up, and leaves the middle classes alone.

    Args:
        classes:
            The number of wealth classes.

    Returns:
        pattern:
            The rate pattern `eta[h, 0]`, 1 for the classes the policy acts on and 0 for the others.

        table:
            The transition table `A[h, 0, i]` of the policy.

    Examples:
        >>> from skcomplex.models.economics.econophysics import policy_table
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


def gini_coefficient(s: Symbols, wealth: npt.NDArray[np.float64]) -> Expression:
    """Return the Gini coefficient of a distribution over wealth classes, read by every member of its group.

    The coefficient is `sum_ij f_i f_j |w_i - w_j| / (2 F M)`, with the density `F` and the total wealth `M`.

    Args:
        s:
            The namespace of one member of the distribution.

        wealth:
            The wealth of every class.

    Returns:
        gini:
            The coefficient as an expression of reductions.
    """
    spread = total(s.f, weights=np.abs(wealth[:, None] - wealth[None, :]))
    return total(s.f * spread) / (2 * moment(s.f, 0, wealth) * moment(s.f, 1, wealth))


def wealth_density(s: Symbols, wealth: npt.NDArray[np.float64]) -> Expression:
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
    return moment(s.f, 0, wealth)


def mean_wealth(s: Symbols, wealth: npt.NDArray[np.float64]) -> Expression:
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
    return moment(s.f, 1, wealth, normalized=True)


def _classes_of(system: System) -> int:
    """Return the number of wealth classes of a model, the size of its distribution."""
    return int(next(item.size or 0 for item in system.components if item.label == 'wealth'))


def wealth_observables(classes: int) -> list[Observable]:
    """Return the density, the mean wealth, and the Gini coefficient of a distribution `wealth` over wealth classes.

    Args:
        classes:
            The number of wealth classes.

    Returns:
        observables:
            The observables, each reading the moments of the distribution.

    Examples:
        >>> from skcomplex.models.economics.econophysics import wealth_observables
        >>> [item.label for item in wealth_observables(9)]
        ['density', 'mean_wealth', 'gini']
    """
    wealth = wealth_classes(classes)
    return [
        Observable('density', partial(wealth_density, wealth=wealth), group='wealth'),
        Observable('mean_wealth', partial(mean_wealth, wealth=wealth), group='wealth'),
        Observable('gini', partial(gini_coefficient, wealth=wealth), group='wealth'),
    ]


class WealthClass(Element):
    """Density of the population at one wealth class, a member of a distribution over wealth classes.

    Args:
        label:
            The label of the distribution.

        f:
            The initial density.

    Examples:
        >>> from skcomplex.models.economics.econophysics import WealthClass
        >>> WealthClass.group('wealth', 9, f=1 / 9).size
        9
    """

    f: float = state(default=0.0, domain=Domain.NONNEGATIVE)


class WealthExchange(Interaction):
    """Conservative encounters between wealth classes, with cooperation and competition.

    Args:
        label:
            The label of the interaction.

        population:
            The path of the distribution over wealth classes.

        classes:
            The number of wealth classes.

        eta:
            The encounter rate.

        cooperation:
            The probability of the move that narrows the gap between the classes of an encounter.

        competition:
            The probability of the move that widens it.

    Examples:
        >>> from skcomplex.models.economics.econophysics import WealthExchange
        >>> WealthExchange('exchange', population='wealth').competition
        0.3
    """

    population: str = role()
    classes: int = parameter(default=9, domain=Domain.POSITIVE)
    eta: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    cooperation: float = parameter(default=0.2, domain=Domain.NONNEGATIVE)
    competition: float = parameter(default=0.3, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the encounter of the population with itself.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        table = exchange_table(int(self.classes), s.cooperation, s.competition)
        return [Encounter(s.population.f, s.population.f, s.eta, table)]


class WelfarePolicy(Interaction):
    """Welfare policy that taxes the richest classes and helps the poorest, a micro-scale external action.

    The policy is an encounter whose field role binds a group with a fixed density, so it acts on the transitions of
    the population at the rate `tax` times that density.

    Args:
        label:
            The label of the interaction.

        population:
            The path of the distribution over wealth classes.

        policy:
            The path of the group of one member whose fixed density is the strength of the policy.

        classes:
            The number of wealth classes.

        tax:
            The rate at which the policy moves the richest classes down and the poorest up.

    Examples:
        >>> from skcomplex.models.economics.econophysics import WelfarePolicy
        >>> WelfarePolicy('welfare', population='wealth', policy='state').tax
        0.1
    """

    population: str = role()
    policy: str = role()
    classes: int = parameter(default=9, domain=Domain.POSITIVE)
    tax: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the encounter of the population with the policy.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        pattern, table = policy_table(int(self.classes))
        return [Encounter(s.population.f, s.policy.f, pattern * s.tax, table)]


@register_model(
    field='economics/econophysics',
    tags=('wealth', 'kinetic', 'distribution', 'welfare'),
    references=WELFARE_REFERENCES,
    defaults={'until': 100, 'dt': 1.0},
    observables=lambda system: wealth_observables(_classes_of(system)),
)
def kinetic_wealth_welfare(
    classes: int = 9,
    *,
    cooperation: float = 0.2,
    competition: float = 0.3,
    eta: float = 1.0,
    tax: float = 0.0,
) -> System:
    """Return the kinetic theory of wealth classes, with a welfare policy when the tax is positive.

    The population is a distribution over wealth classes, a site group on a lattice of the classes. Encounters
    between classes move particles one class up or down, with cooperation narrowing the gaps and competition widening
    them, so the total density stays constant. When competition wins, wealth polarises toward the poorest and the
    richest classes, and the Gini coefficient grows. A welfare policy is an encounter with a group of fixed density,
    which moves the richest third of the classes down and the poorest third up and lowers the stationary Gini
    coefficient.

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

    Returns:
        system:
            The model, with the distribution `wealth`, the encounters `exchange`, and with a tax the group `state` and
            the policy `welfare`. Its standard observables are the density, the mean wealth, and the Gini coefficient
            of `wealth`.

    Examples:
        >>> from skcomplex.models.economics.econophysics import kinetic_wealth_welfare
        >>> [item.label for item in kinetic_wealth_welfare(tax=0.2).interactions]
        ['exchange', 'welfare']
    """
    initial = np.exp(-((wealth_classes(classes) - 0.5) ** 2) / 0.05)
    components: list[Group] = [
        Group('wealth', element=WealthClass, size=classes, on='sites', values={'f': initial / initial.sum()}),
    ]
    interactions: list[Interaction] = [
        WealthExchange(
            'exchange',
            population='wealth',
            classes=classes,
            eta=eta,
            cooperation=cooperation,
            competition=competition,
        ),
    ]
    if tax > 0:
        components.append(WealthClass.group('state', 1, f=1.0))
        interactions.append(WelfarePolicy('welfare', population='wealth', policy='state', classes=classes, tax=tax))
    return System(
        components=components,
        interactions=interactions,
        space=LatticeSpace(classes, boundary='fixed'),
        time=ContinuousTime(),
    )

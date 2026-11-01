"""Econophysics models: the kinetic theory of wealth classes, with welfare policies that act on the encounters."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from functools import partial
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression

from ....core.elements import DensityElement, GroupElement
from ....core.interactions import BaseInteraction
from ....core.laws.utils import moment, total
from ....core.spaces import ContinuousTime, LatticeSpace
from ....core.system import System
from ....specs import ObservableSpec
from ..._catalog import register_model
from ._wealth_exchange import WealthExchangeInteraction
from ._welfare_policy import WelfarePolicyInteraction

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


def gini_coefficient(s: SimpleNamespace, wealth: npt.NDArray[np.float64]) -> Expression:
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
    return moment(s.f, 0, wealth)


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
    return moment(s.f, 1, wealth, normalized=True)


def _classes_of(system: System) -> int:
    """Return the number of wealth classes of a model, the size of its distribution."""
    return int(next(item.size or 0 for item in system.elements if item.label == 'wealth'))


def wealth_observables(classes: int) -> list[ObservableSpec]:
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
        ObservableSpec('density', partial(wealth_density, wealth=wealth), group='wealth'),
        ObservableSpec('mean_wealth', partial(mean_wealth, wealth=wealth), group='wealth'),
        ObservableSpec('gini', partial(gini_coefficient, wealth=wealth), group='wealth'),
    ]


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
    components: list[GroupElement] = [
        GroupElement(
            'wealth',
            element=DensityElement,
            size=classes,
            on='sites',
            values={'f': initial / initial.sum()},
        ),
    ]
    interactions: list[BaseInteraction] = [
        WealthExchangeInteraction(
            'exchange',
            candidate='wealth',
            field='wealth',
            classes=classes,
            eta=eta,
            cooperation=cooperation,
            competition=competition,
        ),
    ]
    if tax > 0:
        components.append(GroupElement('state', element=DensityElement, size=1, values={'f': 1.0}))
        interactions.append(
            WelfarePolicyInteraction('welfare', candidate='wealth', field='state', classes=classes, tax=tax),
        )
    return System(
        elements=components,
        interactions=interactions,
        space=LatticeSpace(classes, boundary='fixed'),
        time=ContinuousTime(),
    )

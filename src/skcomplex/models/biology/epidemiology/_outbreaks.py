"""Epidemiology models: outbreaks that confer immunity on networks of heterogeneous degree, in mean field."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import total
from ....core.system import System
from ....specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model


def _recovered_share(member: SimpleNamespace) -> Expression:
    """Return the share of the whole population that a degree class holds as recovered individuals."""
    return member.P * member.r


def _infected_share(member: SimpleNamespace) -> Expression:
    """Return the share of the whole population that a degree class holds as infected individuals."""
    return member.P * member.i


class ImmunizingDegreeClassElement(Element):
    """Susceptible, infected, and recovered densities of the nodes of one degree, in heterogeneous mean field.

    A susceptible node of degree `k` is infected at rate `beta k theta` and an infected node recovers at rate `mu`, so
    `d i_k / dt = -mu i_k + beta k s_k theta` and `d r_k / dt = mu i_k`. On an uncorrelated network, the probability
    that a contact of a susceptible node is infected is `theta = sum_k (k - 1) P(k) i_k / <k>`, a total over the
    degree classes, since an infected neighbour of degree `k` has at most `k - 1` other contacts. The classes are the
    members of a group.

    Args:
        label:
            The label of the class or the group of classes.

        s:
            The initial susceptible density.

        i:
            The initial infected density.

        r:
            The initial recovered density.

        beta:
            The rate of transmission along one contact.

        mu:
            The recovery rate.

        k:
            The degree of the class.

        P:
            The probability `P(k)` of the degree.

    Examples:
        >>> from skcomplex.models.biology.epidemiology import ImmunizingDegreeClassElement
        >>> GroupElement(
        ...     'network',
        ...     element=ImmunizingDegreeClassElement,
        ...     size=3,
        ...     values={'k': [1.0, 2.0, 3.0], 'P': [0.5, 0.3, 0.2]},
        ... ).size
        3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        s: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.999,
        i: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.001,
        r: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        beta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        mu: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        k: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        P: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.s = s
        self.i = i
        self.r = r
        self.beta = beta
        self.mu = mu
        self.k = k
        self.P = P

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the infections along contacts and the recoveries of the class.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        theta = total((s.k - 1) * s.P * s.i) / total(s.k * s.P)
        infections = s.beta * s.k * s.s * theta
        recoveries = s.mu * s.i
        return [RateLaw(s.s, -infections), RateLaw(s.i, infections - recoveries), RateLaw(s.r, recoveries)]


@register_model(
    field='biology/epidemiology',
    tags=('network', 'mean field', 'ode', 'epidemic threshold', 'scale-free', 'final size'),
    references=(
        'Bertin 2021',
        'Moreno, Pastor-Satorras, and Vespignani 2002',
        'Barrat, Barthelemy, and Vespignani 2008',
    ),
    defaults={'until': 100.0, 'dt': 0.5},
    observables=lambda system: [
        ObservableSpec('infected', _infected_share, group='network', reduce='sum'),
        ObservableSpec('recovered', _recovered_share, group='network', reduce='sum'),
    ],
)
def heterogeneous_mean_field_sir(
    exponent: float = 2.5,
    k_min: int = 3,
    k_max: int = 100,
    *,
    beta: float = 0.1,
    mu: float = 1.0,
    infected: float = 1e-4,
) -> System:
    """Return the SIR epidemic on an uncorrelated network of power-law degrees, in heterogeneous mean field.

    The degrees run from `k_min` to `k_max` with `P(k) ~ k^-exponent`. An outbreak grows when
    `beta / mu > <k> / (<k^2> - <k>)`, a threshold close to `<k> / <k^2>` on heterogeneous networks, which vanishes on
    scale-free networks as `k_max` grows. Below it the epidemic dies out after infecting a share of the order of the
    initial one, and above it a finite share of the population ends recovered. The hubs are infected first.

    Args:
        exponent:
            The exponent of the degree distribution.

        k_min:
            The smallest degree.

        k_max:
            The largest degree.

        beta:
            The rate of transmission along one contact.

        mu:
            The recovery rate.

        infected:
            The initial infected density of every class.

    Returns:
        system:
            The model, with the densities `network.s`, `network.i`, and `network.r` of the degree classes from `k_min`
            to `k_max`, members of the group `network`, and the shares `infected` and `recovered` of the whole
            population, the last one the final epidemic size at the end of the outbreak.

    Examples:
        >>> from skcomplex.models.biology.epidemiology import heterogeneous_mean_field_sir
        >>> heterogeneous_mean_field_sir(k_min=2, k_max=5).elements[0].check().prepare().size_
        4
    """
    degrees = np.arange(k_min, k_max + 1, dtype=float)
    weights = degrees**-exponent / np.sum(degrees**-exponent)
    classes = GroupElement(
        'network',
        element=ImmunizingDegreeClassElement,
        size=len(degrees),
        values={'s': 1.0 - infected, 'i': infected, 'r': 0.0, 'beta': beta, 'mu': mu, 'k': degrees, 'P': weights},
    )
    return System(
        elements=[classes],
    )

"""Epidemiology models: outbreaks that confer immunity on networks of heterogeneous degree, in mean field."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ....core.base import Domain, Expression, parameter, state
from ....core.model import Element, Law, Observable, Rate, Symbols, System, total
from ....core.runtime import register_model


def _recovered_share(member: Symbols) -> Expression:
    """Return the share of the whole population that a degree class holds as recovered individuals."""
    return member.P * member.r


def _infected_share(member: Symbols) -> Expression:
    """Return the share of the whole population that a degree class holds as infected individuals."""
    return member.P * member.i


class ImmunizingDegreeClass(Element):
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
        >>> from skcomplex.models.biology.epidemiology import ImmunizingDegreeClass
        >>> ImmunizingDegreeClass.group('network', 3, k=[1.0, 2.0, 3.0], P=[0.5, 0.3, 0.2]).size
        3
    """

    s: float = state(default=0.999, domain=Domain.NONNEGATIVE)
    i: float = state(default=0.001, domain=Domain.NONNEGATIVE)
    r: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    beta: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)
    mu: float = parameter(default=1.0, domain=Domain.POSITIVE)
    k: float = parameter(default=1.0, domain=Domain.POSITIVE)
    P: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
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
        return [Rate(s.s, -infections), Rate(s.i, infections - recoveries), Rate(s.r, recoveries)]


@register_model(
    field='biology/epidemiology',
    tags=('network', 'mean field', 'ode', 'epidemic threshold', 'scale-free', 'final size'),
    references=(
        'Bertin 2021',
        'Moreno, Pastor-Satorras, and Vespignani 2002',
        'Barrat, Barthelemy, and Vespignani 2008',
    ),
    defaults={'until': 100.0, 'dt': 0.5},
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
        >>> heterogeneous_mean_field_sir(k_min=2, k_max=5).elements[0].size
        4
    """
    degrees = np.arange(k_min, k_max + 1, dtype=float)
    weights = degrees**-exponent / np.sum(degrees**-exponent)
    classes = ImmunizingDegreeClass.group(
        'network',
        len(degrees),
        s=1.0 - infected,
        i=infected,
        r=0.0,
        beta=beta,
        mu=mu,
        k=degrees,
        P=weights,
    )
    return System(
        elements=[classes],
        observables=[
            Observable('infected', _infected_share, group='network', reduce='sum'),
            Observable('recovered', _recovered_share, group='network', reduce='sum'),
        ],
    )

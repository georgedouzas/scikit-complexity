"""Epidemiology systems: outbreaks that confer immunity, in mean field.

The outbreaks spread on networks of heterogeneous degree.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import CouplingInteraction, TransferInteraction
from ....core.components.systems import BaseSystem, System
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec


def _recovered_share(member: SimpleNamespace) -> Expression:
    """Return the share of the population a degree class holds as recovered."""
    return member.P * member.r


def _infected_share(member: SimpleNamespace) -> Expression:
    """Return the share of the population a degree class holds as infected."""
    return member.P * member.i


class ImmunizingDegreeClassElement(Element):
    """Susceptible, infected, and recovered densities of the nodes of one degree.

    The densities are in heterogeneous mean field.

    A susceptible node of degree `k` is infected at rate `beta k theta` and an infected node recovers at rate `mu`, so
    `d i_k / dt = -mu i_k + beta k s_k theta` and `d r_k / dt = mu i_k`. On an uncorrelated network, the probability
    that a contact of a susceptible node is infected is `theta = sum_k (k - 1) P(k) i_k / <k>`, a total over the
    degree classes, since an infected neighbour of degree `k` has at most `k - 1` other contacts. The classes are the
    members of a group, and the infections and the recoveries are interactions of the group, which
    `HeterogeneousMeanFieldSIRSystem` adds.

    Args:
        label:
            The label of the class or the group of classes.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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
        >>> from skcomplex.systems.biology.epidemiology import ImmunizingDegreeClassElement
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


class HeterogeneousMeanFieldSIRSystem(BaseSystem):
    """The SIR epidemic on an uncorrelated network of power-law degrees.

    It is in heterogeneous mean field.

    The degrees run from `k_min` to `k_max` with `P(k) ~ k^-exponent`. An outbreak grows when `beta / mu > <k> / (<k^2>
    - <k>)`, a threshold close to `<k> / <k^2>` on heterogeneous networks, which vanishes on scale-free networks as
    `k_max` grows. Below it the epidemic dies out after infecting a share of the order of the initial one, and above it
    a finite share of the population ends recovered. The hubs are infected first. The system has the densities
    `network.s`, `network.i`, and `network.r` of the degree classes from `k_min` to `k_max`, members of the group
    `network`, and the shares `infected` and `recovered` of the whole population, the last one the final epidemic size
    at the end of the outbreak.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import HeterogeneousMeanFieldSIRSystem
        >>> HeterogeneousMeanFieldSIRSystem(k_min=2, k_max=5).prepare().system_.elements[0].prepare().size_
        4
    """

    field = 'biology/epidemiology'
    topics = ('network', 'mean field', 'ode', 'epidemic threshold', 'scale-free', 'final size')
    references = (
        'Bertin 2021',
        'Moreno, Pastor-Satorras, and Vespignani 2002',
        'Barrat, Barthelemy, and Vespignani 2008',
    )
    until = 100.0
    dt = 0.5
    observables = (
        ObservableSpec('infected', _infected_share, group='network', reduce='sum'),
        ObservableSpec('recovered', _recovered_share, group='network', reduce='sum'),
    )

    def __init__(
        self,
        exponent: float = 2.5,
        k_min: int = 3,
        k_max: int = 100,
        *,
        beta: float = 0.1,
        mu: float = 1.0,
        infected: float = 1e-4,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.exponent = exponent
        self.k_min = k_min
        self.k_max = k_max
        self.beta = beta
        self.mu = mu
        self.infected = infected

    @staticmethod
    def _edge_weight(namespace: SimpleNamespace) -> Expression:
        """Return the share `k P(k)` of the ends of edges that another degree class holds."""
        return namespace.other.k * namespace.other.P

    @staticmethod
    def _infection(namespace: SimpleNamespace) -> Expression:
        """Return the infection `beta k s_k (k' - 1) i_k' / k'` of a degree class.

        The infection is by the nodes of another class.
        """
        member, other = namespace.member, namespace.other
        return member.beta * member.k * member.s * (other.k - 1) * other.i / other.k

    def _susceptible_loss(self, namespace: SimpleNamespace) -> Expression:
        """Return the loss of susceptible density of a degree class.

        The loss is to the infection by another class.
        """
        return -self._infection(namespace)

    @staticmethod
    def _recovery(namespace: SimpleNamespace) -> Expression:
        """Return the recovery `mu i_k` of a degree class."""
        return namespace.source.mu * namespace.source.i

    def _build_system(self) -> System:
        """Return the parts of the SIR epidemic on an uncorrelated power-law network."""
        degrees = np.arange(self.k_min, self.k_max + 1, dtype=float)
        weights = degrees**-self.exponent / np.sum(degrees**-self.exponent)
        classes = GroupElement(
            'network',
            element=ImmunizingDegreeClassElement,
            size=len(degrees),
            values={
                's': 1.0 - self.infected,
                'i': self.infected,
                'r': 0.0,
                'beta': self.beta,
                'mu': self.mu,
                'k': degrees,
                'P': weights,
            },
        )
        interactions = [
            CouplingInteraction(
                'infection',
                target='network',
                read='i',
                write='i',
                function=self._infection,
                reduce='normalized-mean',
                mode='group',
                weights=self._edge_weight,
                include_self=True,
            ),
            CouplingInteraction(
                'contagion',
                target='network',
                read='s',
                write='s',
                function=self._susceptible_loss,
                reduce='normalized-mean',
                mode='group',
                weights=self._edge_weight,
                include_self=True,
            ),
            TransferInteraction(
                'recovery',
                source='network',
                target='network',
                state='i',
                into='r',
                amount=self._recovery,
            ),
        ]
        return System(self.label, elements=[classes], interactions=interactions)

"""Probabilistic and mean-field systems: master equations, particle exchange, and Landau relaxation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ....core.components.elements import GroupElement, LatticeElement
from ....core.components.spaces import LatticeSpace
from ....core.components.system import System
from ..._catalog import register_system
from ._microstate import MicrostateElement
from ._order_parameter import OrderParameterElement
from ._zone import ZoneElement


@register_system(
    field='physics/statistical_mechanics',
    tags=('master equation', 'probability', 'detailed balance', 'lattice'),
    references=('Jensen 2023', 'van Kampen 2007'),
    defaults={'until': 500, 'dt': 10},
)
def discrete_master_equation(
    n_states: int = 50,
    *,
    beta: float = 1.0,
    depth: float = 2.0,
    w: float = 0.5,
) -> System:
    """Return the discrete-time master equation of states on a ring, which relaxes to the Boltzmann distribution.

    The potential is `U = depth cos(2 pi i / n_states)`, the probability starts on the state of highest potential,
    and moves to neighbouring states with Glauber probabilities, which satisfy detailed balance.

    Args:
        n_states:
            The number of states.

        beta:
            The inverse temperature.

        depth:
            The amplitude of the potential.

        w:
            The attempt probability of a move to each neighbour, at most 1/2.

    Returns:
        system:
            The system, with the probabilities `states.P` and potentials `states.U`.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import discrete_master_equation
        >>> discrete_master_equation(10).elements[0].prepare().size_
        10
    """
    potential = depth * np.cos(2 * np.pi * np.arange(n_states) / n_states)
    probability = np.zeros(n_states)
    probability[0] = 1.0
    space = LatticeSpace(n_states)
    return System(
        elements=[
            LatticeElement(
                'states',
                element=MicrostateElement,
                space=space,
                values={'P': probability, 'U': potential, 'beta': beta, 'w': w},
            ),
        ],
    )


@register_system(
    field='physics/statistical_mechanics',
    tags=('jump process', 'binomial', 'fluctuations', 'two-phase flow'),
    references=('Barsky 2020',),
    defaults={'until': 20.0, 'dt': 0.5},
)
def two_phase_zone_exchange(
    n_zones: int = 100,
    *,
    Na: float = 100.0,
    p: float = 0.1,
    k: float = 1.0,
) -> System:
    """Return independent zones of a two-phase apparatus, each exchanging particles with the rest of the apparatus.

    The zones start empty and their numbers of particles settle on the binomial distribution of mean `Na p` and
    variance `Na p (1 - p)`.

    Args:
        n_zones:
            The number of independent zones.

        Na:
            The number of particles in the apparatus and a zone.

        p:
            The volume fraction of a zone.

        k:
            The rate of exchange.

    Returns:
        system:
            The system, with the numbers `zones.N`.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import two_phase_zone_exchange
        >>> two_phase_zone_exchange(5).elements[0].prepare().size_
        5
    """
    return System(
        elements=[GroupElement('zones', element=ZoneElement, size=n_zones, values={'Na': Na, 'p': p, 'k': k})],
    )


@register_system(
    field='physics/statistical_mechanics',
    tags=('ode', 'mean field', 'phase transition', 'critical slowing down'),
    references=('Landau 1937', 'Gros 2015'),
    defaults={'until': 20.0, 'dt': 0.1},
)
def landau_relaxation(
    phi: float = 0.1,
    *,
    T: float = 0.5,
    Tc: float = 1.0,
    a: float = 1.0,
    b: float = 1.0,
    h: float = 0.0,
    Gamma: float = 1.0,
) -> System:
    """Return the relaxation of a Landau order parameter, to `sqrt(a (Tc - T) / b)` below `Tc` and to 0 above it.

    The relaxation time `1 / (Gamma a |T - Tc|)` diverges at the critical temperature, which is critical slowing
    down.

    Args:
        phi:
            The initial order parameter.

        T:
            The temperature.

        Tc:
            The critical temperature.

        a:
            The coefficient of the quadratic term per unit temperature.

        b:
            The coefficient of the quartic term.

        h:
            The external field.

        Gamma:
            The kinetic coefficient.

    Returns:
        system:
            The system, with the order parameter `magnet.phi`.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import landau_relaxation
        >>> landau_relaxation(T=2.0).elements[0].T
        2.0
    """
    return System(elements=[OrderParameterElement('magnet', phi=phi, a=a, b=b, T=T, Tc=Tc, h=h, Gamma=Gamma)])

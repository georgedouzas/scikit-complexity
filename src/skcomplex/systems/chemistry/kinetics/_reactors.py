"""Reactors with surface kinetics: catalyst pellets and pulsed electrodes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression

from ....core.components.interactions import ReactionInteraction, RelaxationInteraction
from ....core.components.system import System
from ..._catalog import register_system
from ._catalyst_pellet import CatalystPelletElement
from ._pulsed_electrode import PulsedElectrodeElement


def _get_surface_reaction(s: SimpleNamespace) -> Expression:
    """Return the Langmuir-Hinshelwood propensity `k c / (1 + K c)^2` of the reaction at the surface of a pellet."""
    pellet = s.reactants[0]
    return pellet.k * pellet.c / (1 + pellet.K * pellet.c) ** 2


def _get_transfer_time(s: SimpleNamespace) -> Expression:
    """Return the time `1 / h` of the mass transfer from the bulk to the surface of a pellet."""
    return 1 / s.target.h


@register_system(
    field='chemistry/kinetics',
    tags=('ode', 'catalysis', 'bistability', 'Langmuir-Hinshelwood'),
    references=('Trzaska 2023', 'Aris 1975'),
    defaults={'until': 100.0, 'dt': 1.0},
)
def catalyst_pellet(c: float = 1.0, *, h: float = 0.1, k: float = 10.0, K: float = 20.0) -> System:
    """Return the surface concentration of a catalyst pellet, bistable through Langmuir-Hinshelwood kinetics.

    The concentration at the surface, relative to the bulk, follows `dc/dt = h (1 - c) - k c / (1 + K c)^2`, where
    mass transfer from the bulk balances a reaction that the adsorbed reactant inhibits. For the default parameters a
    reacting state of low concentration and an inhibited state of high concentration are both stable.

    Args:
        c:
            The initial surface concentration.

        h:
            The coefficient of mass transfer from the bulk.

        k:
            The rate constant of the surface reaction.

        K:
            The adsorption constant, which sets the inhibition.

    Returns:
        system:
            The system, with the concentration `pellet.c`.

    Examples:
        >>> from skcomplex.systems.chemistry.kinetics import catalyst_pellet
        >>> catalyst_pellet(k=5.0).elements[0].k
        5.0
    """
    return System(
        elements=[CatalystPelletElement('pellet', c=c, h=h, k=k, K=K)],
        interactions=[
            RelaxationInteraction('transfer', target='pellet', state='c', goal=1.0, time=_get_transfer_time),
            ReactionInteraction('reaction', reactants={('pellet', 'c'): 1}, propensity=_get_surface_reaction),
        ],
    )


@register_system(
    field='chemistry/kinetics',
    tags=('ode', 'electrochemistry', 'pulses', 'driven'),
    references=('Trzaska 2023', 'Bard and Faulkner 2001'),
    defaults={'until': 60.0, 'dt': 0.1},
)
def pulsed_electrochemical_reactor(
    eta: float = 0.0,
    c: float = 1.0,
    *,
    C: float = 1.0,
    alpha: float = 0.5,
    tau: float = 10.0,
    kappa: float = 0.05,
    current: float = 5.0,
    period: float = 10.0,
    width: float = 2.0,
) -> System:
    """Return a stirred electrochemical reactor whose electrode is driven by a train of current pulses.

    Every pulse charges the double layer, the overpotential drives a Butler-Volmer current that consumes the reactant,
    and the feed restores it between pulses.

    Args:
        eta:
            The initial overpotential.

        c:
            The initial reactant concentration, relative to the feed.

        C:
            The capacitance of the double layer.

        alpha:
            The transfer coefficient.

        tau:
            The residence time of the reactor.

        kappa:
            The consumption of reactant per unit of Faradaic current.

        current:
            The current of a pulse.

        period:
            The time between the starts of pulses.

        width:
            The duration of a pulse.

    Returns:
        system:
            The system, with the overpotential `electrode.eta` and the concentration `electrode.c`.

    Examples:
        >>> from skcomplex.systems.chemistry.kinetics import pulsed_electrochemical_reactor
        >>> pulsed_electrochemical_reactor(width=1.0).elements[0].width
        1.0
    """
    electrode = PulsedElectrodeElement(
        'electrode',
        eta=eta,
        c=c,
        C=C,
        alpha=alpha,
        tau=tau,
        kappa=kappa,
        current=current,
        period=period,
        width=width,
    )
    return System(elements=[electrode])

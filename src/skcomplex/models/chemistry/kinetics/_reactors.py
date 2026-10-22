"""Reactors with surface kinetics: catalyst pellets and pulsed electrodes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import exp

from ....core.base import Domain, parameter, state
from ....core.components import Element
from ....core.laws import Law, Rate, Symbols, pulse_train
from ....core.runtime import register_model
from ....core.system import System

CatalystPellet = Element.define(
    'CatalystPellet',
    states={'c': state(default=1.0, domain=Domain.NONNEGATIVE)},
    parameters={'h': 0.1, 'k': 10.0, 'K': 20.0},
    laws=lambda s: [Rate(s.c, s.h * (1 - s.c) - s.k * s.c / (1 + s.K * s.c) ** 2)],
)


class PulsedElectrode(Element):
    """Electrode of a stirred electrochemical reactor, charged by a train of current pulses.

    In units of the exchange current and of the thermal voltage, the overpotential follows
    `C deta/dt = i(t) - i_F`, with the Butler-Volmer current `i_F = c exp(alpha eta) - exp(-(1 - alpha) eta)`, and the
    reactant concentration follows `dc/dt = (1 - c) / tau - kappa i_F`.

    Args:
        label:
            The label of the electrode.

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

    Examples:
        >>> from skcomplex.models.chemistry.kinetics import PulsedElectrode
        >>> PulsedElectrode('electrode').period
        10.0
    """

    eta: float = state(default=0.0, domain=Domain.REAL)
    c: float = state(default=1.0, domain=Domain.NONNEGATIVE)
    C: float = parameter(default=1.0, domain=Domain.POSITIVE)
    alpha: float = parameter(default=0.5, domain=Domain.POSITIVE)
    tau: float = parameter(default=10.0, domain=Domain.POSITIVE)
    kappa: float = parameter(default=0.05, domain=Domain.NONNEGATIVE)
    current: float = parameter(default=5.0, domain=Domain.REAL)
    period: float = 10.0
    width: float = 2.0

    def laws(self, s: Symbols) -> list[Law]:
        """Return the charging of the double layer and the balance of the reactant.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        faradaic = s.c * exp(s.alpha * s.eta) - exp(-(1 - s.alpha) * s.eta)
        applied = s.current * pulse_train(s.t, self.period, self.width)
        return [
            Rate(s.eta, (applied - faradaic) / s.C),
            Rate(s.c, (1 - s.c) / s.tau - s.kappa * faradaic),
        ]


@register_model(
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
            The model, with the concentration `pellet.c`.

    Examples:
        >>> from skcomplex.models.chemistry.kinetics import catalyst_pellet
        >>> catalyst_pellet(k=5.0).elements[0].k
        5.0
    """
    return System(components=[CatalystPellet('pellet', c=c, h=h, k=k, K=K)])


@register_model(
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
            The model, with the overpotential `electrode.eta` and the concentration `electrode.c`.

    Examples:
        >>> from skcomplex.models.chemistry.kinetics import pulsed_electrochemical_reactor
        >>> pulsed_electrochemical_reactor(width=1.0).elements[0].width
        1.0
    """
    electrode = PulsedElectrode(
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
    return System(components=[electrode])

"""Reactors with surface kinetics: catalyst pellets and pulsed electrodes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import exp, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import pulse_train
from ....core.system import System
from ....engine.runtime import register_model
from ....specs import ParameterSpec, StateSpec, VisualSpec


class CatalystPelletElement(Element):
    """Catalyst pellet element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        c: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        h: Annotated[object, ParameterSpec()] = 0.1,
        k: Annotated[object, ParameterSpec()] = 10.0,
        K: Annotated[object, ParameterSpec()] = 20.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.c = c
        self.h = h
        self.k = k
        self.K = K

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the catalyst pellet element."""
        return [RateLaw(s.c, s.h * (1 - s.c) - s.k * s.c / (1 + s.K * s.c) ** 2)]


class PulsedElectrodeElement(Element):
    """Electrode of a stirred electrochemical reactor, charged by a train of current pulses.

    In units of the exchange current and of the thermal voltage, the overpotential follows
    `C deta/dt = i(t) - i_F`, with the Butler-Volmer current `i_F = c exp(alpha eta) - exp(-(1 - alpha) eta)`, and the
    reactant concentration follows `dc/dt = (1 - c) / tau - kappa i_F`.

    Args:
        label:
            The label of the electrode.

        period:
            The time between the starts of pulses.

        width:
            The duration of a pulse.

        **values:
            The values of the declared variables, by name:

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

    Examples:
        >>> from skcomplex.models.chemistry.kinetics import PulsedElectrodeElement
        >>> PulsedElectrodeElement('electrode').period
        10.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        eta: Annotated[float, StateSpec()] = 0.0,
        c: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        C: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        tau: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        kappa: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        current: Annotated[float, ParameterSpec()] = 5.0,
        period: float = 10.0,
        width: float = 2.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.eta = eta
        self.c = c
        self.C = C
        self.alpha = alpha
        self.tau = tau
        self.kappa = kappa
        self.current = current
        self.period = period
        self.width = width

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
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
            RateLaw(s.eta, (applied - faradaic) / s.C),
            RateLaw(s.c, (1 - s.c) / s.tau - s.kappa * faradaic),
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
    return System(elements=[CatalystPelletElement('pellet', c=c, h=h, k=k, K=K)])


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

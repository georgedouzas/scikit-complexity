"""Reactors with surface kinetics: catalyst pellets and pulsed electrodes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, exp, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import pulse_train
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ...general.reactions import ReactionInteraction
from ._mass_exchange import MassExchangeInteraction


class CatalystPelletElement(Element):
    """Catalyst pellet, the concentration `c` of the reactant at its surface.

    The concentration is relative to the bulk.

    The mass transfer and the surface reaction of `CatalystPelletSystem` change the concentration and read their
    constants from the pellet.

    Args:
        label:
            The label of the pellet.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        c:
            The initial surface concentration.

        h:
            The coefficient of mass transfer from the bulk.

        k:
            The rate constant of the surface reaction.

        K:
            The adsorption constant, which sets the inhibition.

    Examples:
        >>> from skcomplex.systems.chemistry.kinetics import CatalystPelletElement
        >>> CatalystPelletElement('pellet').K
        20.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        c: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        h: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.1,
        k: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        K: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 20.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.c = c
        self.h = h
        self.k = k
        self.K = K


class PulsedElectrodeElement(Element):
    """Electrode of a stirred electrochemical reactor.

    A train of current pulses charges it.

    In units of the exchange current and of the thermal voltage, the overpotential follows
    `C deta/dt = i(t) - i_F`, with the Butler-Volmer current `i_F = c exp(alpha eta) - exp(-(1 - alpha) eta)`, and the
    reactant concentration follows `dc/dt = (1 - c) / tau - kappa i_F`.

    Args:
        label:
            The label of the electrode.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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
        >>> from skcomplex.systems.chemistry.kinetics import PulsedElectrodeElement
        >>> PulsedElectrodeElement('electrode').period
        10.0
    """

    law_classes = (RateLaw,)

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


class CatalystPelletSystem(BaseSystem):
    """The surface concentration of a catalyst pellet.

    It is bistable through Langmuir-Hinshelwood kinetics.

    The concentration at the surface, relative to the bulk, follows `dc/dt = h (1 - c) - k c / (1 + K c)^2`, where
    mass transfer from the bulk balances a reaction that the adsorbed reactant inhibits. For the default parameters a
    reacting state of low concentration and an inhibited state of high concentration are both stable. The
    concentration is `pellet.c`.

    Args:
        c:
            The initial surface concentration.

        h:
            The coefficient of mass transfer from the bulk.

        k:
            The rate constant of the surface reaction.

        K:
            The adsorption constant, which sets the inhibition.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.chemistry.kinetics import CatalystPelletSystem
        >>> CatalystPelletSystem(k=5.0).prepare().system_.elements[0].k
        5.0
    """

    field = 'chemistry/kinetics'
    topics = ('ode', 'catalysis', 'bistability', 'Langmuir-Hinshelwood')
    references = ('Trzaska 2023', 'Aris 1975')
    until = 100.0
    dt = 1.0

    def __init__(
        self,
        c: float = 1.0,
        *,
        h: float = 0.1,
        k: float = 10.0,
        K: float = 20.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.c = c
        self.h = h
        self.k = k
        self.K = K

    @staticmethod
    def _get_surface_reaction(s: SimpleNamespace) -> Expression:
        """Return the Langmuir-Hinshelwood propensity `k c / (1 + K c)^2`.

        It is the propensity of the reaction at the surface of a pellet.
        """
        pellet = s.reactants[0]
        return pellet.k * pellet.c / (1 + pellet.K * pellet.c) ** 2

    @staticmethod
    def _get_transfer_time(s: SimpleNamespace) -> Expression:
        """Return the time `1 / h` of the mass transfer to the surface of a pellet.

        The transfer is from the bulk.
        """
        return 1 / s.target.h

    def _build_system(self) -> CompositeSystem:
        """Return the pellet, the mass transfer, and the reaction at the surface.

        The mass transfer is from the bulk.
        """
        return CompositeSystem(
            self.label,
            elements=[CatalystPelletElement('pellet', c=self.c, h=self.h, k=self.k, K=self.K)],
            interactions=[
                MassExchangeInteraction('transfer', target='pellet', state='c', goal=1.0, time=self._get_transfer_time),
                ReactionInteraction('reaction', reactants={('pellet', 'c'): 1}, propensity=self._get_surface_reaction),
            ],
        )


class PulsedElectrochemicalReactorSystem(BaseSystem):
    """A stirred electrochemical reactor driven by current pulses.

    A train of current pulses drives its electrode.

    Every pulse charges the double layer, the overpotential drives a Butler-Volmer current that consumes the reactant,
    and the feed restores it between pulses. The overpotential is `electrode.eta` and the concentration
    `electrode.c`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.chemistry.kinetics import PulsedElectrochemicalReactorSystem
        >>> PulsedElectrochemicalReactorSystem(width=1.0).prepare().system_.elements[0].width
        1.0
        >>> sorted(PulsedElectrochemicalReactorSystem().get_tags().laws)
        ['rate']
    """

    field = 'chemistry/kinetics'
    topics = ('ode', 'electrochemistry', 'pulses', 'driven')
    references = ('Trzaska 2023', 'Bard and Faulkner 2001')
    until = 60.0
    dt = 0.1

    def __init__(
        self,
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
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.eta = eta
        self.c = c
        self.C = C
        self.alpha = alpha
        self.tau = tau
        self.kappa = kappa
        self.current = current
        self.period = period
        self.width = width

    def _build_system(self) -> CompositeSystem:
        """Return the electrode.

        Its own laws charge the double layer and balance the reactant.
        """
        electrode = PulsedElectrodeElement(
            'electrode',
            eta=self.eta,
            c=self.c,
            C=self.C,
            alpha=self.alpha,
            tau=self.tau,
            kappa=self.kappa,
            current=self.current,
            period=self.period,
            width=self.width,
        )
        return CompositeSystem(self.label, elements=[electrode])

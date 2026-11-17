"""Reactors with surface kinetics: catalyst pellets and pulsed electrodes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

from sage.all__sagemath_symbolics import Expression, exp, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import pulse_train
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.specs import ParameterSpec, Role
from ....core.systems import BaseSystem, System
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
        *,
        c: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        h: Annotated[float, RealSet.open(0, oo)] = 0.1,
        k: Annotated[float, RealSet.closed_open(0, oo)] = 10.0,
        K: Annotated[float, RealSet.closed_open(0, oo)] = 20.0,
    ) -> None:
        super().__init__(label)
        self.c = c
        self.h = h
        self.k = k
        self.K = K


class PulsedElectrodeElement(Element):
    """Electrode of a stirred electrochemical reactor, with its overpotential `eta` and reactant concentration `c`.

    The electrode kinetics of `PulsedElectrochemicalReactorSystem` change both.

    Args:
        label:
            The label of the electrode.

        eta:
            The initial overpotential.

        c:
            The initial reactant concentration, relative to the feed.

    Examples:
        >>> from skcomplex.systems.chemistry.kinetics import PulsedElectrodeElement
        >>> PulsedElectrodeElement('electrode', c=0.5).c
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        eta: float = 0.0,
        c: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.eta = eta
        self.c = c


class ElectrodeKineticsInteraction(Interaction):
    """Kinetics of an electrode that a train of current pulses charges.

    In units of the exchange current and of the thermal voltage, the overpotential follows
    `C deta/dt = i(t) - i_F`, with the Butler-Volmer current `i_F = c exp(alpha eta) - exp(-(1 - alpha) eta)`, and the
    reactant concentration follows `dc/dt = (1 - c) / tau - kappa i_F`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the electrode.

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
        >>> from skcomplex.systems.chemistry.kinetics import ElectrodeKineticsInteraction
        >>> ElectrodeKineticsInteraction('kinetics', target='electrode').period
        10.0
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        C: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        tau: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        kappa: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        current: Annotated[float, ParameterSpec()] = 5.0,
        period: float = 10.0,
        width: float = 2.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.C = C
        self.alpha = alpha
        self.tau = tau
        self.kappa = kappa
        self.current = current
        self.period = period
        self.width = width

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the charging of the double layer and the balance of the reactant.

        Args:
            context:
                The space the interaction inherits from its system, or the namespace of the electrode.
        """
        super()._prepare(context)
        if not isinstance(context, SimpleNamespace):
            return
        s, eta, c = context, context.target.eta, context.target.c
        faradaic = c * exp(s.alpha * eta) - exp(-(1 - s.alpha) * eta)
        applied = s.current * pulse_train(s.t, self.period, self.width)
        self.laws_: list[BaseComponent] = [
            RateLaw(eta.diff(s.t), (applied - faradaic) / s.C),
            RateLaw(c.diff(s.t), (1 - c) / s.tau - s.kappa * faradaic),
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
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

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the pellet, the mass transfer, and the reaction at the surface.

        The mass transfer is from the bulk.
        """
        super()._prepare(context)
        self.system_ = System(
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.chemistry.kinetics import PulsedElectrochemicalReactorSystem
        >>> PulsedElectrochemicalReactorSystem(width=1.0).prepare().system_.interactions[0].width
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.eta = eta
        self.c = c
        self.C = C
        self.alpha = alpha
        self.tau = tau
        self.kappa = kappa
        self.current = current
        self.period = period
        self.width = width

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the electrode and the kinetics that charge the double layer and balance the reactant."""
        super()._prepare(context)
        kinetics = ElectrodeKineticsInteraction(
            'kinetics',
            target='electrode',
            C=self.C,
            alpha=self.alpha,
            tau=self.tau,
            kappa=self.kappa,
            current=self.current,
            period=self.period,
            width=self.width,
        )
        electrode = PulsedElectrodeElement('electrode', eta=self.eta, c=self.c)
        self.system_ = System(self.label, elements=[electrode], interactions=[kinetics])

"""Climate variability.

The module covers stochastic climate and the delayed oscillator of El Niño.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import NoiseLaw, RateLaw
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, EulerMaruyamaSolver
from ....core.specs import Role
from ....core.systems import BaseSystem, System


class ClimateAnomalyElement(Element):
    """Slow temperature anomaly of Hasselmann's stochastic climate.

    The feedback of `HasselmannSystem` damps the anomaly and weather noise drives it.

    Args:
        label:
            The label of the anomaly.

        T:
            The initial temperature anomaly.

    Examples:
        >>> from skcomplex.systems.physics.climate import ClimateAnomalyElement
        >>> ClimateAnomalyElement('ocean', T=0.5).T
        0.5
    """

    def __init__(self, label: str, *, T: float = 0.0) -> None:
        super().__init__(label)
        self.T = T


class ClimateFeedbackInteraction(Interaction):
    """Feedback that damps a temperature anomaly under weather noise.

    The anomaly `T` of the target follows `dT = -lam T dt + sigma dW`, the relaxation to zero over the time `1 / lam`
    with white weather noise of amplitude `sigma`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the anomaly.

        lam:
            The feedback rate that damps the anomaly.

        sigma:
            The amplitude of the weather noise.

    Examples:
        >>> from skcomplex.systems.physics.climate import ClimateFeedbackInteraction
        >>> ClimateFeedbackInteraction('feedback', target='ocean').lam
        0.5
    """

    law_classes = (RateLaw, NoiseLaw)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        lam: Annotated[float, RealSet.open(0, oo)] = 0.5,
        sigma: Annotated[float, RealSet.closed_open(0, oo)] = 0.3,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.lam = lam
        self.sigma = sigma

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the relaxation of the anomaly and its weather noise.

        Args:
            context:
                The namespace of the anomaly, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            T = context.target.T
            self.laws_ = [RateLaw(T.diff(context.t), (0.0 - T) / (1 / context.lam)), NoiseLaw(T, context.sigma)]


class HasselmannSystem(BaseSystem):
    """Hasselmann's stochastic climate.

    Fast weather noise drives a slow, damped temperature anomaly.

    The anomaly follows `dT = -lam T dt + sigma dW`, an Ornstein-Uhlenbeck process whose red spectrum integrates white
    weather noise, with the stationary variance `sigma^2 / (2 lam)`. The system runs with `EulerMaruyamaSolver(dt=0.01)`
    and has the anomaly `ocean.T`.

    Args:
        T:
            The initial temperature anomaly.

        lam:
            The feedback rate that damps the anomaly.

        sigma:
            The amplitude of the weather noise.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.climate import HasselmannSystem
        >>> HasselmannSystem(lam=0.1).prepare().system_.interactions[0].lam
        0.1
    """

    field = 'physics/climate'
    topics = ('sde', 'Ornstein-Uhlenbeck', 'red noise', 'stochastic climate')
    references = ('Hasselmann 1976', 'Ghil and Lucarini 2020')
    until = 100.0
    dt = 0.1

    def __init__(
        self,
        T: float = 0.0,
        *,
        lam: float = 0.5,
        sigma: float = 0.3,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.T = T
        self.lam = lam
        self.sigma = sigma

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the anomaly damped by its feedback under weather noise.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[ClimateAnomalyElement('ocean', T=self.T)],
            interactions=[ClimateFeedbackInteraction('feedback', target='ocean', lam=self.lam, sigma=self.sigma)],
            solver=EulerMaruyamaSolver(dt=0.01),
        )


class DelayedOscillatorInteraction(Interaction):
    """Local growth and delayed feedback of a temperature anomaly.

    The anomaly `T` of the target follows `T' = T - T^3 - alpha T(t - delta)`, a growth saturated by the cube and
    damped by its own value a delay `delta` earlier.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the anomaly.

        alpha:
            The strength of the delayed negative feedback.

        delta:
            The delay of the feedback.

    Examples:
        >>> from skcomplex.systems.physics.climate import DelayedOscillatorInteraction
        >>> DelayedOscillatorInteraction('oscillator', target='pacific', delta=4.0).delta
        4.0
    """

    law_classes = (RateLaw,)

    def __init__(self, label: str, *, target: Role, alpha: float = 0.75, delta: float = 6.0) -> None:
        super().__init__(label)
        self.target = target
        self.alpha = alpha
        self.delta = delta

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rate of the anomaly with its delayed feedback.

        Args:
            context:
                The namespace of the anomaly, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            T, t = context.target.T, context.t
            self.laws_ = [RateLaw(T.diff(t), T - T**3 - context.alpha * T(t=t - context.delta))]


class ENSOSystem(BaseSystem):
    """The delayed oscillator of El Niño of Suarez and Schopf.

    The oscillator is `dT/dt = T - T^3 - alpha T(t - delta)`.

    The sea surface temperature anomaly of the eastern Pacific grows by local feedback and is damped by the delayed
    return of oceanic waves reflected at the western boundary. For `alpha < 1` and a long enough delay the warm and cold
    states lose stability to an oscillation between El Niño and La Niña. The history before the start is the initial
    anomaly. The system has the anomaly `pacific.T`.

    Args:
        T:
            The initial temperature anomaly, also its history.

        alpha:
            The strength of the delayed negative feedback.

        delta:
            The delay of the wave transit.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.climate import ENSOSystem
        >>> ENSOSystem(delta=4.0).prepare().system_.interactions[0].delta
        4.0
    """

    field = 'physics/climate'
    topics = ('delay', 'oscillation', 'El Niño', 'ENSO')
    references = ('Suarez and Schopf 1988', 'Battisti and Hirst 1989', 'Ghil and Lucarini 2020')
    until = 200.0
    dt = 0.5

    def __init__(
        self,
        T: float = 0.1,
        *,
        alpha: float = 0.75,
        delta: float = 6.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.T = T
        self.alpha = alpha
        self.delta = delta

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the anomaly of the eastern Pacific and its delayed feedback.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[Element('pacific', values={'T': self.T})],
            interactions=[
                DelayedOscillatorInteraction('oscillator', target='pacific', alpha=self.alpha, delta=self.delta),
            ],
        )

"""Paleoclimate oscillators.

The module covers glacial cycles and the excitable carbon cycle.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace

from sage.all__sagemath_symbolics import cos, pi

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import RateLaw
from ....core.components.laws.utils import hill
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role


class SaltzmanMaaschElement(Element):
    """Climate of the Saltzman-Maasch glacial cycles.

    The climate holds the anomalies of global ice volume `x`, atmospheric carbon dioxide `y`, and deep ocean
    temperature `z`.

    Args:
        label:
            The label of the climate.

        x:
            The initial ice volume anomaly.

        y:
            The initial carbon dioxide anomaly.

        z:
            The initial deep ocean temperature anomaly.

    Examples:
        >>> from skcomplex.systems.physics.climate import SaltzmanMaaschElement
        >>> SaltzmanMaaschElement('climate').x
        -0.2
    """

    def __init__(self, label: str, *, x: float = -0.2, y: float = 1.0, z: float = 0.0) -> None:
        super().__init__(label)
        self.x = x
        self.y = y
        self.z = z


class SaltzmanMaaschInteraction(Interaction):
    """Feedbacks of ice volume, carbon dioxide, and deep ocean temperature, with the orbital forcing.

    The anomalies of the target follow `dx/dt = -x - y - u z - nu F(t)`, `dy/dt = -p z + r y - s y^2 - y^3`, and
    `dz/dt = -q (x + z)`, where `F(t) = cos(2π t0 t / 22.1) - cos(2π t0 t / 44.1)` is the orbital forcing of
    precession and obliquity.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the climate.

        p:
            The effect of the deep ocean temperature on carbon dioxide.

        q:
            The relaxation rate of the deep ocean temperature to the ice volume.

        r:
            The growth rate of carbon dioxide by its own feedback.

        s:
            The quadratic saturation of carbon dioxide.

        u:
            The effect of the deep ocean temperature on ice volume.

        nu:
            The amplitude of the orbital forcing.

        t0:
            The time unit in thousands of years, which scales the orbital periods.

    Examples:
        >>> from skcomplex.systems.physics.climate import SaltzmanMaaschInteraction
        >>> SaltzmanMaaschInteraction('feedbacks', target='climate').q
        2.5
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        p: float = 1.0,
        q: float = 2.5,
        r: float = 1.3,
        s: float = 0.6,
        u: float = 0.2,
        nu: float = 0.0,
        t0: float = 10.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.p = p
        self.q = q
        self.r = r
        self.s = s
        self.u = u
        self.nu = nu
        self.t0 = t0

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rates of the three anomalies.

        Args:
            context:
                The namespace of the climate, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x, y, z, t = context.target.x, context.target.y, context.target.z, context.t
            forcing = cos(2 * pi * context.t0 * t / 22.1) - cos(2 * pi * context.t0 * t / 44.1)
            self.laws_ = [
                RateLaw(x.diff(t), -x - y - context.u * z - context.nu * forcing),
                RateLaw(y.diff(t), -context.p * z + context.r * y - context.s * y**2 - y**3),
                RateLaw(z.diff(t), -context.q * (x + z)),
            ]


class SaltzmanGlaciationSystem(BaseSystem):
    """The glacial cycles of Saltzman and Maasch.

    The cycles are a self-sustained oscillation of ice, carbon, and ocean.

    The anomalies of global ice volume `x`, atmospheric carbon dioxide `y`, and deep ocean temperature `z` follow `dx/dt
    = -x - y - u z - nu F(t)`, `dy/dt = -p z + r y - s y^2 - y^3`, and `dz/dt = -q (x + z)`. The time unit is 10 kyr,
    and `F(t) = cos(2π t / 2.21) - cos(2π t / 4.41)` is a quasiperiodic orbital forcing with the periods of precession
    and obliquity. Without forcing the default parameters give a limit cycle of about 100 kyr. The system has the
    anomalies `climate.x`, `climate.y`, and `climate.z`.

    Args:
        x:
            The initial ice volume anomaly.

        y:
            The initial carbon dioxide anomaly.

        z:
            The initial deep ocean temperature anomaly.

        p:
            The effect of the deep ocean temperature on carbon dioxide.

        q:
            The relaxation rate of the deep ocean temperature to the ice volume.

        r:
            The growth rate of carbon dioxide by its own feedback.

        s:
            The quadratic saturation of carbon dioxide.

        u:
            The effect of the deep ocean temperature on ice volume.

        nu:
            The amplitude of the orbital forcing.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.climate import SaltzmanGlaciationSystem
        >>> SaltzmanGlaciationSystem(nu=0.5).prepare().system_.interactions[0].nu
        0.5
    """

    field = 'physics/climate'
    topics = ('ode', 'limit cycle', 'ice ages', 'paleoclimate')
    references = ('Saltzman and Maasch 1991', 'Datseris and Parlitz 2022')
    until = 50.0
    dt = 0.05

    def __init__(
        self,
        x: float = -0.2,
        y: float = 1.0,
        z: float = 0.0,
        *,
        p: float = 1.0,
        q: float = 2.5,
        r: float = 1.3,
        s: float = 0.6,
        u: float = 0.2,
        nu: float = 0.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.y = y
        self.z = z
        self.p = p
        self.q = q
        self.r = r
        self.s = s
        self.u = u
        self.nu = nu

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the climate and its feedbacks.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        feedbacks = SaltzmanMaaschInteraction(
            'feedbacks',
            target='climate',
            p=self.p,
            q=self.q,
            r=self.r,
            s=self.s,
            u=self.u,
            nu=self.nu,
        )
        self.system_ = System(
            self.label,
            elements=[SaltzmanMaaschElement('climate', x=self.x, y=self.y, z=self.z)],
            interactions=[feedbacks],
        )


class CarbonCycleInteraction(Interaction):
    """Rothman's balance of carbonate ion and dissolved inorganic carbon in the ocean.

    The carbonate ion concentration `c` and the dissolved inorganic carbon `w` of the target follow `dc/dt = f(c) (mu (1
    - b s(c, c_p) - theta (1 - s(c, c_x)) - nu) + w - w0)` and `dw/dt = mu (1 - b s(c, c_p) + theta (1 - s(c, c_x)) +
    nu) - w + w0`, with the sigmoids `s(c, c*) = c^gamma / (c^gamma + c*^gamma)` of burial and respiration and the
    buffer `f(c) = f0 c^beta / (c^beta + c_f^beta)`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the ocean.

        mu:
            The characteristic rate of the carbon fluxes.

        b:
            The strength of the burial of organic carbon.

        theta:
            The strength of the respiration of organic carbon.

        c_x:
            The crossover carbonate concentration of respiration, which sets the excitability.

        c_p:
            The carbonate concentration at which the burial is half its maximum.

        nu:
            The rate of external carbon injection.

        w0:
            The dissolved inorganic carbon at which the input balances the output.

        gamma:
            The steepness of the sigmoids of burial and respiration.

        f0:
            The maximum of the buffer factor.

        c_f:
            The carbonate concentration at which the buffer factor is half its maximum.

        beta:
            The steepness of the buffer factor.

    Examples:
        >>> from skcomplex.systems.physics.climate import CarbonCycleInteraction
        >>> CarbonCycleInteraction('carbon', target='ocean', nu=0.1).nu
        0.1
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        mu: float = 250.0,
        b: float = 4.0,
        theta: float = 5.0,
        c_x: float = 55.0,
        c_p: float = 110.0,
        nu: float = 0.0,
        w0: float = 2000.0,
        gamma: float = 4.0,
        f0: float = 0.694,
        c_f: float = 43.9,
        beta: float = 1.7,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.mu = mu
        self.b = b
        self.theta = theta
        self.c_x = c_x
        self.c_p = c_p
        self.nu = nu
        self.w0 = w0
        self.gamma = gamma
        self.f0 = f0
        self.c_f = c_f
        self.beta = beta

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rates of the carbonate ion and the dissolved inorganic carbon.

        Args:
            context:
                The namespace of the ocean, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            c, w, t = context.target.c, context.target.w, context.t
            burial = context.b * hill(c, context.c_p, context.gamma)
            respiration = context.theta * (1 - hill(c, context.c_x, context.gamma))
            buffer = context.f0 * hill(c, context.c_f, context.beta)
            self.laws_ = [
                RateLaw(
                    c.diff(t),
                    buffer * (context.mu * (1 - burial - respiration - context.nu) + w - context.w0),
                ),
                RateLaw(w.diff(t), context.mu * (1 - burial + respiration + context.nu) - w + context.w0),
            ]


class CarbonCycleSystem(BaseSystem):
    """Rothman's excitable carbon cycle.

    A large enough perturbation of the ocean sets off an excursion.

    The carbonate ion concentration `c` and the dissolved inorganic carbon `w`, in μmol kg⁻¹, follow `dc/dt = f(c) (mu
    (1 - b s(c, c_p) - theta (1 - s(c, c_x)) - nu) + w - w0)` and `dw/dt = mu (1 - b s(c, c_p) + theta (1 - s(c, c_x)) +
    nu) - w + w0`, with the sigmoids `s(c, c*) = c^gamma / (c^gamma + c*^gamma)` of burial and respiration and the
    buffer `f(c) = f0 c^beta / (c^beta + c_f^beta)`. The steady state near `c = 83.6` and `w = 2197` is stable, and from
    a dissolved carbon far enough below it the ocean returns only after a large excursion of `c`. The system has the
    concentrations `ocean.c` and `ocean.w`.

    Args:
        c:
            The initial carbonate ion concentration.

        w:
            The initial dissolved inorganic carbon.

        c_x:
            The crossover carbonate concentration of respiration, which sets the excitability.

        nu:
            The rate of external carbon injection.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.climate import CarbonCycleSystem
        >>> CarbonCycleSystem(nu=0.1).prepare().system_.interactions[0].nu
        0.1
    """

    field = 'physics/climate'
    topics = ('ode', 'excitability', 'carbon cycle', 'mass extinctions')
    references = ('Rothman 2019', 'Datseris and Parlitz 2022')
    until = 20.0
    dt = 0.05

    def __init__(
        self,
        c: float = 83.58,
        w: float = 2050.0,
        *,
        c_x: float = 55.0,
        nu: float = 0.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.c = c
        self.w = w
        self.c_x = c_x
        self.nu = nu

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the ocean and its carbon balance.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[Element('ocean', values={'c': self.c, 'w': self.w})],
            interactions=[CarbonCycleInteraction('carbon', target='ocean', c_x=self.c_x, nu=self.nu)],
        )


class VanDerPolIceAgesInteraction(Interaction):
    """Van der Pol relaxation of the ice volume and the climate, paced by the orbital forcing.

    The slow ice volume `x` and the fast climate state `y` of the target follow `tau x' = -(y + beta + gamma F(t))` and
    `tau y' = -alpha (y^3/3 - y - x)`, where `F(t) = cos(2 pi t/23) + cos(2 pi t/41)` stands for precession and
    obliquity, with time in thousands of years.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the climate.

        alpha:
            The time-scale separation of the fast climate state.

        beta:
            The asymmetry of the cycle, between build-up and termination.

        gamma:
            The strength of the orbital forcing.

        tau:
            The time scale, in thousands of years.

    Examples:
        >>> from skcomplex.systems.physics.climate import VanDerPolIceAgesInteraction
        >>> VanDerPolIceAgesInteraction('relaxation', target='climate', gamma=0.4).gamma
        0.4
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        alpha: float = 30.0,
        beta: float = 0.75,
        gamma: float = 0.0,
        tau: float = 35.09,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rates of the slow ice volume and the fast climate.

        Args:
            context:
                The namespace of the climate, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x, y, t = context.target.x, context.target.y, context.t
            forcing = cos(2 * pi * t / 23) + cos(2 * pi * t / 41)
            self.laws_ = [
                RateLaw(x.diff(t), -(y + context.beta + context.gamma * forcing) / context.tau),
                RateLaw(y.diff(t), -context.alpha * (y**3 / 3 - y - x) / context.tau),
            ]


class VanDerPolIceAgesSystem(BaseSystem):
    """Glacial cycles as a Van der Pol relaxation oscillator.

    The oscillator is optionally paced by orbital forcing.

    Time is in thousands of years. The slow variable `x` stands for ice volume and the fast variable `y` for the climate
    state, with `tau x' = -(y + beta + gamma F(t))` and `tau y' = -alpha (y^3/3 - y - x)`, where `F(t) = cos(2 pi t/23)
    + cos(2 pi t/41)` stands for precession and obliquity. For `|beta| < 1` the unforced oscillator has a relaxation
    limit cycle of slow ice build-up and fast terminations, and `gamma` sets how strongly the orbit paces it. The system
    has the states `climate.x` and `climate.y`.

    Args:
        x:
            The initial ice volume.

        y:
            The initial climate state.

        alpha:
            The time-scale separation of the fast climate state.

        beta:
            The asymmetry of the cycle, between build-up and termination.

        gamma:
            The strength of the orbital forcing.

        tau:
            The time scale, in thousands of years.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.climate import VanDerPolIceAgesSystem
        >>> VanDerPolIceAgesSystem(gamma=0.4).prepare().system_.interactions[0].gamma
        0.4
    """

    field = 'physics/climate'
    topics = ('ode', 'limit cycle', 'relaxation oscillator', 'ice ages', 'paleoclimate')
    references = ('De Saedeleer, Crucifix, and Wieczorek 2013', 'Datseris and Parlitz 2022')
    until = 400.0
    dt = 0.5

    def __init__(
        self,
        x: float = 0.0,
        y: float = 1.0,
        *,
        alpha: float = 30.0,
        beta: float = 0.75,
        gamma: float = 0.0,
        tau: float = 35.09,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.y = y
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the climate and its relaxation oscillation.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        relaxation = VanDerPolIceAgesInteraction(
            'relaxation',
            target='climate',
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            tau=self.tau,
        )
        self.system_ = System(
            self.label,
            elements=[Element('climate', values={'x': self.x, 'y': self.y})],
            interactions=[relaxation],
        )

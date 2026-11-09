"""Paleoclimate oscillators.

The module covers glacial cycles and the excitable carbon cycle.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, cos, oo, pi
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import hill
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import DeclaredVariableSpec, ParameterSpec, StateSpec, VisualSpec
from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations


class SaltzmanMaaschElement(Element):
    """Climate of the Saltzman-Maasch glacial cycles.

    The climate holds the anomalies of ice, carbon dioxide, and deep ocean temperature.

    The anomalies of global ice volume `x`, atmospheric carbon dioxide `y`, and deep ocean temperature `z` follow
    `dx/dt = -x - y - u z - nu F(t)`, `dy/dt = -p z + r y - s y^2 - y^3`, and `dz/dt = -q (x + z)`, where
    `F(t) = cos(2π t0 t / 22.1) - cos(2π t0 t / 44.1)` is the orbital forcing of precession and obliquity.

    Args:
        label:
            The label of the climate.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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

        t0:
            The time unit in thousands of years, which scales the orbital periods.

    Examples:
        >>> from skcomplex.systems.physics.climate import SaltzmanMaaschElement
        >>> SaltzmanMaaschElement('climate').q
        2.5
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = -0.2,
        y: Annotated[float, StateSpec()] = 1.0,
        z: Annotated[float, StateSpec()] = 0.0,
        p: Annotated[float, ParameterSpec()] = 1.0,
        q: Annotated[float, ParameterSpec()] = 2.5,
        r: Annotated[float, ParameterSpec()] = 1.3,
        s: Annotated[float, ParameterSpec()] = 0.6,
        u: Annotated[float, ParameterSpec()] = 0.2,
        nu: Annotated[float, ParameterSpec()] = 0.0,
        t0: Annotated[float, ParameterSpec()] = 10.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.z = z
        self.p = p
        self.q = q
        self.r = r
        self.s = s
        self.u = u
        self.nu = nu
        self.t0 = t0

    def _get_orbital_forcing(self, s: SimpleNamespace) -> Expression:
        """Return the quasiperiodic orbital forcing.

        The forcing has the periods of precession and obliquity.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            forcing:
                The orbital forcing.
        """
        return cos(2 * pi * s.t0 * s.t / 22.1) - cos(2 * pi * s.t0 * s.t / 44.1)

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the feedbacks of ice volume, carbon dioxide, and deep ocean temperature.

        The feedbacks include the orbital forcing.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate laws of the three anomalies.
        """
        forcing = self._get_orbital_forcing(s)
        return [
            RateLaw(s.x, -s.x - s.y - s.u * s.z - s.nu * forcing),
            RateLaw(s.y, -s.p * s.z + s.r * s.y - s.s * s.y**2 - s.y**3),
            RateLaw(s.z, -s.q * (s.x + s.z)),
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

    Examples:
        >>> from skcomplex.systems.physics.climate import SaltzmanGlaciationSystem
        >>> SaltzmanGlaciationSystem(nu=0.5).prepare().system_.elements[0].nu
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
    ) -> None:
        super().__init__(label)
        self.x = x
        self.y = y
        self.z = z
        self.p = p
        self.q = q
        self.r = r
        self.s = s
        self.u = u
        self.nu = nu

    def _build_system(self) -> CompositeSystem:
        """Return the glacial cycles of Saltzman and Maasch.

        The cycles are a self-sustained oscillation of ice, carbon, and ocean.
        """
        return CompositeSystem(
            self.label,
            elements=[
                SaltzmanMaaschElement(
                    'climate',
                    x=self.x,
                    y=self.y,
                    z=self.z,
                    p=self.p,
                    q=self.q,
                    r=self.r,
                    s=self.s,
                    u=self.u,
                    nu=self.nu,
                ),
            ],
        )


class CarbonCycleSystem(DynamicalSystem):
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

    Examples:
        >>> from skcomplex.systems.physics.climate import CarbonCycleSystem
        >>> CarbonCycleSystem(nu=0.1).prepare().system_.elements[0].get_value('nu')
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
    ) -> None:
        super().__init__(label=label)
        self.c = c
        self.w = w
        self.c_x = c_x
        self.nu = nu

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'ocean'

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the states and the parameters with their specs."""
        return {
            'c': DeclaredVariableSpec(self.c, StateSpec(domain=RealSet.open(0, oo))),
            'w': DeclaredVariableSpec(self.w, StateSpec(domain=RealSet.open(0, oo))),
            'mu': DeclaredVariableSpec(250.0, ParameterSpec()),
            'b': DeclaredVariableSpec(4.0, ParameterSpec()),
            'theta': DeclaredVariableSpec(5.0, ParameterSpec()),
            'c_x': DeclaredVariableSpec(self.c_x, ParameterSpec()),
            'c_p': DeclaredVariableSpec(110.0, ParameterSpec()),
            'nu': DeclaredVariableSpec(self.nu, ParameterSpec()),
            'w0': DeclaredVariableSpec(2000.0, ParameterSpec()),
            'gamma': DeclaredVariableSpec(4.0, ParameterSpec()),
            'f0': DeclaredVariableSpec(0.694, ParameterSpec()),
            'c_f': DeclaredVariableSpec(43.9, ParameterSpec()),
            'beta': DeclaredVariableSpec(1.7, ParameterSpec()),
        }

    def _get_equations(self) -> Equations:
        """Return Rothman's balance of carbonate ion and dissolved inorganic carbon."""
        return {
            'c': lambda s: (
                self._buffer(s)
                * (s.target.mu * (1 - self._burial(s) - self._respiration(s) - s.target.nu) + s.target.w - s.target.w0)
            ),
            'w': lambda s: (
                s.target.mu * (1 - self._burial(s) + self._respiration(s) + s.target.nu) - s.target.w + s.target.w0
            ),
        }

    def _burial(self, s: SimpleNamespace) -> Expression:
        """Return `burial` from the namespace of an equation."""
        return self._get_burial(s)

    def _respiration(self, s: SimpleNamespace) -> Expression:
        """Return `respiration` from the namespace of an equation."""
        return self._get_respiration(s)

    def _buffer(self, s: SimpleNamespace) -> Expression:
        """Return `buffer` from the namespace of an equation."""
        return self._get_buffer(s)

    def _get_burial(self, s: SimpleNamespace) -> Expression:
        """Return the burial of organic carbon.

        The burial is a Hill function of the carbonate ion concentration.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            burial:
                The burial, relative to the characteristic rate.
        """
        return s.target.b * hill(s.target.c, s.target.c_p, s.target.gamma)

    def _get_respiration(self, s: SimpleNamespace) -> Expression:
        """Return the respiration of organic carbon.

        The respiration is a decreasing Hill function of the carbonate ion concentration.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            respiration:
                The respiration, relative to the characteristic rate.
        """
        return s.target.theta * (1 - hill(s.target.c, s.target.c_x, s.target.gamma))

    def _get_buffer(self, s: SimpleNamespace) -> Expression:
        """Return the buffer factor of the carbonate ion concentration.

        The factor is a Hill function of the concentration.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            buffer:
                The buffer factor.
        """
        return s.target.f0 * hill(s.target.c, s.target.c_f, s.target.beta)


class VanDerPolIceAgesSystem(DynamicalSystem):
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

    Examples:
        >>> from skcomplex.systems.physics.climate import VanDerPolIceAgesSystem
        >>> VanDerPolIceAgesSystem(gamma=0.4).prepare().system_.elements[0].get_value('gamma')
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
    ) -> None:
        super().__init__(label=label)
        self.x = x
        self.y = y
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'climate'

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state."""
        return {
            'x': self.x,
            'y': self.y,
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the parameters of the equations."""
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'tau': self.tau,
        }

    def _get_equations(self) -> Equations:
        """Return the slow ice volume and the fast climate of the relaxation oscillator."""
        return {
            'x': lambda s: -(s.target.y + s.target.beta + s.target.gamma * self._forcing(s)) / s.target.tau,
            'y': lambda s: -s.target.alpha * (s.target.y**3 / 3 - s.target.y - s.target.x) / s.target.tau,
        }

    def _forcing(self, s: SimpleNamespace) -> Expression:
        """Return `forcing` from the namespace of an equation."""
        return self._get_orbital_forcing(s)

    def _get_orbital_forcing(self, s: SimpleNamespace) -> Expression:
        """Return the orbital forcing, the sum of the cycles of precession and obliquity.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            forcing:
                The orbital forcing.
        """
        return cos(2 * pi * s.t / 23) + cos(2 * pi * s.t / 41)

"""Paleoclimate oscillators: glacial cycles and the excitable carbon cycle."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, cos, oo, pi
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import hill
from ....core.system import System
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model


class SaltzmanMaaschElement(Element):
    """Climate of the Saltzman-Maasch glacial cycles, the anomalies of ice, carbon dioxide, and deep ocean temperature.

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
        >>> from skcomplex.models.earth.climate import SaltzmanMaaschElement
        >>> SaltzmanMaaschElement('climate').q
        2.5
    """

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
        """Return the quasiperiodic orbital forcing, with the periods of precession and obliquity.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            forcing:
                The orbital forcing.
        """
        return cos(2 * pi * s.t0 * s.t / 22.1) - cos(2 * pi * s.t0 * s.t / 44.1)

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the feedbacks of ice volume, carbon dioxide, and deep ocean temperature, with orbital forcing.

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


class VanDerPolIceAgesElement(Element):
    """Climate of glacial cycles as a Van der Pol relaxation oscillator, a slow ice volume and a fast climate state.

    With time in thousands of years, `tau x' = -(y + beta + gamma F(t))` and `tau y' = -alpha (y^3/3 - y - x)`, where
    `F(t) = cos(2 pi t/23) + cos(2 pi t/41)` stands for precession and obliquity.

    Args:
        label:
            The label of the climate.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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

    Examples:
        >>> from skcomplex.models.earth.climate import VanDerPolIceAgesElement
        >>> VanDerPolIceAgesElement('climate').alpha
        30.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.0,
        y: Annotated[float, StateSpec()] = 1.0,
        alpha: Annotated[float, ParameterSpec()] = 30.0,
        beta: Annotated[float, ParameterSpec()] = 0.75,
        gamma: Annotated[float, ParameterSpec()] = 0.0,
        tau: Annotated[float, ParameterSpec()] = 35.09,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

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

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the slow ice volume and the fast climate of the relaxation oscillator under orbital forcing.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate laws of the ice volume and the climate state.
        """
        forcing = self._get_orbital_forcing(s)
        return [
            RateLaw(s.x, -(s.y + s.beta + s.gamma * forcing) / s.tau),
            RateLaw(s.y, -s.alpha * (s.y**3 / 3 - s.y - s.x) / s.tau),
        ]


class CarbonCycleElement(Element):
    """Ocean of Rothman's excitable carbon cycle, its carbonate ion concentration and dissolved inorganic carbon.

    The carbonate ion concentration `c` and the dissolved inorganic carbon `w`, in μmol kg⁻¹, follow
    `dc/dt = f(c) (mu (1 - b s(c, c_p) - theta (1 - s(c, c_x)) - nu) + w - w0)` and
    `dw/dt = mu (1 - b s(c, c_p) + theta (1 - s(c, c_x)) + nu) - w + w0`, with the Hill functions
    `s(c, c*) = c^gamma / (c^gamma + c*^gamma)` of burial and respiration and the buffer
    `f(c) = f0 c^beta / (c^beta + c_f^beta)`.

    Args:
        label:
            The label of the ocean.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        c:
            The initial carbonate ion concentration.

        w:
            The initial dissolved inorganic carbon.

        mu:
            The characteristic rate of the carbon fluxes.

        b:
            The maximal burial of organic carbon.

        theta:
            The maximal respiration of organic carbon.

        c_x:
            The crossover carbonate concentration of respiration, which sets the excitability.

        c_p:
            The crossover carbonate concentration of burial.

        nu:
            The rate of external carbon injection.

        w0:
            The input of dissolved inorganic carbon by weathering and volcanism.

        gamma:
            The Hill coefficient of burial and respiration.

        f0:
            The maximal buffer of the carbonate ion concentration.

        c_f:
            The carbonate concentration where the buffer is half its maximum.

        beta:
            The Hill coefficient of the buffer.

    Examples:
        >>> from skcomplex.models.earth.climate import CarbonCycleElement
        >>> CarbonCycleElement('ocean').c_x
        55.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        c: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 83.58,
        w: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 2050.0,
        mu: Annotated[float, ParameterSpec()] = 250.0,
        b: Annotated[float, ParameterSpec()] = 4.0,
        theta: Annotated[float, ParameterSpec()] = 5.0,
        c_x: Annotated[float, ParameterSpec()] = 55.0,
        c_p: Annotated[float, ParameterSpec()] = 110.0,
        nu: Annotated[float, ParameterSpec()] = 0.0,
        w0: Annotated[float, ParameterSpec()] = 2000.0,
        gamma: Annotated[float, ParameterSpec()] = 4.0,
        f0: Annotated[float, ParameterSpec()] = 0.694,
        c_f: Annotated[float, ParameterSpec()] = 43.9,
        beta: Annotated[float, ParameterSpec()] = 1.7,
    ) -> None:
        super().__init__(label, visual=visual)
        self.c = c
        self.w = w
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

    def _get_burial(self, s: SimpleNamespace) -> Expression:
        """Return the burial of organic carbon, a Hill function of the carbonate ion concentration.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            burial:
                The burial, relative to the characteristic rate.
        """
        return s.b * hill(s.c, s.c_p, s.gamma)

    def _get_respiration(self, s: SimpleNamespace) -> Expression:
        """Return the respiration of organic carbon, a decreasing Hill function of the carbonate ion concentration.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            respiration:
                The respiration, relative to the characteristic rate.
        """
        return s.theta * (1 - hill(s.c, s.c_x, s.gamma))

    def _get_buffer(self, s: SimpleNamespace) -> Expression:
        """Return the buffer factor of the carbonate ion concentration, a Hill function of it.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            buffer:
                The buffer factor.
        """
        return s.f0 * hill(s.c, s.c_f, s.beta)

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return Rothman's balance of carbonate ion and dissolved inorganic carbon.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate laws of the carbonate ion concentration and the dissolved inorganic carbon.
        """
        burial = self._get_burial(s)
        respiration = self._get_respiration(s)
        buffer = self._get_buffer(s)
        return [
            RateLaw(s.c, buffer * (s.mu * (1 - burial - respiration - s.nu) + s.w - s.w0)),
            RateLaw(s.w, s.mu * (1 - burial + respiration + s.nu) - s.w + s.w0),
        ]


@register_model(
    field='earth/climate',
    tags=('ode', 'limit cycle', 'ice ages', 'paleoclimate'),
    references=('Saltzman and Maasch 1991', 'Datseris and Parlitz 2022'),
    defaults={'until': 50.0, 'dt': 0.05},
)
def saltzman_glaciation(
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
) -> System:
    """Return the glacial cycles of Saltzman and Maasch, a self-sustained oscillation of ice, carbon, and ocean.

    The anomalies of global ice volume `x`, atmospheric carbon dioxide `y`, and deep ocean temperature `z` follow
    `dx/dt = -x - y - u z - nu F(t)`, `dy/dt = -p z + r y - s y^2 - y^3`, and `dz/dt = -q (x + z)`. The time unit is
    10 kyr, and `F(t) = cos(2π t / 2.21) - cos(2π t / 4.41)` is a quasiperiodic orbital forcing with the periods of
    precession and obliquity. Without forcing the default parameters give a limit cycle of about 100 kyr.

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

    Returns:
        system:
            The model, with the anomalies `climate.x`, `climate.y`, and `climate.z`.

    Examples:
        >>> from skcomplex.models.earth.climate import saltzman_glaciation
        >>> saltzman_glaciation(nu=0.5).elements[0].nu
        0.5
    """
    return System(elements=[SaltzmanMaaschElement('climate', x=x, y=y, z=z, p=p, q=q, r=r, s=s, u=u, nu=nu)])


@register_model(
    field='earth/climate',
    tags=('ode', 'excitability', 'carbon cycle', 'mass extinctions'),
    references=('Rothman 2019', 'Datseris and Parlitz 2022'),
    defaults={'until': 20.0, 'dt': 0.05},
)
def carbon_cycle(
    c: float = 83.58,
    w: float = 2050.0,
    *,
    c_x: float = 55.0,
    nu: float = 0.0,
) -> System:
    """Return Rothman's excitable carbon cycle, where a large enough perturbation of the ocean sets off an excursion.

    The carbonate ion concentration `c` and the dissolved inorganic carbon `w`, in μmol kg⁻¹, follow
    `dc/dt = f(c) (mu (1 - b s(c, c_p) - theta (1 - s(c, c_x)) - nu) + w - w0)` and
    `dw/dt = mu (1 - b s(c, c_p) + theta (1 - s(c, c_x)) + nu) - w + w0`, with the sigmoids
    `s(c, c*) = c^gamma / (c^gamma + c*^gamma)` of burial and respiration and the buffer
    `f(c) = f0 c^beta / (c^beta + c_f^beta)`. The steady state near `c = 83.6` and `w = 2197` is stable, and from a
    dissolved carbon far enough below it the ocean returns only after a large excursion of `c`.

    Args:
        c:
            The initial carbonate ion concentration.

        w:
            The initial dissolved inorganic carbon.

        c_x:
            The crossover carbonate concentration of respiration, which sets the excitability.

        nu:
            The rate of external carbon injection.

    Returns:
        system:
            The model, with the concentrations `ocean.c` and `ocean.w`.

    Examples:
        >>> from skcomplex.models.earth.climate import carbon_cycle
        >>> carbon_cycle(nu=0.1).elements[0].nu
        0.1
    """
    return System(elements=[CarbonCycleElement('ocean', c=c, w=w, c_x=c_x, nu=nu)])


@register_model(
    field='earth/climate',
    tags=('ode', 'limit cycle', 'relaxation oscillator', 'ice ages', 'paleoclimate'),
    references=('De Saedeleer, Crucifix, and Wieczorek 2013', 'Datseris and Parlitz 2022'),
    defaults={'until': 400.0, 'dt': 0.5},
)
def van_der_pol_ice_ages(
    x: float = 0.0,
    y: float = 1.0,
    *,
    alpha: float = 30.0,
    beta: float = 0.75,
    gamma: float = 0.0,
    tau: float = 35.09,
) -> System:
    """Return glacial cycles as a Van der Pol relaxation oscillator, optionally paced by orbital forcing.

    Time is in thousands of years. The slow variable `x` stands for ice volume and the fast variable `y` for the climate
    state, with `tau x' = -(y + beta + gamma F(t))` and `tau y' = -alpha (y^3/3 - y - x)`, where
    `F(t) = cos(2 pi t/23) + cos(2 pi t/41)` stands for precession and obliquity. For `|beta| < 1` the unforced
    oscillator has a relaxation limit cycle of slow ice build-up and fast terminations, and `gamma` sets how strongly
    the orbit paces it.

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

    Returns:
        system:
            The model, with the states `climate.x` and `climate.y`.

    Examples:
        >>> from skcomplex.models.earth.climate import van_der_pol_ice_ages
        >>> van_der_pol_ice_ages(gamma=0.4).elements[0].gamma
        0.4
    """
    return System(
        elements=[VanDerPolIceAgesElement('climate', x=x, y=y, alpha=alpha, beta=beta, gamma=gamma, tau=tau)],
    )

"""Fluid dynamics systems.

The module covers incompressible flow, convection, turbulence, shocks, and shallow water.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Mapping, Sequence
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, FieldElement
from ....core.components.interactions import AdvectionInteraction, ConservationInteraction, Interaction, RateInteraction
from ....core.components.laws.utils import gradient, inverse_helmholtz, laplacian
from ....core.components.spaces import LatticeSpace
from ....core.components.systems import BaseSystem, System
from ....core.engine.solvers import ETDRK4Solver, SSPRKSolver
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ....core.types import Array
from ..common.fields import DiffusivityInteraction


class BoussinesqElement(Element):
    """Vorticity and temperature of Boussinesq convection.

    The units are those of the thermal diffusion time.

    The temperature `theta` is the departure from the linear profile between a hot bottom and a cold top, so
    `dtheta/dt + u . grad(theta) = w + laplacian(theta)` and
    `domega/dt + u . grad(omega) = Pr laplacian(omega) + Ra Pr dtheta/dx`. The advections, the diffusions, and the
    buoyancy are interactions of the system.

    Args:
        label:
            The label of the site or the field.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        omega:
            The initial vorticity.

        theta:
            The initial temperature departure.

        Ra:
            The Rayleigh number.

        Pr:
            The Prandtl number.

    Examples:
        >>> from skcomplex.systems.physics.fluids import BoussinesqElement
        >>> BoussinesqElement('site').Ra
        1000.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        omega: Annotated[float, StateSpec()] = 0.0,
        theta: Annotated[float, StateSpec()] = 0.0,
        Ra: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1000.0,
        Pr: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.omega = omega
        self.theta = theta
        self.Ra = Ra
        self.Pr = Pr


class BurgersElement(Element):
    """Site of a field `u` carried by the Burgers equation.

    The equation is `du/dt + d(u^2 / 2)/dx = 0`.

    Args:
        label:
            The label of the site.

        visual:
            How the site is drawn, or `None` for none.

        u:
            The initial value.

    Examples:
        >>> from skcomplex.systems.physics.fluids import BurgersElement
        >>> BurgersElement('site', u=1.5).u
        1.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u


class KuramotoSivashinskyElement(Element):
    """Site of a flame front or a thin film.

    The field follows `du/dt = -d(u^2 / 2)/dx - laplacian(u) - laplacian(laplacian(u))`.

    The conservation of the flux `u^2 / 2` and the rate of the linear operator are interactions of the system.

    Args:
        label:
            The label of the site.

        visual:
            How the site is drawn, or `None` for none.

        u:
            The initial height of the front.

    Examples:
        >>> from skcomplex.systems.physics.fluids import KuramotoSivashinskyElement
        >>> KuramotoSivashinskyElement('site', u=0.5).u
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u


class ShallowWaterElement(Element):
    """Site of a channel of shallow water.

    The site has the depth `h` and the discharge `q` per unit width.

    The mass `h` and the momentum `q` are conserved by two conservation interactions, with the fluxes `q` and
    `q^2 / h + g h^2 / 2`, which form one hyperbolic system.

    Args:
        label:
            The label of the site.

        visual:
            How the site is drawn, or `None` for none.

        h:
            The initial depth.

        q:
            The initial discharge.

        g:
            The gravitational acceleration.

    Examples:
        >>> from skcomplex.systems.physics.fluids import ShallowWaterElement
        >>> ShallowWaterElement('site', h=2.0).h
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        h: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 1.0,
        q: Annotated[float, StateSpec()] = 0.0,
        g: Annotated[float, ParameterSpec()] = 9.81,
    ) -> None:
        super().__init__(label, visual=visual)
        self.h = h
        self.q = q
        self.g = g


class VorticityElement(Element):
    """Vorticity of an incompressible two-dimensional flow.

    The vorticity follows `domega/dt + u . grad(omega) = nu laplacian(omega) + f`.

    The velocity comes from the streamfunction, the solution of a Poisson equation, so the flow is divergence free,
    and the advection is written in conservative form so that dealiasing applies to it. The advection, the viscosity,
    and the forcing are interactions of the system.

    Args:
        label:
            The label of the site or the field.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        omega:
            The initial vorticity.

        nu:
            The kinematic viscosity.

        f:
            The forcing of the vorticity.

    Examples:
        >>> from skcomplex.systems.physics.fluids import VorticityElement
        >>> VorticityElement('site').nu
        0.001
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        omega: Annotated[float, StateSpec()] = 0.0,
        nu: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.001,
        f: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.omega = omega
        self.nu = nu
        self.f = f


class KuramotoSivashinskySystem(BaseSystem):
    """The Kuramoto-Sivashinsky equation `u_t = -u u_x - u_xx - u_xxxx`.

    The equation is spatiotemporally chaotic.

    The field is spectral on a periodic line, started from `cos(x / 16) (1 + sin(x / 16))` as in Kassam and Trefethen,
    and the stiff fourth derivative suits exponential time differencing.

    The derivative of the nonlinear term `u²/2` is dealiased by the two-thirds rule: it keeps only the wavenumbers below
    a third of `2 π modes / length`, 2.67 for the defaults, so the spectrum drops to round-off from `q = 2.69` on. The
    cut costs little, since the modes there decay at the rate `q⁴ - q²` above 45, and with the defaults the power at the
    last kept mode is five decades below the peak near `q = 0.6`. More modes on the same length move the cut up in
    proportion. The system has the field `line.u`.

    Args:
        modes:
            The number of sites, and of Fourier modes.

        length:
            The length of the line.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.fluids import KuramotoSivashinskySystem
        >>> KuramotoSivashinskySystem(32).prepare().system_.elements[0].discretization
        'spectral'
    """

    field = 'physics/fluids'
    topics = ('field', 'chaos', 'spectral', 'stiff', 'pde')
    references = ('Kuramoto 1978', 'Sivashinsky 1977', 'Kassam and Trefethen 2005')
    until = 150.0
    dt = 1.0
    solver = ETDRK4Solver(dt=0.25)

    def __init__(
        self,
        modes: int = 128,
        *,
        length: float = 32 * math.pi,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.modes = modes
        self.length = length

    def _build_system(self) -> System:
        """Return the Kuramoto-Sivashinsky equation `u_t = -u u_x - u_xx - u_xxxx`.

        The equation is spatiotemporally chaotic.
        """
        spacing = self.length / self.modes
        positions = np.arange(1, self.modes + 1) * spacing
        values = {'u': np.cos(positions / 16) * (1 + np.sin(positions / 16))}
        space = LatticeSpace(self.modes, spacing=spacing)
        transport = ConservationInteraction(
            'transport',
            target='line',
            state='u',
            flux=lambda s: s.target.u**2 / 2,
            riemann=None,
        )
        linear = RateInteraction(
            'linear',
            target='line',
            state='u',
            rate=lambda s: -laplacian(s.target.u) - laplacian(laplacian(s.target.u)),
        )
        return System(
            self.label,
            elements=[
                FieldElement(
                    'line',
                    element=KuramotoSivashinskyElement,
                    space=space,
                    values=values,
                    discretization='spectral',
                ),
            ],
            interactions=[transport, linear],
        )


def _velocity(omega: Expression) -> tuple[Expression, Expression]:
    """Return the velocity of a vorticity.

    The velocity comes from its streamfunction `psi` with `laplacian(psi) = -omega`.
    """
    slopes = gradient(inverse_helmholtz(-omega), 2)
    return slopes[1], -slopes[0]


def _get_flow_interactions(buoyant: bool) -> list[Interaction]:
    """Return the advection and the viscosity of the vorticity.

    The laws include its forcing or the buoyancy and the heat.
    """
    advection = AdvectionInteraction(
        'advection',
        target='flow',
        state='omega',
        drift=lambda s: _velocity(s.target.omega),
        riemann=None,
    )
    if not buoyant:
        viscosity = DiffusivityInteraction('viscosity', target='flow', state='omega', coefficient=lambda s: s.target.nu)
        forcing = RateInteraction('forcing', target='flow', state='omega', rate=lambda s: s.target.f)
        return [advection, viscosity, forcing]
    return [
        advection,
        DiffusivityInteraction('viscosity', target='flow', state='omega', coefficient=lambda s: s.target.Pr),
        RateInteraction(
            'buoyancy',
            target='flow',
            state='omega',
            rate=lambda s: s.target.Ra * s.target.Pr * gradient(s.target.theta, 2)[0],
        ),
        AdvectionInteraction(
            'convection',
            target='flow',
            state='theta',
            drift=lambda s: _velocity(s.target.omega),
            source=lambda s: _velocity(s.target.omega)[1],
            riemann=None,
        ),
        DiffusivityInteraction('conduction', target='flow', state='theta', coefficient=1.0),
    ]


def incompressible_flow(
    shape: Sequence[int] = (64, 64),
    size: Sequence[float] = (2 * math.pi, 2 * math.pi),
    *,
    nu: float = 1e-3,
    forcing: npt.ArrayLike | None = None,
    buoyancy: Mapping[str, float] | None = None,
    walls: bool = False,
    vorticity: npt.ArrayLike | None = None,
    temperature: npt.ArrayLike | None = None,
    label: str = 'system',
) -> System:
    """Return incompressible two-dimensional flow in vorticity-streamfunction form.

    The flow is on a periodic spectral lattice.

    Args:
        shape:
            The number of sites along both axes.

        size:
            The lengths of the domain along both axes.

        nu:
            The kinematic viscosity, unused with buoyancy, whose viscosity is the Prandtl number.

        forcing:
            The forcing of the vorticity at every site, or `None`.

        buoyancy:
            The Rayleigh number `Ra` and the Prandtl number `Pr` of Boussinesq convection with a temperature field,
            or `None` for a flow without temperature.

        walls:
            Whether the second axis is twice the height of a layer between free-slip walls in its middle and at its
            ends, the vorticity and temperature being odd about them.

        vorticity:
            The initial vorticity, or `None` for rest.

        temperature:
            The initial temperature departure of convection, or `None` for zero.

        label:
            The label of the system.

    Returns:
        system:
            The flow, with the vorticity `flow.omega` and, with buoyancy, the temperature `flow.theta`.

    Examples:
        >>> from skcomplex.systems.physics.fluids import incompressible_flow
        >>> incompressible_flow((16, 16), buoyancy={'Ra': 800.0, 'Pr': 1.0}).elements[0].element.__name__
        'BoussinesqElement'
    """
    shape = tuple(int(item) for item in shape)
    space = LatticeSpace(shape, spacing=tuple(length / count for length, count in zip(size, shape, strict=True)))
    values: dict[str, object] = {}
    if vorticity is not None:
        values['omega'] = np.asarray(vorticity, dtype=float)
    if buoyancy is None:
        element: type[Element] = VorticityElement
        values['nu'] = nu
        if forcing is not None:
            values['f'] = np.asarray(forcing, dtype=float)
    else:
        element = BoussinesqElement
        values.update({key: float(value) for key, value in buoyancy.items()})
        if temperature is not None:
            values['theta'] = np.asarray(temperature, dtype=float)
    states = ['omega'] if buoyancy is None else ['omega', 'theta']
    symmetry = dict.fromkeys(states, (None, 'odd')) if walls else None
    field = FieldElement(
        'flow',
        element=element,
        space=space,
        values=values,
        discretization='spectral',
        symmetry=symmetry,
    )
    return System(label, elements=[field], interactions=_get_flow_interactions(buoyancy is not None))


class DecayingTurbulenceSystem(BaseSystem):
    """Decaying two-dimensional turbulence from a random vorticity field.

    The field lies on a periodic square of side 2 pi.

    Energy flows to large scales and enstrophy to small scales, where viscosity removes it, so vortices merge while the
    enstrophy decays faster than the energy. The system has the vorticity `flow.omega`.

    Args:
        shape:
            The number of sites along both axes.

        nu:
            The kinematic viscosity.

        modes:
            The largest wavenumber of the initial vorticity.

        random_state:
            The seed or generator of the initial vorticity.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.fluids import DecayingTurbulenceSystem
        >>> DecayingTurbulenceSystem((16, 16)).prepare().system_.elements[0].prepare().size_
        256
    """

    field = 'physics/fluids'
    topics = ('field', 'turbulence', 'spectral', 'pde')
    references = ('Kraichnan 1967', 'Satz 2022')
    until = 20.0
    dt = 1.0
    solver = ETDRK4Solver(dt=0.05)

    def __init__(
        self,
        shape: Sequence[int] = (64, 64),
        *,
        nu: float = 1e-3,
        modes: int = 6,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.nu = nu
        self.modes = modes
        self.random_state = random_state

    @staticmethod
    def _smooth_noise(shape: tuple[int, ...], modes: int, random_state: int | np.random.Generator | None) -> Array:
        """Return random values with only the Fourier modes below a wavenumber.

        The values are scaled to unit largest value.
        """
        rng = np.random.default_rng(random_state)
        spectrum = np.fft.rfftn(rng.standard_normal(shape))
        frequencies = np.meshgrid(
            *[np.abs(np.fft.fftfreq(size, d=1.0 / size)) for size in shape[:-1]],
            np.fft.rfftfreq(shape[-1], d=1.0 / shape[-1]),
            indexing='ij',
        )
        spectrum[np.sqrt(sum(item**2 for item in frequencies)) > modes] = 0
        values = np.fft.irfftn(spectrum, s=shape, axes=range(len(shape)))
        return values / np.abs(values).max()

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        shape = tuple(self.shape)
        vorticity = 5 * self._smooth_noise(shape, self.modes, self.random_state)
        return incompressible_flow(shape, nu=self.nu, vorticity=vorticity, label=self.label)


class RayleighBenardSystem(BaseSystem):
    """Rayleigh-Bénard convection between free-slip walls.

    The convection starts from a small perturbation of the critical mode.

    The layer has height 1 and width `2 sqrt(2)`, the wavelength of the critical mode, and convection sets in above `Ra
    = 27 pi^4 / 4`, about 657.5. The lattice holds twice the layer's height with odd states, which gives the free-slip
    walls. The registered step `ETDRK4Solver(dt=0.005)` is stable up to about `Ra = 4000`, and faster convection needs a
    smaller step, such as `ETDRK4Solver(dt=0.0025)` at `Ra = 5000`. The system has the vorticity `flow.omega` and the
    temperature departure `flow.theta`.

    Args:
        Ra:
            The Rayleigh number.

        Pr:
            The Prandtl number.

        shape:
            The number of sites along the width and twice the height.

        amplitude:
            The amplitude of the initial temperature perturbation.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.fluids import RayleighBenardSystem
        >>> RayleighBenardSystem(500.0).prepare().system_.elements[0].values['Ra']
        500.0
    """

    field = 'physics/fluids'
    topics = ('field', 'convection', 'instability', 'spectral', 'pde')
    references = ('Rayleigh 1916', 'Chandrasekhar 1961', 'Mihailović et al. 2023')
    until = 10.0
    dt = 0.5
    solver = ETDRK4Solver(dt=0.005)

    def __init__(
        self,
        Ra: float = 1000.0,
        *,
        Pr: float = 1.0,
        shape: Sequence[int] = (32, 32),
        amplitude: float = 1e-3,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.Ra = Ra
        self.Pr = Pr
        self.shape = shape
        self.amplitude = amplitude

    def _build_system(self) -> System:
        """Return Rayleigh-Bénard convection between free-slip walls.

        The convection starts from a small perturbation of the critical mode.
        """
        width, height = 2 * math.sqrt(2), 2.0
        x = (np.arange(self.shape[0]) * width / self.shape[0])[:, None]
        z = (np.arange(self.shape[1]) * height / self.shape[1])[None, :]
        temperature = self.amplitude * np.cos(math.pi / math.sqrt(2) * x) * np.sin(math.pi * z)
        return incompressible_flow(
            self.shape,
            (width, height),
            buoyancy={'Ra': self.Ra, 'Pr': self.Pr},
            walls=True,
            temperature=temperature,
            label=self.label,
        )


class BurgersSystem(BaseSystem):
    """The inviscid Burgers equation on a periodic interval of length 2 pi.

    Its sine steepens into a shock.

    The cells hold averages updated by MUSCL reconstruction with a minmod limiter and Rusanov fluxes, so the total is
    conserved and the shock forms no new extrema. The shock forms at time `1 / amplitude`. The system has the velocity
    `line.u`.

    Args:
        n:
            The number of cells.

        mean:
            The mean of the initial wave.

        amplitude:
            The amplitude of the initial wave.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.fluids import BurgersSystem
        >>> BurgersSystem(16).prepare().system_.elements[0].prepare().size_
        16
    """

    field = 'physics/fluids'
    topics = ('field', 'shocks', 'conservation law', 'finite volume', 'pde')
    references = ('Burgers 1948', 'LeVeque 2002')
    until = 2.0
    dt = 0.1
    solver = SSPRKSolver()

    def __init__(
        self,
        n: int = 200,
        *,
        mean: float = 1.5,
        amplitude: float = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n = n
        self.mean = mean
        self.amplitude = amplitude

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        spacing = 2 * math.pi / self.n
        centers = (np.arange(self.n) + 0.5) * spacing
        values = {'u': self.mean + self.amplitude * np.sin(centers)}
        space = LatticeSpace(self.n, spacing=spacing)
        return System(
            self.label,
            elements=[
                FieldElement(
                    'line',
                    element=BurgersElement,
                    space=space,
                    values=values,
                    discretization='finite-volume',
                ),
            ],
            interactions=[
                ConservationInteraction('transport', target='line', state='u', flux=lambda s: s.target.u**2 / 2),
            ],
        )


class ShallowWaterSystem(BaseSystem):
    """The shallow-water dam break.

    Still water at two depths was separated at the middle by a removed dam.

    A shock runs downstream and a rarefaction upstream, with the depth and the discharge conserved by finite volumes and
    open edges. The system has the depth `channel.h` and the discharge `channel.q`.

    Args:
        n:
            The number of cells.

        length:
            The length of the channel.

        upstream:
            The depth upstream of the dam.

        downstream:
            The depth downstream of the dam.

        g:
            The gravitational acceleration.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.fluids import ShallowWaterSystem
        >>> float(ShallowWaterSystem(10).prepare().system_.elements[0].values['h'][0])
        2.0
    """

    field = 'physics/fluids'
    topics = ('field', 'waves', 'shocks', 'conservation law', 'finite volume', 'pde')
    references = ('Stoker 1957', 'LeVeque 2002')
    until = 0.5
    dt = 0.05
    solver = SSPRKSolver()

    def __init__(
        self,
        n: int = 400,
        *,
        length: float = 10.0,
        upstream: float = 2.0,
        downstream: float = 1.0,
        g: float = 9.81,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n = n
        self.length = length
        self.upstream = upstream
        self.downstream = downstream
        self.g = g

    def _build_system(self) -> System:
        """Return the shallow-water dam break.

        Still water at two depths was separated at the middle by a removed dam.
        """
        depth = np.where(np.arange(self.n) < self.n // 2, self.upstream, self.downstream).astype(float)
        space = LatticeSpace(self.n, spacing=self.length / self.n)
        values = {'h': depth, 'q': np.zeros(self.n), 'g': self.g}
        boundary = {'h': 'zero-flux', 'q': 'zero-flux'}
        field = FieldElement(
            'channel',
            element=ShallowWaterElement,
            space=space,
            boundary=boundary,
            values=values,
            discretization='finite-volume',
        )
        mass = ConservationInteraction('mass', target='channel', state='h', flux=lambda s: s.target.q)
        momentum = ConservationInteraction(
            'momentum',
            target='channel',
            state='q',
            flux=lambda s: s.target.q**2 / s.target.h + s.target.g * s.target.h**2 / 2,
        )
        return System(self.label, elements=[field], interactions=[mass, momentum])

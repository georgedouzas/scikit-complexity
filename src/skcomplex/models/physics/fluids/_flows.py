"""Fluid dynamics models: incompressible flow, convection, turbulence, shocks, and shallow water."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from ....core.base import Domain, parameter, state
from ....core.components import Element, Field
from ....core.laws import Flux, Law, Rate, Symbols, divergence, gradient, inverse_laplacian, laplacian
from ....core.runtime import register_model
from ....core.solvers import ETDRK4, SSPRK
from ....core.space import LatticeSpace
from ....core.system import System

type Array = npt.NDArray[np.float64]

KuramotoSivashinsky = Element.define(
    'KuramotoSivashinsky',
    states={'u': 0.0},
    laws=lambda s: [Rate(s.u, -divergence([s.u**2 / 2]) - laplacian(s.u) - laplacian(laplacian(s.u)))],
)


@register_model(
    field='physics/fluids',
    tags=('field', 'chaos', 'spectral', 'stiff', 'pde'),
    references=('Kuramoto 1978', 'Sivashinsky 1977', 'Kassam and Trefethen 2005'),
    defaults={'until': 150.0, 'dt': 1.0, 'solver': ETDRK4(dt=0.25)},
)
def kuramoto_sivashinsky(modes: int = 128, *, length: float = 32 * math.pi) -> System:
    """Return the Kuramoto-Sivashinsky equation `u_t = -u u_x - u_xx - u_xxxx`, spatiotemporally chaotic.

    The field is spectral on a periodic line, started from `cos(x / 16) (1 + sin(x / 16))` as in Kassam and Trefethen,
    and the stiff fourth derivative suits exponential time differencing.

    The derivative of the nonlinear term `u²/2` is dealiased by the two-thirds rule: it keeps only the wavenumbers
    below a third of `2 π modes / length`, 2.67 for the defaults, so the spectrum drops to round-off from `q = 2.69`
    on. The cut costs little, since the modes there decay at the rate `q⁴ - q²` above 45, and with the defaults the
    power at the last kept mode is five decades below the peak near `q = 0.6`. More modes on the same length move the
    cut up in proportion.

    Args:
        modes:
            The number of sites, and of Fourier modes.

        length:
            The length of the line.

    Returns:
        system:
            The model, with the field `line.u`.

    Examples:
        >>> from skcomplex.models.physics.fluids import kuramoto_sivashinsky
        >>> kuramoto_sivashinsky(32).elements[0].discretization
        'spectral'
    """
    spacing = length / modes
    positions = np.arange(1, modes + 1) * spacing
    values = {'u': np.cos(positions / 16) * (1 + np.sin(positions / 16))}
    space = LatticeSpace(modes, spacing=spacing)
    return System(components=[Field('line', KuramotoSivashinsky, space, values=values, discretization='spectral')])


def _velocity(omega: Any) -> tuple[Any, Any]:  # noqa: ANN401
    """Return the velocity of a vorticity, from its streamfunction `psi` with `laplacian(psi) = -omega`."""
    slopes = gradient(inverse_laplacian(-omega), 2)
    return slopes[1], -slopes[0]


class Vorticity(Element):
    """Vorticity of an incompressible two-dimensional flow, `domega/dt + u . grad(omega) = nu laplacian(omega) + f`.

    The velocity comes from the streamfunction, the solution of a Poisson equation, so the flow is divergence free,
    and the advection is written in conservative form so that dealiasing applies to it.

    Args:
        label:
            The label of the site or the field.

        omega:
            The initial vorticity.

        nu:
            The kinematic viscosity.

        f:
            The forcing of the vorticity.

    Examples:
        >>> from skcomplex.models.physics.fluids import Vorticity
        >>> Vorticity('site').nu
        0.001
    """

    omega: float = state(default=0.0)
    nu: float = parameter(default=1e-3, domain=Domain.NONNEGATIVE)
    f: float = parameter(default=0.0)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the vorticity equation.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        u, w = _velocity(s.omega)
        return [Rate(s.omega, -divergence([u * s.omega, w * s.omega]) + s.nu * laplacian(s.omega) + s.f)]


class Boussinesq(Element):
    """Vorticity and temperature of Boussinesq convection, in units of the thermal diffusion time.

    The temperature `theta` is the departure from the linear profile between a hot bottom and a cold top, so
    `dtheta/dt + u . grad(theta) = w + laplacian(theta)` and
    `domega/dt + u . grad(omega) = Pr laplacian(omega) + Ra Pr dtheta/dx`.

    Args:
        label:
            The label of the site or the field.

        omega:
            The initial vorticity.

        theta:
            The initial temperature departure.

        Ra:
            The Rayleigh number.

        Pr:
            The Prandtl number.

    Examples:
        >>> from skcomplex.models.physics.fluids import Boussinesq
        >>> Boussinesq('site').Ra
        1000.0
    """

    omega: float = state(default=0.0)
    theta: float = state(default=0.0)
    Ra: float = parameter(default=1000.0, domain=Domain.NONNEGATIVE)
    Pr: float = parameter(default=1.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the vorticity and temperature equations.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        u, w = _velocity(s.omega)
        buoyancy = gradient(s.theta, 2)[0]
        return [
            Rate(
                s.omega,
                -divergence([u * s.omega, w * s.omega]) + s.Pr * laplacian(s.omega) + s.Ra * s.Pr * buoyancy,
            ),
            Rate(s.theta, -divergence([u * s.theta, w * s.theta]) + w + laplacian(s.theta)),
        ]


Burgers = Element.define(
    'Burgers',
    states={'u': 0.0},
    laws=lambda s: [Flux(s.u, s.u**2 / 2)],
)

ShallowWater = Element.define(
    'ShallowWater',
    states={'h': state(default=1.0, domain=Domain.POSITIVE), 'q': 0.0},
    parameters={'g': 9.81},
    laws=lambda s: [Flux(s.h, s.q), Flux(s.q, s.q**2 / s.h + s.g * s.h**2 / 2)],
)


def _smooth_noise(shape: tuple[int, ...], modes: int, random_state: int | np.random.Generator | None) -> Array:
    """Return random values with only the Fourier modes below a wavenumber, scaled to unit largest value."""
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
) -> System:
    """Return incompressible two-dimensional flow in vorticity-streamfunction form on a periodic spectral lattice.

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

    Returns:
        system:
            The flow, with the vorticity `flow.omega` and, with buoyancy, the temperature `flow.theta`.

    Examples:
        >>> from skcomplex.models.physics.fluids import incompressible_flow
        >>> incompressible_flow((16, 16), buoyancy={'Ra': 800.0, 'Pr': 1.0}).elements[0].element.__name__
        'Boussinesq'
    """
    shape = tuple(int(item) for item in shape)
    space = LatticeSpace(shape, spacing=tuple(length / count for length, count in zip(size, shape, strict=True)))
    values: dict[str, Any] = {}
    if vorticity is not None:
        values['omega'] = np.asarray(vorticity, dtype=float)
    if buoyancy is None:
        element: type[Element] = Vorticity
        values['nu'] = nu
        if forcing is not None:
            values['f'] = np.asarray(forcing, dtype=float)
    else:
        element = Boussinesq
        values.update({key: float(value) for key, value in buoyancy.items()})
        if temperature is not None:
            values['theta'] = np.asarray(temperature, dtype=float)
    states = ['omega'] if buoyancy is None else ['omega', 'theta']
    symmetry = dict.fromkeys(states, (None, 'odd')) if walls else None
    field = Field('flow', element, space, values=values, discretization='spectral', symmetry=symmetry)
    return System(components=[field])


@register_model(
    field='physics/fluids',
    tags=('field', 'turbulence', 'spectral', 'pde'),
    references=('Kraichnan 1967', 'Satz 2022'),
    defaults={'until': 20.0, 'dt': 1.0, 'solver': ETDRK4(dt=0.05)},
)
def decaying_turbulence(
    shape: Sequence[int] = (64, 64),
    *,
    nu: float = 1e-3,
    modes: int = 6,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return decaying two-dimensional turbulence from a random vorticity field on a periodic square of side 2 pi.

    Energy flows to large scales and enstrophy to small scales, where viscosity removes it, so vortices merge while
    the enstrophy decays faster than the energy.

    Args:
        shape:
            The number of sites along both axes.

        nu:
            The kinematic viscosity.

        modes:
            The largest wavenumber of the initial vorticity.

        random_state:
            The seed or generator of the initial vorticity.

    Returns:
        system:
            The model, with the vorticity `flow.omega`.

    Examples:
        >>> from skcomplex.models.physics.fluids import decaying_turbulence
        >>> decaying_turbulence((16, 16)).elements[0].size
        256
    """
    shape = tuple(shape)
    vorticity = 5 * _smooth_noise(shape, modes, random_state)
    return incompressible_flow(shape, nu=nu, vorticity=vorticity)


@register_model(
    field='physics/fluids',
    tags=('field', 'convection', 'instability', 'spectral', 'pde'),
    references=('Rayleigh 1916', 'Chandrasekhar 1961', 'Mihailović et al. 2023'),
    defaults={'until': 10.0, 'dt': 0.5, 'solver': ETDRK4(dt=0.005)},
)
def rayleigh_benard(
    Ra: float = 1000.0,
    *,
    Pr: float = 1.0,
    shape: Sequence[int] = (32, 32),
    amplitude: float = 1e-3,
) -> System:
    """Return Rayleigh-Bénard convection between free-slip walls, from a small perturbation of the critical mode.

    The layer has height 1 and width `2 sqrt(2)`, the wavelength of the critical mode, and convection sets in above
    `Ra = 27 pi^4 / 4`, about 657.5. The lattice holds twice the layer's height with odd states, which gives the
    free-slip walls. The registered step `ETDRK4(dt=0.005)` is stable up to about `Ra = 4000`, and faster convection
    needs a smaller step, such as `ETDRK4(dt=0.0025)` at `Ra = 5000`.

    Args:
        Ra:
            The Rayleigh number.

        Pr:
            The Prandtl number.

        shape:
            The number of sites along the width and twice the height.

        amplitude:
            The amplitude of the initial temperature perturbation.

    Returns:
        system:
            The model, with the vorticity `flow.omega` and the temperature departure `flow.theta`.

    Examples:
        >>> from skcomplex.models.physics.fluids import rayleigh_benard
        >>> rayleigh_benard(500.0).elements[0].values['Ra']
        500.0
    """
    width, height = 2 * math.sqrt(2), 2.0
    x = (np.arange(shape[0]) * width / shape[0])[:, None]
    z = (np.arange(shape[1]) * height / shape[1])[None, :]
    temperature = amplitude * np.cos(math.pi / math.sqrt(2) * x) * np.sin(math.pi * z)
    return incompressible_flow(
        shape,
        (width, height),
        buoyancy={'Ra': Ra, 'Pr': Pr},
        walls=True,
        temperature=temperature,
    )


@register_model(
    field='physics/fluids',
    tags=('field', 'shocks', 'conservation law', 'finite volume', 'pde'),
    references=('Burgers 1948', 'LeVeque 2002'),
    defaults={'until': 2.0, 'dt': 0.1, 'solver': SSPRK()},
)
def burgers(n: int = 200, *, mean: float = 1.5, amplitude: float = 1.0) -> System:
    """Return the inviscid Burgers equation on a periodic interval of length 2 pi, whose sine steepens into a shock.

    The cells hold averages updated by MUSCL reconstruction with a minmod limiter and Rusanov fluxes, so the total is
    conserved and the shock forms no new extrema. The shock forms at time `1 / amplitude`.

    Args:
        n:
            The number of cells.

        mean:
            The mean of the initial wave.

        amplitude:
            The amplitude of the initial wave.

    Returns:
        system:
            The model, with the velocity `line.u`.

    Examples:
        >>> from skcomplex.models.physics.fluids import burgers
        >>> burgers(16).elements[0].size
        16
    """
    spacing = 2 * math.pi / n
    centers = (np.arange(n) + 0.5) * spacing
    values = {'u': mean + amplitude * np.sin(centers)}
    space = LatticeSpace(n, spacing=spacing)
    return System(components=[Field('line', Burgers, space, values=values, discretization='finite-volume')])


@register_model(
    field='physics/fluids',
    tags=('field', 'waves', 'shocks', 'conservation law', 'finite volume', 'pde'),
    references=('Stoker 1957', 'LeVeque 2002'),
    defaults={'until': 0.5, 'dt': 0.05, 'solver': SSPRK()},
)
def shallow_water(
    n: int = 400,
    *,
    length: float = 10.0,
    upstream: float = 2.0,
    downstream: float = 1.0,
    g: float = 9.81,
) -> System:
    """Return the shallow-water dam break: still water at two depths that a removed dam separated at the middle.

    A shock runs downstream and a rarefaction upstream, with the depth and the discharge conserved by finite volumes
    and open edges.

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

    Returns:
        system:
            The model, with the depth `channel.h` and the discharge `channel.q`.

    Examples:
        >>> from skcomplex.models.physics.fluids import shallow_water
        >>> float(shallow_water(10).elements[0].values['h'][0])
        2.0
    """
    depth = np.where(np.arange(n) < n // 2, upstream, downstream).astype(float)
    space = LatticeSpace(n, spacing=length / n)
    values = {'h': depth, 'q': np.zeros(n), 'g': g}
    boundary = {'h': 'zero-flux', 'q': 'zero-flux'}
    field = Field('channel', ShallowWater, space, boundary=boundary, values=values, discretization='finite-volume')
    return System(components=[field])

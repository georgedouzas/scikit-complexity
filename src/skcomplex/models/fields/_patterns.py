"""Model diffusion and reaction-diffusion fields: heat, fronts, Gray-Scott, Brusselator, excitable media, and Turing."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from ...core import Domain, parameter, state
from ...model import Element, Field, LatticeSpace, Mesh, Rate, System, divergence, laplacian
from ...runtime import register_model
from ...solvers import ETDRK4, RK4, IMEXEuler

type Array = npt.NDArray[np.float64]

Heat = Element.define(
    'Heat',
    states={'u': 0.0},
    parameters={'D': parameter(default=1.0, domain=Domain.NONNEGATIVE)},
    laws=lambda s: [Rate(s.u, s.D * laplacian(s.u))],
)

FisherKPP = Element.define(
    'FisherKPP',
    states={'u': state(default=0.0, domain=Domain.NONNEGATIVE)},
    parameters={'D': parameter(default=1.0, domain=Domain.NONNEGATIVE), 'r': 1.0},
    laws=lambda s: [Rate(s.u, s.D * laplacian(s.u) + s.r * s.u * (1 - s.u))],
)

GrayScott = Element.define(
    'GrayScott',
    states={'u': 1.0, 'v': 0.0},
    parameters={'Du': 0.16, 'Dv': 0.08, 'F': 0.035, 'k': 0.065},
    laws=lambda s: [
        Rate(s.u, s.Du * laplacian(s.u) - s.u * s.v**2 + s.F * (1 - s.u)),
        Rate(s.v, s.Dv * laplacian(s.v) + s.u * s.v**2 - (s.F + s.k) * s.v),
    ],
)

Brusselator = Element.define(
    'Brusselator',
    states={'u': 1.0, 'v': 3.0},
    parameters={'a': 1.0, 'b': 3.0, 'Du': 1.0, 'Dv': 8.0},
    laws=lambda s: [
        Rate(s.u, s.Du * laplacian(s.u) + s.a - (s.b + 1) * s.u + s.u**2 * s.v),
        Rate(s.v, s.Dv * laplacian(s.v) + s.b * s.u - s.u**2 * s.v),
    ],
)

FitzHughNagumo = Element.define(
    'FitzHughNagumo',
    states={'u': 0.0, 'v': 0.0},
    parameters={'D': 1.0, 'a': 0.7, 'b': 0.8, 'epsilon': 0.02},
    laws=lambda s: [
        Rate(s.u, s.D * laplacian(s.u) + s.u - s.u**3 / 3 - s.v),
        Rate(s.v, s.epsilon * (s.u + s.a - s.b * s.v)),
    ],
)

Schnakenberg = Element.define(
    'Schnakenberg',
    states={'u': 1.0, 'v': 0.9},
    parameters={'a': 0.1, 'b': 0.9, 'Du': 1.0, 'Dv': 40.0},
    laws=lambda s: [
        Rate(s.u, s.Du * laplacian(s.u) + s.a - s.u + s.u**2 * s.v),
        Rate(s.v, s.Dv * laplacian(s.v) + s.b - s.u**2 * s.v),
    ],
)


def _noise(shape: tuple[int, ...], scale: float, random_state: int | np.random.Generator | None) -> Array:
    """Return small uniform noise of a shape."""
    return scale * np.random.default_rng(random_state).uniform(-1.0, 1.0, size=shape)


@register_model(
    field='pattern formation',
    tags=('field', 'diffusion', 'pde'),
    references=('Fourier 1822', 'Datseris and Parlitz 2022'),
    defaults={'until': 50.0, 'dt': 1.0},
)
def heat_diffusion(
    shape: Sequence[int] = (64, 64),
    *,
    D: float = 1.0,
    spacing: float = 1.0,
    boundary: str = 'zero-flux',
) -> System:
    """Return heat diffusing on a plate from a hot square in its middle, `du/dt = D laplacian(u)`.

    Args:
        shape:
            The number of sites along every dimension.

        D:
            The diffusion coefficient.

        spacing:
            The distance between sites.

        boundary:
            `'periodic'`, `'zero-flux'` for an insulated plate, or `'fixed'` for edges held at zero.

    Returns:
        system:
            The model, with the temperature `plate.u`.

    Examples:
        >>> from skcomplex.models.fields import heat_diffusion
        >>> heat_diffusion((10, 10)).elements[0].size
        100
    """
    shape = tuple(shape)
    u = np.zeros(shape)
    middle = tuple(slice(size // 2 - max(size // 8, 1), size // 2 + max(size // 8, 1)) for size in shape)
    u[middle] = 1.0
    condition = ('fixed', 0.0) if boundary == 'fixed' else boundary
    space = LatticeSpace(shape, spacing=spacing)
    return System(elements=[Field('plate', Heat, space, boundary={'u': condition}, values={'u': u, 'D': D})])


@register_model(
    field='pattern formation',
    tags=('field', 'fronts', 'reaction-diffusion', 'pde'),
    references=('Fisher 1937', 'Kolmogorov, Petrovsky, and Piskunov 1937'),
    defaults={'until': 40.0, 'dt': 1.0},
)
def fisher_kpp(
    length: int = 800,
    *,
    D: float = 1.0,
    r: float = 1.0,
    spacing: float = 0.25,
    front: float = 0.05,
) -> System:
    """Return the Fisher-KPP equation, whose step invades the empty line as a front of speed `2 sqrt(r D)`.

    Args:
        length:
            The number of sites of the line.

        D:
            The diffusion coefficient.

        r:
            The growth rate.

        spacing:
            The distance between sites.

        front:
            The fraction of the line that starts occupied, on its left.

    Returns:
        system:
            The model, with the density `line.u`.

    Examples:
        >>> from skcomplex.models.fields import fisher_kpp
        >>> int(fisher_kpp(100, front=0.1).elements[0].values['u'].sum())
        10
    """
    u = (np.arange(length) < front * length).astype(float)
    space = LatticeSpace(length, spacing=spacing)
    return System(
        elements=[Field('line', FisherKPP, space, boundary={'u': 'zero-flux'}, values={'u': u, 'D': D, 'r': r})],
    )


@register_model(
    field='pattern formation',
    tags=('field', 'reaction-diffusion', 'self-replication', 'pde'),
    references=('Gray and Scott 1984', 'Pearson 1993'),
    defaults={'until': 2000.0, 'dt': 20.0, 'solver': RK4(dt=1.0)},
)
def gray_scott(
    shape: Sequence[int] = (128, 128),
    *,
    Du: float = 0.16,
    Dv: float = 0.08,
    F: float = 0.035,
    k: float = 0.065,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Gray-Scott reaction-diffusion model, seeded in its middle, which grows spots that replicate.

    Args:
        shape:
            The number of rows and columns of the periodic lattice.

        Du:
            The diffusion of the substrate `u`.

        Dv:
            The diffusion of the activator `v`.

        F:
            The feed rate.

        k:
            The kill rate.

        random_state:
            The seed or generator of the noise on the seed.

    Returns:
        system:
            The model, with the substrate `cells.u` and the activator `cells.v`.

    Examples:
        >>> from skcomplex.models.fields import gray_scott
        >>> gray_scott((32, 32), F=0.04).elements[0].values['F']
        0.04
    """
    shape = tuple(shape)
    u, v = np.ones(shape), np.zeros(shape)
    middle = tuple(slice(size // 2 - size // 10, size // 2 + size // 10) for size in shape)
    u[middle], v[middle] = 0.5, 0.25
    noise = _noise(shape, 0.01, random_state)
    values = {'u': u + noise, 'v': v + np.abs(noise), 'Du': Du, 'Dv': Dv, 'F': F, 'k': k}
    return System(elements=[Field('cells', GrayScott, LatticeSpace(shape), values=values)])


@register_model(
    field='pattern formation',
    tags=('field', 'reaction-diffusion', 'Turing', 'oscillation', 'pde'),
    references=('Prigogine and Lefever 1968', 'Gros 2015'),
    defaults={'until': 100.0, 'dt': 1.0, 'solver': RK4(dt=0.01)},
)
def brusselator(
    shape: Sequence[int] = (64, 64),
    *,
    a: float = 1.0,
    b: float = 3.0,
    Du: float = 1.0,
    Dv: float = 8.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Brusselator reaction-diffusion model from small noise around its steady state `(a, b / a)`.

    Args:
        shape:
            The number of rows and columns of the periodic lattice.

        a:
            The feed of `u`.

        b:
            The conversion of `u` into `v`.

        Du:
            The diffusion of `u`.

        Dv:
            The diffusion of `v`.

        random_state:
            The seed or generator of the noise.

    Returns:
        system:
            The model, with the concentrations `cells.u` and `cells.v`.

    Examples:
        >>> from skcomplex.models.fields import brusselator
        >>> brusselator((8, 8), b=2.5).elements[0].values['b']
        2.5
    """
    shape = tuple(shape)
    values = {
        'u': a + _noise(shape, 0.01, random_state),
        'v': np.full(shape, b / a),
        'a': a,
        'b': b,
        'Du': Du,
        'Dv': Dv,
    }
    return System(elements=[Field('cells', Brusselator, LatticeSpace(shape), values=values)])


@register_model(
    field='pattern formation',
    tags=('field', 'excitable media', 'spirals', 'pde'),
    references=('FitzHugh 1961', 'Nagumo, Arimoto, and Yoshizawa 1962', 'Gros 2015'),
    defaults={'until': 200.0, 'dt': 2.0, 'solver': RK4(dt=0.05)},
)
def fitzhugh_nagumo(
    shape: Sequence[int] = (100, 100),
    *,
    D: float = 1.0,
    a: float = 0.7,
    b: float = 0.8,
    epsilon: float = 0.02,
) -> System:
    """Return the FitzHugh-Nagumo excitable medium, started from a broken wave that curls into a spiral.

    The upper half starts excited and the left half refractory, so the free end of the wave rotates.

    Args:
        shape:
            The number of rows and columns of the lattice with zero-flux edges.

        D:
            The diffusion of the potential `u`.

        a:
            The offset of the recovery.

        b:
            The damping of the recovery.

        epsilon:
            The time scale of the recovery.

    Returns:
        system:
            The model, with the potential `medium.u` and the recovery `medium.v`.

    Examples:
        >>> from skcomplex.models.fields import fitzhugh_nagumo
        >>> fitzhugh_nagumo((20, 20)).elements[0].values['epsilon']
        0.02
    """
    rows, columns = shape
    u, v = np.full(shape, -1.2), np.full(shape, -0.62)
    u[: rows // 2, :] = 1.5
    v[:, : columns // 2] = 1.0
    values = {'u': u, 'v': v, 'D': D, 'a': a, 'b': b, 'epsilon': epsilon}
    space = LatticeSpace((rows, columns), boundary='reflecting')
    return System(elements=[Field('medium', FitzHughNagumo, space, values=values)])


@register_model(
    field='pattern formation',
    tags=('field', 'reaction-diffusion', 'Turing', 'pde'),
    references=('Schnakenberg 1979', 'Turing 1952', 'Murray 2003'),
    defaults={'until': 200.0, 'dt': 5.0, 'solver': RK4(dt=0.005)},
)
def schnakenberg(
    shape: Sequence[int] = (64, 64),
    *,
    a: float = 0.1,
    b: float = 0.9,
    Du: float = 1.0,
    Dv: float = 40.0,
    spacing: float = 1.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Schnakenberg reaction-diffusion model, which forms Turing patterns from noise when `Dv >> Du`.

    The steady state `u = a + b`, `v = b / (a + b)^2` is stable without diffusion and unstable to perturbations of a
    band of wavelengths with it.

    Args:
        shape:
            The number of sites along every dimension of the periodic lattice.

        a:
            The production of `u`.

        b:
            The production of `v`.

        Du:
            The diffusion of the activator `u`.

        Dv:
            The diffusion of the inhibitor `v`.

        spacing:
            The distance between sites.

        random_state:
            The seed or generator of the noise.

    Returns:
        system:
            The model, with the concentrations `cells.u` and `cells.v`.

    Examples:
        >>> from skcomplex.models.fields import schnakenberg
        >>> schnakenberg((8, 8)).elements[0].values['Dv']
        40.0
    """
    shape = tuple(shape)
    steady_u, steady_v = a + b, b / (a + b) ** 2
    values = {
        'u': steady_u * (1 + _noise(shape, 0.01, random_state)),
        'v': np.full(shape, steady_v),
        'a': a,
        'b': b,
        'Du': Du,
        'Dv': Dv,
    }
    return System(elements=[Field('cells', Schnakenberg, LatticeSpace(shape, spacing=spacing), values=values)])


KuramotoSivashinsky = Element.define(
    'KuramotoSivashinsky',
    states={'u': 0.0},
    laws=lambda s: [Rate(s.u, -divergence([s.u**2 / 2]) - laplacian(s.u) - laplacian(laplacian(s.u)))],
)

GinzburgLandau = Element.define(
    'GinzburgLandau',
    states={'u': 0.0, 'v': 0.0},
    parameters={'b': 1.0, 'c': -1.5},
    laws=lambda s: [
        Rate(s.u, s.u + laplacian(s.u) - s.b * laplacian(s.v) - (s.u**2 + s.v**2) * (s.u - s.c * s.v)),
        Rate(s.v, s.v + s.b * laplacian(s.u) + laplacian(s.v) - (s.u**2 + s.v**2) * (s.c * s.u + s.v)),
    ],
)

SwiftHohenberg = Element.define(
    'SwiftHohenberg',
    states={'u': 0.0},
    parameters={'r': 0.3},
    laws=lambda s: [Rate(s.u, (s.r - 1) * s.u - 2 * laplacian(s.u) - laplacian(laplacian(s.u)) - s.u**3)],
)


def harmonic(x: Array, y: Array) -> Array:
    """Return the harmonic function `exp(x) sin(y)`, a steady temperature known in closed form.

    Args:
        x:
            The first coordinates.

        y:
            The second coordinates.

    Returns:
        values:
            The temperatures.

    Examples:
        >>> from skcomplex.models.fields import harmonic
        >>> float(harmonic(0.0, 0.0))
        0.0
    """
    return np.exp(x) * np.sin(y)


@register_model(
    field='pattern formation',
    tags=('field', 'chaos', 'spectral', 'stiff', 'pde'),
    references=('Kuramoto 1978', 'Sivashinsky 1977', 'Kassam and Trefethen 2005'),
    defaults={'until': 150.0, 'dt': 1.0, 'solver': ETDRK4(dt=0.25)},
)
def kuramoto_sivashinsky(modes: int = 128, *, length: float = 32 * math.pi) -> System:
    """Return the Kuramoto-Sivashinsky equation `u_t = -u u_x - u_xx - u_xxxx`, spatiotemporally chaotic.

    The field is spectral on a periodic line, started from `cos(x / 16) (1 + sin(x / 16))` as in Kassam and Trefethen,
    and the stiff fourth derivative suits exponential time differencing.

    Args:
        modes:
            The number of sites, and of Fourier modes.

        length:
            The length of the line.

    Returns:
        system:
            The model, with the field `line.u`.

    Examples:
        >>> from skcomplex.models.fields import kuramoto_sivashinsky
        >>> kuramoto_sivashinsky(32).elements[0].discretization
        'spectral'
    """
    spacing = length / modes
    positions = np.arange(1, modes + 1) * spacing
    values = {'u': np.cos(positions / 16) * (1 + np.sin(positions / 16))}
    space = LatticeSpace(modes, spacing=spacing)
    return System(elements=[Field('line', KuramotoSivashinsky, space, values=values, discretization='spectral')])


@register_model(
    field='pattern formation',
    tags=('field', 'chaos', 'spirals', 'spectral', 'pde'),
    references=('Aranson and Kramer 2002', 'Gros 2015'),
    defaults={'until': 50.0, 'dt': 1.0, 'solver': ETDRK4(dt=0.1)},
)
def complex_ginzburg_landau(
    shape: Sequence[int] = (64, 64),
    *,
    length: float = 64.0,
    b: float = 1.0,
    c: float = -1.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the complex Ginzburg-Landau equation `A_t = A + (1 + i b) laplacian(A) - (1 + i c) |A|^2 A`.

    The amplitude `A = u + i v` is held as its real and imaginary parts, and the linear part couples them per Fourier
    mode. With `1 + b c < 0` the uniform oscillation is unstable and spiral defects turn into turbulence.

    Args:
        shape:
            The number of sites along both axes of the periodic lattice.

        length:
            The side of the square.

        b:
            The linear dispersion.

        c:
            The nonlinear dispersion.

        random_state:
            The seed or generator of the small random initial amplitude.

    Returns:
        system:
            The model, with the real part `medium.u` and the imaginary part `medium.v`.

    Examples:
        >>> from skcomplex.models.fields import complex_ginzburg_landau
        >>> complex_ginzburg_landau((8, 8)).elements[0].values['c']
        -1.5
    """
    shape = tuple(shape)
    values = {'u': _noise(shape, 0.01, random_state), 'v': _noise(shape, 0.01, None), 'b': b, 'c': c}
    space = LatticeSpace(shape, spacing=length / shape[0])
    return System(elements=[Field('medium', GinzburgLandau, space, values=values, discretization='spectral')])


@register_model(
    field='pattern formation',
    tags=('field', 'rolls', 'convection', 'spectral', 'pde'),
    references=('Swift and Hohenberg 1977', 'Cross and Hohenberg 1993', 'Datseris and Parlitz 2022'),
    defaults={'until': 200.0, 'dt': 5.0, 'solver': ETDRK4(dt=0.5)},
)
def swift_hohenberg(
    shape: Sequence[int] = (64, 64),
    *,
    length: float = 16 * math.pi,
    r: float = 0.3,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Swift-Hohenberg equation `u_t = r u - (1 + laplacian)^2 u - u^3`, which forms rolls of length 2 pi.

    Above onset, `r > 0`, small noise grows into rolls whose wavenumber is near 1, the fastest growing one.

    Args:
        shape:
            The number of sites along every axis of the periodic lattice.

        length:
            The side of the domain.

        r:
            The distance above onset.

        random_state:
            The seed or generator of the noise.

    Returns:
        system:
            The model, with the field `cells.u`.

    Examples:
        >>> from skcomplex.models.fields import swift_hohenberg
        >>> swift_hohenberg((8, 8), r=0.1).elements[0].values['r']
        0.1
    """
    shape = tuple(shape)
    values = {'u': _noise(shape, 0.01, random_state), 'r': r}
    space = LatticeSpace(shape, spacing=length / shape[0])
    return System(elements=[Field('cells', SwiftHohenberg, space, values=values, discretization='spectral')])


@register_model(
    field='pattern formation',
    tags=('field', 'diffusion', 'finite element', 'mesh', 'pde'),
    references=('Strang and Fix 1973', 'Gustafsson and McBain 2020'),
    defaults={'until': 2.0, 'dt': 0.1, 'solver': IMEXEuler(dt=0.05)},
)
def heat_l_shape(
    refine: int = 3,
    *,
    D: float = 1.0,
    temperature: Callable[..., Any] | float = harmonic,
) -> System:
    """Return heat on the L-shaped domain with fixed boundary temperatures, discretised by linear finite elements.

    The plate starts cold inside, and its temperature relaxes to the solution of Laplace's equation with the
    boundary temperatures. The default boundary temperature is harmonic, so the steady state is known.

    Args:
        refine:
            The number of uniform refinements of the mesh.

        D:
            The diffusion coefficient.

        temperature:
            The boundary temperature, a number or a function of the coordinates.

    Returns:
        system:
            The model, with the temperature `plate.u` at the nodes of the mesh.

    Examples:
        >>> from skcomplex.models.fields import heat_l_shape
        >>> heat_l_shape(1).elements[0].size
        21
    """
    mesh = Mesh.l_shape(refine)
    return System(
        elements=[
            Field(
                'plate',
                Heat,
                mesh,
                boundary={'u': ('fixed', temperature)},
                values={'D': D},
                discretization='finite-element',
            ),
        ],
    )

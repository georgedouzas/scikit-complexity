"""Dynamical systems systems: discrete maps."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

from sage.all__sagemath_symbolics import pi, sin

from ....core.components.interactions import MapInteraction
from ....core.components.laws.utils import mod, where
from ....core.components.system import System
from ..._catalog import register_system
from ._baker_map import BakerMapElement
from ._cat_map import CatMapElement
from ._circle_map import CircleMapElement
from ._henon_map import HenonMapElement
from ._intermittency_map import IntermittencyMapElement
from ._logistic_map import LogisticMapElement
from ._newton_map import NewtonMapElement
from ._standard_map import GOLDEN_MEAN, StandardMapElement
from ._tent_map import TentMapElement
from ._towel_map import TowelMapElement


@register_system(
    field='mathematics/dynamical_systems',
    tags=('chaos', 'discrete map', 'period doubling'),
    references=('May 1976', 'Satz 2022'),
    defaults={'until': 200, 'dt': 1},
)
def logistic_map(x: float = 0.2, *, r: float = 3.7) -> System:
    """Return the logistic map `x_{t+1} = r x_t (1 - x_t)`, which reaches chaos through period doubling as `r` grows.

    Args:
        x:
            The initial value, between 0 and 1.

        r:
            The growth rate, between 0 and 4.

    Returns:
        system:
            The system, with the state `map.x`.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import logistic_map
        >>> logistic_map(r=3.2).elements[0].r
        3.2
    """
    logistic = MapInteraction(
        'x_map',
        target='map',
        state='x',
        function=lambda s: s.target.r * s.target.x * (1 - s.target.x),
    )
    return System(elements=[LogisticMapElement('map', x=x, r=r)], interactions=[logistic])


@register_system(
    field='mathematics/dynamical_systems',
    tags=('chaos', 'discrete map', 'strange attractor'),
    references=('Hénon 1976', 'Datseris and Parlitz 2022'),
    defaults={'until': 1000, 'dt': 1},
)
def henon_map(x: float = 0.0, y: float = 0.0, *, a: float = 1.4, b: float = 0.3) -> System:
    """Return the Hénon map `x' = 1 - a x² + y`, `y' = b x`, with a fractal strange attractor for its defaults.

    Args:
        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        a:
            The nonlinearity.

        b:
            The contraction.

    Returns:
        system:
            The system, with the states `map.x` and `map.y`.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import henon_map
        >>> henon_map(a=1.2).elements[0].a
        1.2
    """
    first = MapInteraction(
        'x_map',
        target='map',
        state='x',
        function=lambda s: 1 - s.target.a * s.target.x**2 + s.target.y,
    )
    second = MapInteraction('y_map', target='map', state='y', function=lambda s: s.target.b * s.target.x)
    return System(elements=[HenonMapElement('map', x=x, y=y, a=a, b=b)], interactions=[first, second])


@register_system(
    field='mathematics/dynamical_systems',
    tags=('chaos', 'discrete map', 'hamiltonian', 'kam'),
    references=('Chirikov 1979', 'Datseris and Parlitz 2022', 'Dilão 2023', 'Lindgren 2024'),
    defaults={'until': 1000, 'dt': 1},
)
def standard_map(theta: float = 0.0, p: float = 2 * math.pi * GOLDEN_MEAN, *, K: float = 1.2) -> System:
    """Return the Chirikov standard map `p' = p + K sin θ`, `θ' = θ + p'`, with global chaos above K ≈ 0.9716.

    The last invariant circle that spans the angle, the golden-mean torus, breaks at K ≈ 0.971635.
    The angle is taken modulo 2π and the momentum is kept on the real line, so its spread measures diffusion.

    Args:
        theta:
            The initial angle.

        p:
            The initial momentum, by default on the golden-mean torus of the unkicked rotor.

        K:
            The kick strength.

    Returns:
        system:
            The system, with the states `map.theta` and `map.p`.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import standard_map
        >>> standard_map(K=0.5).elements[0].K
        0.5
    """
    kick = MapInteraction(
        'p_map',
        target='map',
        state='p',
        function=lambda s: s.target.p + s.target.K * sin(s.target.theta),
    )
    rotation = MapInteraction(
        'theta_map',
        target='map',
        state='theta',
        function=lambda s: mod(s.target.theta + s.target.p + s.target.K * sin(s.target.theta), 2 * pi),
    )
    return System(elements=[StandardMapElement('map', theta=theta, p=p, K=K)], interactions=[kick, rotation])


@register_system(
    field='mathematics/dynamical_systems',
    tags=('discrete map', 'mode locking', 'arnold tongues'),
    references=('Arnold 1965', 'Datseris and Parlitz 2022', 'Dilão 2023'),
    defaults={'until': 500, 'dt': 1},
)
def circle_map(theta: float = 0.0, *, Omega: float = 0.25, K: float = 0.9) -> System:
    """Return the sine circle map `θ' = θ + Ω - K / (2π) sin 2πθ (mod 1)`, which locks into Arnold tongues.

    Args:
        theta:
            The initial angle, as a fraction of a turn.

        Omega:
            The bare rotation per step.

        K:
            The nonlinearity, invertible below 1.

    Returns:
        system:
            The system, with the state `map.theta`.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import circle_map
        >>> circle_map(K=1.0).elements[0].K
        1.0
    """
    rotation = MapInteraction(
        'theta_map',
        target='map',
        state='theta',
        function=lambda s: mod(
            s.target.theta + s.target.Omega - s.target.K / (2 * pi) * sin(2 * pi * s.target.theta),
            1,
        ),
    )
    return System(elements=[CircleMapElement('map', theta=theta, Omega=Omega, K=K)], interactions=[rotation])


@register_system(
    field='mathematics/dynamical_systems',
    tags=('chaos', 'discrete map', 'piecewise linear'),
    references=('Datseris and Parlitz 2022', 'Dilão 2023'),
    defaults={'until': 200, 'dt': 1},
)
def tent_map(x: float = 0.3, *, mu: float = 1.99) -> System:
    """Return the tent map `x' = μ min(x, 1 - x)`, chaotic with Lyapunov exponent `ln μ` for μ above 1.

    Args:
        x:
            The initial value, between 0 and 1.

        mu:
            The slope, between 0 and 2.

    Returns:
        system:
            The system, with the state `map.x`.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import tent_map
        >>> tent_map(mu=1.5).elements[0].mu
        1.5
    """
    tent = MapInteraction(
        'x_map',
        target='map',
        state='x',
        function=lambda s: s.target.mu * where(s.target.x < 1 / 2, s.target.x, 1 - s.target.x),
    )
    return System(elements=[TentMapElement('map', x=x, mu=mu)], interactions=[tent])


@register_system(
    field='mathematics/dynamical_systems',
    tags=('chaos', 'discrete map', 'mixing', 'fractal'),
    references=('Farmer, Ott, and Yorke 1983', 'Datseris and Parlitz 2022', 'Lindgren 2024'),
    defaults={'until': 500, 'dt': 1},
)
def baker_map(
    x: float = 0.2,
    y: float = 0.3,
    *,
    alpha: float = 1 / 3,
    lambda_a: float = 1 / 3,
    lambda_b: float = 2 / 3,
) -> System:
    """Return the generalized baker map, which stretches the unit square along `x` and stacks the two pieces in `y`.

    The strip `x < alpha` is stretched by `1 / alpha` and squeezed into the bottom band of height `lambda_a`, and the
    rest is stretched by `1 / (1 - alpha)` and squeezed into the top band of height `lambda_b`. The map preserves area
    when `lambda_a = alpha` and `lambda_b = 1 - alpha`, and it has a fractal attractor when `lambda_a + lambda_b < 1`.

    Args:
        x:
            The initial stretched coordinate.

        y:
            The initial squeezed coordinate.

        alpha:
            The position of the cut.

        lambda_a:
            The height of the bottom band.

        lambda_b:
            The height of the top band.

    Returns:
        system:
            The system, with the states `map.x` and `map.y`.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import baker_map
        >>> baker_map(lambda_a=0.2).elements[0].lambda_a
        0.2
    """
    stretch = MapInteraction(
        'x_map',
        target='map',
        state='x',
        function=lambda s: where(
            s.target.x < s.target.alpha,
            s.target.x / s.target.alpha,
            (s.target.x - s.target.alpha) / (1 - s.target.alpha),
        ),
    )
    stack = MapInteraction(
        'y_map',
        target='map',
        state='y',
        function=lambda s: where(
            s.target.x < s.target.alpha,
            s.target.lambda_a * s.target.y,
            1 - s.target.lambda_b + s.target.lambda_b * s.target.y,
        ),
    )
    baker = BakerMapElement('map', x=x, y=y, alpha=alpha, lambda_a=lambda_a, lambda_b=lambda_b)
    return System(elements=[baker], interactions=[stretch, stack])


@register_system(
    field='mathematics/dynamical_systems',
    tags=('chaos', 'discrete map', 'hyperchaos'),
    references=('Rössler 1979', 'Datseris and Parlitz 2022'),
    defaults={'until': 1000, 'dt': 1},
)
def towel_map(
    x: float = 0.085,
    y: float = -0.121,
    z: float = 0.075,
    *,
    a: float = 3.8,
    b: float = 3.78,
) -> System:
    """Return Rössler's towel map, a three-dimensional map with two positive Lyapunov exponents.

    `x' = a x (1 - x) - 0.05 (y + 0.35)(1 - 2z)`, `y' = 0.1 ((y + 0.35)(1 - 2z) - 1)(1 - 1.9 x)`, and
    `z' = b z (1 - z) + 0.2 y`.

    Args:
        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        z:
            The initial third coordinate.

        a:
            The growth rate of the first logistic coordinate.

        b:
            The growth rate of the third logistic coordinate.

    Returns:
        system:
            The system, with the states `map.x`, `map.y`, and `map.z`.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import towel_map
        >>> towel_map(a=3.7).elements[0].a
        3.7
    """
    first = MapInteraction(
        'x_map',
        target='map',
        state='x',
        function=lambda s: (
            s.target.a * s.target.x * (1 - s.target.x) - 0.05 * (s.target.y + 0.35) * (1 - 2 * s.target.z)
        ),
    )
    second = MapInteraction(
        'y_map',
        target='map',
        state='y',
        function=lambda s: 0.1 * ((s.target.y + 0.35) * (1 - 2 * s.target.z) - 1) * (1 - 1.9 * s.target.x),
    )
    third = MapInteraction(
        'z_map',
        target='map',
        state='z',
        function=lambda s: s.target.b * s.target.z * (1 - s.target.z) + 0.2 * s.target.y,
    )
    return System(elements=[TowelMapElement('map', x=x, y=y, z=z, a=a, b=b)], interactions=[first, second, third])


@register_system(
    field='mathematics/dynamical_systems',
    tags=('chaos', 'discrete map', 'intermittency'),
    references=('Pomeau and Manneville 1980', 'Dilão 2023', 'Jensen 2023'),
    defaults={'until': 1000, 'dt': 1},
)
def intermittency_map(x: float = 0.01, *, z: float = 2.0) -> System:
    """Return the Pomeau-Manneville map `x' = x + x^z (mod 1)`, whose long laminar phases near 0 end in chaotic bursts.

    Args:
        x:
            The initial value, between 0 and 1.

        z:
            The exponent of the tangency at 0, above 1.

    Returns:
        system:
            The system, with the state `map.x`.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import intermittency_map
        >>> intermittency_map(z=1.5).elements[0].z
        1.5
    """
    burst = MapInteraction(
        'x_map',
        target='map',
        state='x',
        function=lambda s: mod(s.target.x + s.target.x**s.target.z, 1),
    )
    return System(elements=[IntermittencyMapElement('map', x=x, z=z)], interactions=[burst])


@register_system(
    field='mathematics/dynamical_systems',
    tags=('discrete map', 'basins', 'fractal'),
    references=('Dilão 2023',),
    defaults={'until': 50, 'dt': 1},
)
def newton_raphson_map(x: float = 0.5, y: float = 0.5) -> System:
    """Return Newton's method for `z³ = 1` in the complex plane, `z' = (2z³ + 1) / (3z²)`, in real coordinates.

    Every start except a set of measure zero converges to one of the three cube roots of unity, and the boundaries of
    their basins are fractal.

    Args:
        x:
            The initial real part.

        y:
            The initial imaginary part.

    Returns:
        system:
            The system, with the states `map.x` and `map.y`.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import newton_raphson_map
        >>> newton_raphson_map(x=1.5).elements[0].x
        1.5
    """
    real = MapInteraction(
        'x_map',
        target='map',
        state='x',
        function=lambda s: (
            2 * s.target.x / 3 + (s.target.x**2 - s.target.y**2) / (3 * (s.target.x**2 + s.target.y**2) ** 2)
        ),
    )
    imaginary = MapInteraction(
        'y_map',
        target='map',
        state='y',
        function=lambda s: (
            2 * s.target.y / 3 - 2 * s.target.x * s.target.y / (3 * (s.target.x**2 + s.target.y**2) ** 2)
        ),
    )
    return System(elements=[NewtonMapElement('map', x=x, y=y)], interactions=[real, imaginary])


@register_system(
    field='mathematics/dynamical_systems',
    tags=('chaos', 'discrete map', 'mixing', 'hamiltonian'),
    references=('Arnold and Avez 1968', 'Dilão 2023'),
    defaults={'until': 200, 'dt': 1},
)
def cat_map(x: float = 0.1, y: float = 0.3) -> System:
    """Return Arnold's cat map `(x, y) → (2x + y, x + y) (mod 1)`, an area-preserving hyperbolic map of the torus.

    Args:
        x:
            The initial first coordinate, between 0 and 1.

        y:
            The initial second coordinate, between 0 and 1.

    Returns:
        system:
            The system, with the states `map.x` and `map.y`.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import cat_map
        >>> cat_map(x=0.5).elements[0].x
        0.5
    """
    first = MapInteraction('x_map', target='map', state='x', function=lambda s: mod(2 * s.target.x + s.target.y, 1))
    second = MapInteraction('y_map', target='map', state='y', function=lambda s: mod(s.target.x + s.target.y, 1))
    return System(elements=[CatMapElement('map', x=x, y=y)], interactions=[first, second])

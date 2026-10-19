"""Dynamical systems models: discrete maps."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

from sage.all__sagemath_symbolics import pi, sin

from ....core.model import Element, System, Update, mod, where
from ....core.runtime import register_model

GOLDEN_MEAN = (math.sqrt(5) - 1) / 2

LogisticMap = Element.define(
    'LogisticMap',
    states={'x': 0.2},
    parameters={'r': 3.7},
    laws=lambda s: [Update(s.x, s.r * s.x * (1 - s.x))],
)

HenonMap = Element.define(
    'HenonMap',
    states={'x': 0.0, 'y': 0.0},
    parameters={'a': 1.4, 'b': 0.3},
    laws=lambda s: [Update(s.x, 1 - s.a * s.x**2 + s.y), Update(s.y, s.b * s.x)],
)


@register_model(
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
            The model, with the state `map.x`.

    Examples:
        >>> from skcomplex.models.mathematics.dynamical_systems import logistic_map
        >>> logistic_map(r=3.2).elements[0].r
        3.2
    """
    return System(elements=[LogisticMap('map', x=x, r=r)])


@register_model(
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
            The model, with the states `map.x` and `map.y`.

    Examples:
        >>> from skcomplex.models.mathematics.dynamical_systems import henon_map
        >>> henon_map(a=1.2).elements[0].a
        1.2
    """
    return System(elements=[HenonMap('map', x=x, y=y, a=a, b=b)])


StandardMap = Element.define(
    'StandardMap',
    states={'theta': 0.0, 'p': 2 * math.pi * GOLDEN_MEAN},
    parameters={'K': 1.2},
    laws=lambda s: [
        Update(s.p, s.p + s.K * sin(s.theta)),
        Update(s.theta, mod(s.theta + s.p + s.K * sin(s.theta), 2 * pi)),
    ],
)

CircleMap = Element.define(
    'CircleMap',
    states={'theta': 0.0},
    parameters={'Omega': 0.25, 'K': 0.9},
    laws=lambda s: [Update(s.theta, mod(s.theta + s.Omega - s.K / (2 * pi) * sin(2 * pi * s.theta), 1))],
)

TentMap = Element.define(
    'TentMap',
    states={'x': 0.3},
    parameters={'mu': 1.99},
    laws=lambda s: [Update(s.x, s.mu * where(s.x < 1 / 2, s.x, 1 - s.x))],
)

BakerMap = Element.define(
    'BakerMap',
    states={'x': 0.2, 'y': 0.3},
    parameters={'alpha': 1 / 3, 'lambda_a': 1 / 3, 'lambda_b': 2 / 3},
    laws=lambda s: [
        Update(s.x, where(s.x < s.alpha, s.x / s.alpha, (s.x - s.alpha) / (1 - s.alpha))),
        Update(s.y, where(s.x < s.alpha, s.lambda_a * s.y, 1 - s.lambda_b + s.lambda_b * s.y)),
    ],
)

TowelMap = Element.define(
    'TowelMap',
    states={'x': 0.085, 'y': -0.121, 'z': 0.075},
    parameters={'a': 3.8, 'b': 3.78},
    laws=lambda s: [
        Update(s.x, s.a * s.x * (1 - s.x) - 0.05 * (s.y + 0.35) * (1 - 2 * s.z)),
        Update(s.y, 0.1 * ((s.y + 0.35) * (1 - 2 * s.z) - 1) * (1 - 1.9 * s.x)),
        Update(s.z, s.b * s.z * (1 - s.z) + 0.2 * s.y),
    ],
)

IntermittencyMap = Element.define(
    'IntermittencyMap',
    states={'x': 0.01},
    parameters={'z': 2.0},
    laws=lambda s: [Update(s.x, mod(s.x + s.x**s.z, 1))],
)

NewtonMap = Element.define(
    'NewtonMap',
    states={'x': 0.5, 'y': 0.5},
    laws=lambda s: [
        Update(s.x, 2 * s.x / 3 + (s.x**2 - s.y**2) / (3 * (s.x**2 + s.y**2) ** 2)),
        Update(s.y, 2 * s.y / 3 - 2 * s.x * s.y / (3 * (s.x**2 + s.y**2) ** 2)),
    ],
)

CatMap = Element.define(
    'CatMap',
    states={'x': 0.1, 'y': 0.3},
    laws=lambda s: [Update(s.x, mod(2 * s.x + s.y, 1)), Update(s.y, mod(s.x + s.y, 1))],
)


@register_model(
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
            The model, with the states `map.theta` and `map.p`.

    Examples:
        >>> from skcomplex.models.mathematics.dynamical_systems import standard_map
        >>> standard_map(K=0.5).elements[0].K
        0.5
    """
    return System(elements=[StandardMap('map', theta=theta, p=p, K=K)])


@register_model(
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
            The model, with the state `map.theta`.

    Examples:
        >>> from skcomplex.models.mathematics.dynamical_systems import circle_map
        >>> circle_map(K=1.0).elements[0].K
        1.0
    """
    return System(elements=[CircleMap('map', theta=theta, Omega=Omega, K=K)])


@register_model(
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
            The model, with the state `map.x`.

    Examples:
        >>> from skcomplex.models.mathematics.dynamical_systems import tent_map
        >>> tent_map(mu=1.5).elements[0].mu
        1.5
    """
    return System(elements=[TentMap('map', x=x, mu=mu)])


@register_model(
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
            The model, with the states `map.x` and `map.y`.

    Examples:
        >>> from skcomplex.models.mathematics.dynamical_systems import baker_map
        >>> baker_map(lambda_a=0.2).elements[0].lambda_a
        0.2
    """
    return System(elements=[BakerMap('map', x=x, y=y, alpha=alpha, lambda_a=lambda_a, lambda_b=lambda_b)])


@register_model(
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
            The model, with the states `map.x`, `map.y`, and `map.z`.

    Examples:
        >>> from skcomplex.models.mathematics.dynamical_systems import towel_map
        >>> towel_map(a=3.7).elements[0].a
        3.7
    """
    return System(elements=[TowelMap('map', x=x, y=y, z=z, a=a, b=b)])


@register_model(
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
            The model, with the state `map.x`.

    Examples:
        >>> from skcomplex.models.mathematics.dynamical_systems import intermittency_map
        >>> intermittency_map(z=1.5).elements[0].z
        1.5
    """
    return System(elements=[IntermittencyMap('map', x=x, z=z)])


@register_model(
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
            The model, with the states `map.x` and `map.y`.

    Examples:
        >>> from skcomplex.models.mathematics.dynamical_systems import newton_raphson_map
        >>> newton_raphson_map(x=1.5).elements[0].x
        1.5
    """
    return System(elements=[NewtonMap('map', x=x, y=y)])


@register_model(
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
            The model, with the states `map.x` and `map.y`.

    Examples:
        >>> from skcomplex.models.mathematics.dynamical_systems import cat_map
        >>> cat_map(x=0.5).elements[0].x
        0.5
    """
    return System(elements=[CatMap('map', x=x, y=y)])

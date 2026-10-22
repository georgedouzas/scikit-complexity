"""Dynamical systems models: normal forms of bifurcations."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.model import Element, Rate, System
from ....core.runtime import register_model

SaddleNode = Element.define(
    'SaddleNode',
    states={'x': 0.0},
    parameters={'mu': 1.0},
    laws=lambda s: [Rate(s.x, s.mu - s.x**2)],
)

Hopf = Element.define(
    'Hopf',
    states={'x': 0.1, 'y': 0.0},
    parameters={'mu': 0.25, 'omega': 1.0, 'a': -1.0},
    laws=lambda s: [
        Rate(s.x, s.mu * s.x - s.omega * s.y + s.a * s.x * (s.x**2 + s.y**2)),
        Rate(s.y, s.omega * s.x + s.mu * s.y + s.a * s.y * (s.x**2 + s.y**2)),
    ],
)

TakensBogdanov = Element.define(
    'TakensBogdanov',
    states={'x': 0.2, 'y': 0.0},
    parameters={'beta1': 0.1, 'beta2': -1.0, 'sign': -1.0},
    laws=lambda s: [Rate(s.x, s.y), Rate(s.y, s.beta1 + s.beta2 * s.x + s.x**2 + s.sign * s.x * s.y)],
)

Cusp = Element.define(
    'Cusp',
    states={'x': 0.5},
    parameters={'beta1': 0.0, 'beta2': 1.0},
    laws=lambda s: [Rate(s.x, s.beta1 + s.beta2 * s.x - s.x**3)],
)


@register_model(
    field='mathematics/dynamical_systems',
    tags=('bifurcation', 'normal form', 'ode'),
    references=('Gros 2015', 'Dilão 2023'),
    defaults={'until': 10.0, 'dt': 0.05},
)
def saddle_node_form(x: float = 0.0, *, mu: float = 1.0) -> System:
    """Return the saddle-node normal form `x' = μ - x²`, whose two equilibria `±√μ` meet and vanish at μ = 0.

    Args:
        x:
            The initial value.

        mu:
            The bifurcation parameter.

    Returns:
        system:
            The model, with the state `form.x`.

    Examples:
        >>> from skcomplex.models.mathematics.dynamical_systems import saddle_node_form
        >>> saddle_node_form(mu=0.5).elements[0].mu
        0.5
    """
    return System(components=[SaddleNode('form', x=x, mu=mu)])


@register_model(
    field='mathematics/dynamical_systems',
    tags=('bifurcation', 'normal form', 'ode', 'limit cycle'),
    references=('Gros 2015', 'Dilão 2023'),
    defaults={'until': 50.0, 'dt': 0.1},
)
def hopf_form(x: float = 0.1, y: float = 0.0, *, mu: float = 0.25, omega: float = 1.0, a: float = -1.0) -> System:
    """Return the Hopf normal form, whose focus gives birth to a limit cycle of radius `√(-μ / a)` as μ crosses 0.

    `x' = μx - ωy + a x (x² + y²)` and `y' = ωx + μy + a y (x² + y²)`, supercritical for `a < 0`.

    Args:
        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        mu:
            The bifurcation parameter.

        omega:
            The angular frequency.

        a:
            The cubic coefficient, the sign of the first Lyapunov coefficient.

    Returns:
        system:
            The model, with the states `form.x` and `form.y`.

    Examples:
        >>> from skcomplex.models.mathematics.dynamical_systems import hopf_form
        >>> hopf_form(mu=-0.1).elements[0].mu
        -0.1
    """
    return System(components=[Hopf('form', x=x, y=y, mu=mu, omega=omega, a=a)])


@register_model(
    field='mathematics/dynamical_systems',
    tags=('bifurcation', 'normal form', 'ode', 'codimension two'),
    references=('Gros 2015', 'Dilão 2023'),
    defaults={'until': 50.0, 'dt': 0.1},
)
def takens_bogdanov_form(
    x: float = 0.2,
    y: float = 0.0,
    *,
    beta1: float = 0.1,
    beta2: float = -1.0,
    sign: float = -1.0,
) -> System:
    """Return the Takens-Bogdanov normal form `x' = y`, `y' = β1 + β2 x + x² + s x y`, where a saddle-node meets a Hopf.

    Args:
        x:
            The initial position.

        y:
            The initial velocity.

        beta1:
            The first unfolding parameter.

        beta2:
            The second unfolding parameter.

        sign:
            The sign `s` of the mixed term, `-1` or `1`.

    Returns:
        system:
            The model, with the states `form.x` and `form.y`.

    Examples:
        >>> from skcomplex.models.mathematics.dynamical_systems import takens_bogdanov_form
        >>> takens_bogdanov_form(beta1=0.0).elements[0].beta1
        0.0
    """
    return System(components=[TakensBogdanov('form', x=x, y=y, beta1=beta1, beta2=beta2, sign=sign)])


@register_model(
    field='mathematics/dynamical_systems',
    tags=('bifurcation', 'normal form', 'ode', 'bistability', 'hysteresis'),
    references=('Gros 2015', 'Dilão 2023'),
    defaults={'until': 20.0, 'dt': 0.1},
)
def cusp_form(x: float = 0.5, *, beta1: float = 0.0, beta2: float = 1.0) -> System:
    """Return the cusp normal form `x' = β1 + β2 x - x³`, bistable inside the cusp `27 β1² < 4 β2³`.

    Args:
        x:
            The initial value.

        beta1:
            The asymmetry.

        beta2:
            The splitting parameter.

    Returns:
        system:
            The model, with the state `form.x`.

    Examples:
        >>> from skcomplex.models.mathematics.dynamical_systems import cusp_form
        >>> cusp_form(beta1=0.2).elements[0].beta1
        0.2
    """
    return System(components=[Cusp('form', x=x, beta1=beta1, beta2=beta2)])

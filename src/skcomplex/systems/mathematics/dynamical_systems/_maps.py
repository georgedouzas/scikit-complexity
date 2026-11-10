"""Dynamical systems of discrete maps."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import Annotated

from sage.all__sagemath_symbolics import abs_symbolic, cos, exp, pi, sin

from ....core.components.elements import Element
from ....core.components.interactions import MapInteraction
from ....core.components.laws.utils import mod, where
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations

GOLDEN_MEAN = (math.sqrt(5) - 1) / 2


class LogisticMapElement(Element):
    """Value and growth rate of the logistic map `x -> r x (1 - x)`.

    A `MapInteraction` iterates the value.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial value, between 0 and 1.

        r:
            The growth rate, between 0 and 4.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import LogisticMapElement
        >>> LogisticMapElement('map', r=3.2).r
        3.2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.2,
        r: Annotated[object, ParameterSpec()] = 3.7,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.r = r


class LogisticMapSystem(BaseSystem):
    """The logistic map `x_{t+1} = r x_t (1 - x_t)`.

    The map reaches chaos through period doubling as `r` grows.

    The system has the state `map.x`.

    Args:
        x:
            The initial value, between 0 and 1.

        r:
            The growth rate, between 0 and 4.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import LogisticMapSystem
        >>> LogisticMapSystem(r=3.2).prepare().system_.elements[0].r
        3.2
    """

    field = 'mathematics/dynamical_systems'
    topics = ('chaos', 'discrete map', 'period doubling')
    references = ('May 1976', 'Satz 2022')
    until = 200
    dt = 1

    def __init__(
        self,
        x: float = 0.2,
        *,
        r: float = 3.7,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.x = x
        self.r = r

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system, built from its model parameters."""
        logistic = MapInteraction(
            'x_map',
            target='map',
            state='x',
            function=lambda s: s.target.r * s.target.x * (1 - s.target.x),
        )
        return CompositeSystem(
            self.label,
            elements=[LogisticMapElement('map', x=self.x, r=self.r)],
            interactions=[logistic],
        )


class HenonMapSystem(DynamicalSystem):
    """The Hénon map `x' = 1 - a x² + y`, `y' = b x`.

    The map has a fractal strange attractor for its defaults.

    The system has the states `map.x` and `map.y`.

    Args:
        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        a:
            The nonlinearity.

        b:
            The contraction.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import HenonMapSystem
        >>> HenonMapSystem(a=1.2).prepare().system_.elements[0].get_value('a')
        1.2
    """

    field = 'mathematics/dynamical_systems'
    topics = ('chaos', 'discrete map', 'strange attractor')
    references = ('Hénon 1976', 'Datseris and Parlitz 2022')
    until = 1000
    dt = 1

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        *,
        a: float = 1.4,
        b: float = 0.3,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.x = x
        self.y = y
        self.a = a
        self.b = b

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'map'

    def _get_time(self) -> str:
        """Return the discrete time of the map."""
        return 'discrete'

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state."""
        return {
            'x': self.x,
            'y': self.y,
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the parameters of the equations."""
        return {
            'a': self.a,
            'b': self.b,
        }

    def _get_equations(self) -> Equations:
        """Return the Hénon map `x' = 1 - a x² + y`, `y' = b x`."""
        return {
            'x': lambda s: 1 - s.target.a * s.target.x**2 + s.target.y,
            'y': lambda s: s.target.b * s.target.x,
        }


class StandardMapSystem(DynamicalSystem):
    """The Chirikov standard map `p' = p + K sin θ`, `θ' = θ + p'`.

    The map has global chaos above K ≈ 0.9716.

    The last invariant circle that spans the angle, the golden-mean torus, breaks at K ≈ 0.971635. The angle is taken
    modulo 2π and the momentum is kept on the real line, so its spread measures diffusion. The system has the states
    `map.theta` and `map.p`.

    Args:
        theta:
            The initial angle.

        p:
            The initial momentum, by default on the golden-mean torus of the unkicked rotor.

        K:
            The kick strength.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import StandardMapSystem
        >>> StandardMapSystem(K=0.5).prepare().system_.elements[0].get_value('K')
        0.5
    """

    field = 'mathematics/dynamical_systems'
    topics = ('chaos', 'discrete map', 'hamiltonian', 'kam')
    references = ('Chirikov 1979', 'Datseris and Parlitz 2022', 'Dilão 2023', 'Lindgren 2024')
    until = 1000
    dt = 1

    def __init__(
        self,
        theta: float = 0.0,
        p: float = 2 * math.pi * GOLDEN_MEAN,
        *,
        K: float = 1.2,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.theta = theta
        self.p = p
        self.K = K

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'map'

    def _get_time(self) -> str:
        """Return the discrete time of the map."""
        return 'discrete'

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state."""
        return {
            'theta': self.theta,
            'p': self.p,
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the parameters of the equations."""
        return {
            'K': self.K,
        }

    def _get_equations(self) -> Equations:
        """Return the Chirikov standard map `p' = p + K sin θ`, `θ' = θ + p'`."""
        return {
            'p': lambda s: s.target.p + s.target.K * sin(s.target.theta),
            'theta': lambda s: mod(s.target.theta + s.target.p + s.target.K * sin(s.target.theta), 2 * pi),
        }


class CircleMapSystem(DynamicalSystem):
    """The sine circle map `θ' = θ + Ω - K / (2π) sin 2πθ (mod 1)`.

    The map locks into Arnold tongues.

    The system has the state `map.theta`.

    Args:
        theta:
            The initial angle, as a fraction of a turn.

        Omega:
            The bare rotation per step.

        K:
            The nonlinearity, invertible below 1.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import CircleMapSystem
        >>> CircleMapSystem(K=1.0).prepare().system_.elements[0].get_value('K')
        1.0
    """

    field = 'mathematics/dynamical_systems'
    topics = ('discrete map', 'mode locking', 'arnold tongues')
    references = ('Arnold 1965', 'Datseris and Parlitz 2022', 'Dilão 2023')
    until = 500
    dt = 1

    def __init__(
        self,
        theta: float = 0.0,
        *,
        Omega: float = 0.25,
        K: float = 0.9,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.theta = theta
        self.Omega = Omega
        self.K = K

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'map'

    def _get_time(self) -> str:
        """Return the discrete time of the map."""
        return 'discrete'

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state."""
        return {
            'theta': self.theta,
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the parameters of the equations."""
        return {
            'Omega': self.Omega,
            'K': self.K,
        }

    def _get_equations(self) -> Equations:
        """Return the sine circle map `θ' = θ + Ω - K / (2π) sin 2πθ (mod 1)`."""
        return {
            'theta': lambda s: mod(
                s.target.theta + s.target.Omega - s.target.K / (2 * pi) * sin(2 * pi * s.target.theta),
                1,
            ),
        }


class TentMapSystem(DynamicalSystem):
    """The tent map `x' = μ min(x, 1 - x)`.

    The map is chaotic with Lyapunov exponent `ln μ` for μ above 1.

    The system has the state `map.x`.

    Args:
        x:
            The initial value, between 0 and 1.

        mu:
            The slope, between 0 and 2.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import TentMapSystem
        >>> TentMapSystem(mu=1.5).prepare().system_.elements[0].get_value('mu')
        1.5
    """

    field = 'mathematics/dynamical_systems'
    topics = ('chaos', 'discrete map', 'piecewise linear')
    references = ('Datseris and Parlitz 2022', 'Dilão 2023')
    until = 200
    dt = 1

    def __init__(
        self,
        x: float = 0.3,
        *,
        mu: float = 1.99,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.x = x
        self.mu = mu

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'map'

    def _get_time(self) -> str:
        """Return the discrete time of the map."""
        return 'discrete'

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state."""
        return {
            'x': self.x,
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the parameters of the equations."""
        return {
            'mu': self.mu,
        }

    def _get_equations(self) -> Equations:
        """Return the tent map `x' = μ min(x, 1 - x)`."""
        return {
            'x': lambda s: s.target.mu * where(s.target.x < 1 / 2, s.target.x, 1 - s.target.x),
        }


class BakerMapSystem(DynamicalSystem):
    """The generalized baker map.

    The map stretches the unit square along `x` and stacks the two pieces in `y`.

    The strip `x < alpha` is stretched by `1 / alpha` and squeezed into the bottom band of height `lambda_a`, and the
    rest is stretched by `1 / (1 - alpha)` and squeezed into the top band of height `lambda_b`. The map preserves area
    when `lambda_a = alpha` and `lambda_b = 1 - alpha`, and it has a fractal attractor when `lambda_a + lambda_b < 1`.
    The system has the states `map.x` and `map.y`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import BakerMapSystem
        >>> BakerMapSystem(lambda_a=0.2).prepare().system_.elements[0].get_value('lambda_a')
        0.2
    """

    field = 'mathematics/dynamical_systems'
    topics = ('chaos', 'discrete map', 'mixing', 'fractal')
    references = ('Farmer, Ott, and Yorke 1983', 'Datseris and Parlitz 2022', 'Lindgren 2024')
    until = 500
    dt = 1

    def __init__(
        self,
        x: float = 0.2,
        y: float = 0.3,
        *,
        alpha: float = 1 / 3,
        lambda_a: float = 1 / 3,
        lambda_b: float = 2 / 3,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.x = x
        self.y = y
        self.alpha = alpha
        self.lambda_a = lambda_a
        self.lambda_b = lambda_b

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'map'

    def _get_time(self) -> str:
        """Return the discrete time of the map."""
        return 'discrete'

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
            'lambda_a': self.lambda_a,
            'lambda_b': self.lambda_b,
        }

    def _get_equations(self) -> Equations:
        """Return the parts of the system, built from its model parameters."""
        return {
            'x': lambda s: where(
                s.target.x < s.target.alpha,
                s.target.x / s.target.alpha,
                (s.target.x - s.target.alpha) / (1 - s.target.alpha),
            ),
            'y': lambda s: where(
                s.target.x < s.target.alpha,
                s.target.lambda_a * s.target.y,
                1 - s.target.lambda_b + s.target.lambda_b * s.target.y,
            ),
        }


class TowelMapSystem(DynamicalSystem):
    """Rössler's towel map.

    It is a three-dimensional map with two positive Lyapunov exponents.

    `x' = a x (1 - x) - 0.05 (y + 0.35)(1 - 2z)`, `y' = 0.1 ((y + 0.35)(1 - 2z) - 1)(1 - 1.9 x)`, and `z' = b z (1 - z)
    + 0.2 y`. The system has the states `map.x`, `map.y`, and `map.z`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import TowelMapSystem
        >>> TowelMapSystem(a=3.7).prepare().system_.elements[0].get_value('a')
        3.7
    """

    field = 'mathematics/dynamical_systems'
    topics = ('chaos', 'discrete map', 'hyperchaos')
    references = ('Rössler 1979', 'Datseris and Parlitz 2022')
    until = 1000
    dt = 1

    def __init__(
        self,
        x: float = 0.085,
        y: float = -0.121,
        z: float = 0.075,
        *,
        a: float = 3.8,
        b: float = 3.78,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.x = x
        self.y = y
        self.z = z
        self.a = a
        self.b = b

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'map'

    def _get_time(self) -> str:
        """Return the discrete time of the map."""
        return 'discrete'

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state."""
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z,
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the parameters of the equations."""
        return {
            'a': self.a,
            'b': self.b,
        }

    def _get_equations(self) -> Equations:
        """Return Rössler's towel map."""
        return {
            'x': lambda s: (
                s.target.a * s.target.x * (1 - s.target.x) - 0.05 * (s.target.y + 0.35) * (1 - 2 * s.target.z)
            ),
            'y': lambda s: 0.1 * ((s.target.y + 0.35) * (1 - 2 * s.target.z) - 1) * (1 - 1.9 * s.target.x),
            'z': lambda s: s.target.b * s.target.z * (1 - s.target.z) + 0.2 * s.target.y,
        }


class IntermittencyMapSystem(DynamicalSystem):
    """The Pomeau-Manneville map `x' = x + x^z (mod 1)`.

    Its long laminar phases near 0 end in chaotic bursts.

    The system has the state `map.x`.

    Args:
        x:
            The initial value, between 0 and 1.

        z:
            The exponent of the tangency at 0, above 1.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import IntermittencyMapSystem
        >>> IntermittencyMapSystem(z=1.5).prepare().system_.elements[0].get_value('z')
        1.5
    """

    field = 'mathematics/dynamical_systems'
    topics = ('chaos', 'discrete map', 'intermittency')
    references = ('Pomeau and Manneville 1980', 'Dilão 2023', 'Jensen 2023')
    until = 1000
    dt = 1

    def __init__(
        self,
        x: float = 0.01,
        *,
        z: float = 2.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.x = x
        self.z = z

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'map'

    def _get_time(self) -> str:
        """Return the discrete time of the map."""
        return 'discrete'

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state."""
        return {
            'x': self.x,
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the parameters of the equations."""
        return {
            'z': self.z,
        }

    def _get_equations(self) -> Equations:
        """Return the parts of the system, built from its model parameters."""
        return {
            'x': lambda s: mod(s.target.x + s.target.x**s.target.z, 1),
        }


class NewtonRaphsonMapSystem(DynamicalSystem):
    """Newton's method for `z³ = 1` in the complex plane, in real coordinates.

    The method is `z' = (2z³ + 1) / (3z²)`.

    Every start except a set of measure zero converges to one of the three cube roots of unity, and the boundaries of
    their basins are fractal. The system has the states `map.x` and `map.y`.

    Args:
        x:
            The initial real part.

        y:
            The initial imaginary part.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import NewtonRaphsonMapSystem
        >>> NewtonRaphsonMapSystem(x=1.5).prepare().system_.elements[0].get_value('x')
        1.5
    """

    field = 'mathematics/dynamical_systems'
    topics = ('discrete map', 'basins', 'fractal')
    references = ('Dilão 2023',)
    until = 50
    dt = 1

    def __init__(
        self,
        x: float = 0.5,
        y: float = 0.5,
        *,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.x = x
        self.y = y

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'map'

    def _get_time(self) -> str:
        """Return the discrete time of the map."""
        return 'discrete'

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state."""
        return {
            'x': self.x,
            'y': self.y,
        }

    def _get_equations(self) -> Equations:
        """Return Newton's method for `z³ = 1` in the complex plane, in real coordinates."""
        return {
            'x': lambda s: (
                2 * s.target.x / 3 + (s.target.x**2 - s.target.y**2) / (3 * (s.target.x**2 + s.target.y**2) ** 2)
            ),
            'y': lambda s: (
                2 * s.target.y / 3 - 2 * s.target.x * s.target.y / (3 * (s.target.x**2 + s.target.y**2) ** 2)
            ),
        }


class CatMapSystem(DynamicalSystem):
    """Arnold's cat map `(x, y) → (2x + y, x + y) (mod 1)`.

    It is an area-preserving hyperbolic map of the torus.

    The system has the states `map.x` and `map.y`.

    Args:
        x:
            The initial first coordinate, between 0 and 1.

        y:
            The initial second coordinate, between 0 and 1.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import CatMapSystem
        >>> CatMapSystem(x=0.5).prepare().system_.elements[0].get_value('x')
        0.5
    """

    field = 'mathematics/dynamical_systems'
    topics = ('chaos', 'discrete map', 'mixing', 'hamiltonian')
    references = ('Arnold and Avez 1968', 'Dilão 2023')
    until = 200
    dt = 1

    def __init__(
        self,
        x: float = 0.1,
        y: float = 0.3,
        *,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.x = x
        self.y = y

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'map'

    def _get_time(self) -> str:
        """Return the discrete time of the map."""
        return 'discrete'

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state."""
        return {
            'x': self.x,
            'y': self.y,
        }

    def _get_equations(self) -> Equations:
        """Return the parts of the system, built from its model parameters."""
        return {
            'x': lambda s: mod(2 * s.target.x + s.target.y, 1),
            'y': lambda s: mod(s.target.x + s.target.y, 1),
        }


class LoziMapSystem(DynamicalSystem):
    """The Lozi map `x' = 1 + y - a |x|`, `y' = b x`.

    It is the piecewise linear Hénon map with a strange attractor.

    The fold of the Hénon map becomes a corner, so the attractor is made of straight segments, and for the defaults of
    Lozi (1978) the orbit is chaotic (Dilão 2023, Exercise 3.16). The system has the states `map.x` and `map.y`.

    Args:
        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        a:
            The slope of the fold.

        b:
            The contraction.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import LoziMapSystem
        >>> LoziMapSystem(a=1.5).prepare().system_.elements[0].get_value('a')
        1.5
    """

    field = 'mathematics/dynamical_systems'
    topics = ('chaos', 'discrete map', 'strange attractor', 'piecewise linear')
    references = ('Lozi 1978', 'Dilão 2023')
    until = 1000
    dt = 1

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        *,
        a: float = 1.7,
        b: float = 0.5,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.x = x
        self.y = y
        self.a = a
        self.b = b

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'map'

    def _get_time(self) -> str:
        """Return the discrete time of the map."""
        return 'discrete'

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state."""
        return {
            'x': self.x,
            'y': self.y,
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the parameters of the equations."""
        return {
            'a': self.a,
            'b': self.b,
        }

    def _get_equations(self) -> Equations:
        """Return the parts of the system, built from its model parameters."""
        return {
            'x': lambda s: 1 + s.target.y - s.target.a * abs_symbolic(s.target.x),
            'y': lambda s: s.target.b * s.target.x,
        }


class ConservativeHenonMapSystem(DynamicalSystem):
    """Hénon's area-preserving quadratic map.

    The map is a rotation by `2π nu` with a quadratic kick.

    The map is `x' = a x + b y + b x²`, `y' = -b x + a y + a x²` with `a = cos 2π nu` and `b = sin 2π nu` (Dilão 2023,
    Example 3.17 and Exercise 2.8). For `nu` in `(0, 1/2)` the origin is a centre surrounded by invariant curves, and
    `(2 tan(π nu), -2 tan²(π nu))` is a saddle whose manifolds bound the region of bounded orbits, the dynamic aperture
    of accelerator physics. The default `nu` gives `cos 2π nu = 0.24`, the case of Hénon (1969). The system has the
    states `map.x` and `map.y`.

    Args:
        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        nu:
            The rotation number of the linear part, in turns.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import ConservativeHenonMapSystem
        >>> ConservativeHenonMapSystem(nu=0.25).prepare().system_.elements[0].get_value('nu')
        0.25
    """

    field = 'mathematics/dynamical_systems'
    topics = ('discrete map', 'area preserving', 'hamiltonian', 'kam', 'dynamic aperture')
    references = ('Hénon 1969', 'Dilão 2023')
    until = 1000
    dt = 1

    def __init__(
        self,
        x: float = 0.1,
        y: float = 0.0,
        *,
        nu: float = 0.2114,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.x = x
        self.y = y
        self.nu = nu

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'map'

    def _get_time(self) -> str:
        """Return the discrete time of the map."""
        return 'discrete'

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state."""
        return {
            'x': self.x,
            'y': self.y,
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the parameters of the equations."""
        return {
            'nu': self.nu,
        }

    def _get_equations(self) -> Equations:
        """Return Hénon's area-preserving quadratic map."""
        return {
            'x': lambda s: (
                cos(2 * pi * s.target.nu) * s.target.x + sin(2 * pi * s.target.nu) * (s.target.y + s.target.x**2)
            ),
            'y': lambda s: (
                -sin(2 * pi * s.target.nu) * s.target.x + cos(2 * pi * s.target.nu) * (s.target.y + s.target.x**2)
            ),
        }


class FermiUlamMapSystem(DynamicalSystem):
    """The Fermi-Ulam map of a ball bouncing between two walls.

    One wall is fixed and the other oscillates in time.

    The speed and the phase of the wall at a bounce follow `u' = |u + ε sin θ|`, `θ' = θ + 2π / u' (mod 2π)`, an
    area-preserving map with a mixed state space like the standard map (Datseris and Parlitz 2022, Exercise 1.11). The
    points `θ = 0` and `θ = π` at the speeds `u = 1 / m` are its fixed points. The system has the states `map.u` and
    `map.theta`.

    Args:
        u:
            The initial speed of the ball, positive.

        theta:
            The initial phase of the oscillating wall.

        epsilon:
            The amplitude of the speed the oscillating wall gives the ball.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import FermiUlamMapSystem
        >>> FermiUlamMapSystem(epsilon=0.2).prepare().system_.elements[0].get_value('epsilon')
        0.2
    """

    field = 'mathematics/dynamical_systems'
    topics = ('discrete map', 'area preserving', 'billiard', 'fermi acceleration')
    references = ('Ulam 1961', 'Lichtenberg and Lieberman 1992', 'Datseris and Parlitz 2022')
    until = 1000
    dt = 1

    def __init__(
        self,
        u: float = 0.5,
        theta: float = 1.0,
        *,
        epsilon: float = 0.1,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.u = u
        self.theta = theta
        self.epsilon = epsilon

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'map'

    def _get_time(self) -> str:
        """Return the discrete time of the map."""
        return 'discrete'

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state."""
        return {
            'u': self.u,
            'theta': self.theta,
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the parameters of the equations."""
        return {
            'epsilon': self.epsilon,
        }

    def _get_equations(self) -> Equations:
        """Return the Fermi-Ulam map of a ball bouncing between two walls."""
        return {
            'u': lambda s: abs_symbolic(s.target.u + s.target.epsilon * sin(s.target.theta)),
            'theta': lambda s: mod(
                s.target.theta + 2 * pi / abs_symbolic(s.target.u + s.target.epsilon * sin(s.target.theta)),
                2 * pi,
            ),
        }


class CDNMapSystem(DynamicalSystem):
    """The Cascais-Dilão-Noronha Gaussian map `x' = exp(-alpha² x²) + β`.

    The map acts on an interval.

    For `alpha = 2.8` the map is chaotic over most of `β` between about -0.8 and -0.25, with periodic windows, and its
    complexity rises and falls again with `β` through period doubling and reverse period doubling, as in the bifurcation
    diagram and the topological entropy of Dilão (2023, Fig. 3.17). Dilão's Exercises 3.4 and 3.5 write the exponent as
    `-alpha² x² / 2π`, the same family with `alpha` scaled by `√(2π)`. The system has the state `map.x`.

    Args:
        x:
            The initial value.

        alpha:
            The steepness of the Gaussian.

        beta:
            The shift of the Gaussian.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import CDNMapSystem
        >>> CDNMapSystem(beta=-0.3).prepare().system_.elements[0].get_value('beta')
        -0.3
    """

    field = 'mathematics/dynamical_systems'
    topics = ('chaos', 'discrete map', 'interval map', 'period doubling', 'reverse bifurcations')
    references = ('Cascais, Dilão, and Noronha da Costa 1983', 'Dilão 2023')
    until = 200
    dt = 1

    def __init__(
        self,
        x: float = 0.1,
        *,
        alpha: float = 2.8,
        beta: float = -0.5,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.x = x
        self.alpha = alpha
        self.beta = beta

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'map'

    def _get_time(self) -> str:
        """Return the discrete time of the map."""
        return 'discrete'

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state."""
        return {
            'x': self.x,
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the parameters of the equations."""
        return {
            'alpha': self.alpha,
            'beta': self.beta,
        }

    def _get_equations(self) -> Equations:
        """Return the Cascais-Dilão-Noronha Gaussian map `x' = exp(-alpha² x²) + β`."""
        return {
            'x': lambda s: exp(-(s.target.alpha**2) * s.target.x**2) + s.target.beta,
        }

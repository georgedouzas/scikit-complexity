"""Dynamical systems of discrete maps."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import abs_symbolic, cos, exp, pi, sin

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import mod, where
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role
from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations

GOLDEN_MEAN = (math.sqrt(5) - 1) / 2


class LogisticMapInteraction(Interaction):
    """The law of the logistic map `x' = r x (1 - x)`.

    The growth rate `r` is a value of the element, so a group of maps can give every member a rate of its own.

    Args:
        label:
            The label of the interaction.

        target:
            The element whose state `x` the map moves, with the growth rate `r` among its values.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import LogisticMapInteraction
        >>> LogisticMapInteraction('logistic', target='map').target
        'map'
    """

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, target: Role = 'map') -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the next value of the state as the law.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        x, r = context.target.x, context.target.r
        self.laws_: list[BaseLaw] = [UpdateLaw(x, r * x * (1 - x))]


class LogisticMapSystem(BaseSystem):
    """The logistic map `x_{t+1} = r x_t (1 - x_t)`.

    The map reaches chaos through period doubling as `r` grows.

    The system has the state `map.x` and the parameter `map.r`. An array of initial values gives an ensemble of
    maps iterated together, with one entry of `map.x` per map.

    Args:
        x:
            The initial value, between 0 and 1, or a one-dimensional array of initial values, one per map.

        r:
            The growth rate, between 0 and 4.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import LogisticMapSystem
        >>> LogisticMapSystem(r=3.2).prepare().simulate(until=2).get_values('map.x').round(3).tolist()
        [0.2, 0.512, 0.8]
        >>> ensemble = LogisticMapSystem([0.2, 0.5], r=3.2).prepare().simulate(until=1)
        >>> ensemble.get_values('map.x').round(3).tolist()
        [[0.2, 0.5], [0.512, 0.8]]
    """

    field = 'mathematics/dynamical_systems'
    topics = ('chaos', 'discrete map', 'period doubling')
    references = ('May 1976', 'Satz 2022')
    until = 200
    dt = 1

    def __init__(
        self,
        x: npt.ArrayLike = 0.2,
        *,
        r: float = 3.7,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.r = r

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the map and its law."""
        super()._prepare(context)
        size = None if np.ndim(self.x) == 0 else len(np.asarray(self.x))
        self.system_ = System(
            self.label,
            elements=[Element('map', values={'x': self.x, 'r': self.r}, size=size)],
            interactions=[LogisticMapInteraction('logistic', target='map')],
        )


class HenonMapInteraction(Interaction):
    """The laws of the Hénon map `x' = 1 - a x² + y`, `y' = b x`.

    Args:
        label:
            The label of the interaction.

        target:
            The element whose states `x` and `y` the map moves.

        a:
            The nonlinearity.

        b:
            The contraction.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import HenonMapInteraction
        >>> HenonMapInteraction('henon', target='map').a
        1.4
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'map',
        a: float = 1.4,
        b: float = 0.3,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.a = a
        self.b = b

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the next value of every state as the laws.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        x, y = context.target.x, context.target.y
        a, b = context.a, context.b
        self.laws_: list[BaseLaw] = [
            UpdateLaw(x, 1 - a * x**2 + y),
            UpdateLaw(y, b * x),
        ]


class HenonMapSystem(BaseSystem):
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import HenonMapSystem
        >>> HenonMapSystem(a=1.2).prepare().system_.interactions[0].a
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.y = y
        self.a = a
        self.b = b

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the map and its laws."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[Element('map', values={'x': self.x, 'y': self.y})],
            interactions=[HenonMapInteraction('henon', target='map', a=self.a, b=self.b)],
        )


class StandardMapInteraction(Interaction):
    """The kick of the Chirikov standard map `p' = p + K sin θ`, `θ' = θ + p'`.

    The angle is taken modulo 2π and the momentum is kept on the real line.

    Args:
        label:
            The label of the interaction.

        target:
            The element whose angle `theta` and momentum `p` the map moves.

        K:
            The kick strength.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import StandardMapInteraction
        >>> StandardMapInteraction('kick', target='map', K=0.5).K
        0.5
    """

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, target: Role = 'map', K: float = 1.2) -> None:
        super().__init__(label)
        self.target = target
        self.K = K

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the kick of the momentum and the rotation of the angle as the laws.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        theta, p = context.target.theta, context.target.p
        kicked = p + context.K * sin(theta)
        self.laws_: list[BaseLaw] = [UpdateLaw(p, kicked), UpdateLaw(theta, mod(theta + kicked, 2 * pi))]


class StandardMapSystem(BaseSystem):
    """The Chirikov standard map `p' = p + K sin θ`, `θ' = θ + p'`.

    The map has global chaos above K ≈ 0.9716.

    The last invariant circle that spans the angle, the golden-mean torus, breaks at K ≈ 0.971635. The angle is taken
    modulo 2π and the momentum is kept on the real line, so its spread measures diffusion. The system has the states
    `map.theta` and `map.p`, and the kick strength `kick.K`.

    Args:
        theta:
            The initial angle.

        p:
            The initial momentum, by default on the golden-mean torus of the unkicked rotor.

        K:
            The kick strength.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import StandardMapSystem
        >>> system = StandardMapSystem(K=0.5).prepare()
        >>> system.states_
        ('map.theta', 'map.p')
        >>> system.simulate(until=2).get_values('map.p').round(3).tolist()
        [3.883, 3.883, 3.545]
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.theta = theta
        self.p = p
        self.K = K

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the map and its kick."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[Element('map', values={'theta': self.theta, 'p': self.p})],
            interactions=[StandardMapInteraction('kick', target='map', K=self.K)],
        )


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import CircleMapSystem
        >>> CircleMapSystem(K=1.0).prepare().system_.elements[0].values['K']
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label=label, solver=solver, backend=backend)
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import TentMapSystem
        >>> TentMapSystem(mu=1.5).prepare().system_.elements[0].values['mu']
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label=label, solver=solver, backend=backend)
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


class BakerMapInteraction(Interaction):
    """The laws of the generalized baker map.

    Args:
        label:
            The label of the interaction.

        target:
            The element whose states `x` and `y` the map moves.

        alpha:
            The position of the cut.

        lambda_a:
            The height of the bottom band.

        lambda_b:
            The height of the top band.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import BakerMapInteraction
        >>> BakerMapInteraction('baker', target='map', alpha=0.4).alpha
        0.4
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'map',
        alpha: float = 1 / 3,
        lambda_a: float = 1 / 3,
        lambda_b: float = 2 / 3,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.alpha = alpha
        self.lambda_a = lambda_a
        self.lambda_b = lambda_b

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the next value of every state as the laws.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        x, y = context.target.x, context.target.y
        alpha, lambda_a, lambda_b = context.alpha, context.lambda_a, context.lambda_b
        self.laws_: list[BaseLaw] = [
            UpdateLaw(
                x,
                where(
                    x < alpha,
                    x / alpha,
                    (x - alpha) / (1 - alpha),
                ),
            ),
            UpdateLaw(
                y,
                where(
                    x < alpha,
                    lambda_a * y,
                    1 - lambda_b + lambda_b * y,
                ),
            ),
        ]


class BakerMapSystem(BaseSystem):
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import BakerMapSystem
        >>> BakerMapSystem(lambda_a=0.2).prepare().system_.interactions[0].lambda_a
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.y = y
        self.alpha = alpha
        self.lambda_a = lambda_a
        self.lambda_b = lambda_b

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the map and its laws."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[Element('map', values={'x': self.x, 'y': self.y})],
            interactions=[
                BakerMapInteraction(
                    'baker',
                    target='map',
                    alpha=self.alpha,
                    lambda_a=self.lambda_a,
                    lambda_b=self.lambda_b,
                ),
            ],
        )


class TowelMapInteraction(Interaction):
    """The laws of rössler's towel map.

    Args:
        label:
            The label of the interaction.

        target:
            The element whose states `x`, `y`, and `z` the map moves.

        a:
            The growth rate of the first logistic coordinate.

        b:
            The growth rate of the third logistic coordinate.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import TowelMapInteraction
        >>> TowelMapInteraction('towel', target='map').a
        3.8
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'map',
        a: float = 3.8,
        b: float = 3.78,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.a = a
        self.b = b

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the next value of every state as the laws.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        x, y, z = context.target.x, context.target.y, context.target.z
        a, b = context.a, context.b
        self.laws_: list[BaseLaw] = [
            UpdateLaw(x, a * x * (1 - x) - 0.05 * (y + 0.35) * (1 - 2 * z)),
            UpdateLaw(y, 0.1 * ((y + 0.35) * (1 - 2 * z) - 1) * (1 - 1.9 * x)),
            UpdateLaw(z, b * z * (1 - z) + 0.2 * y),
        ]


class TowelMapSystem(BaseSystem):
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import TowelMapSystem
        >>> TowelMapSystem(a=3.7).prepare().system_.interactions[0].a
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.y = y
        self.z = z
        self.a = a
        self.b = b

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the map and its laws."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[Element('map', values={'x': self.x, 'y': self.y, 'z': self.z})],
            interactions=[TowelMapInteraction('towel', target='map', a=self.a, b=self.b)],
        )


class IntermittencyMapInteraction(Interaction):
    """The laws of the Pomeau-Manneville map `x' = x + x^z (mod 1)`.

    Args:
        label:
            The label of the interaction.

        target:
            The element whose state `x` the map moves.

        z:
            The exponent of the tangency at 0, above 1.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import IntermittencyMapInteraction
        >>> IntermittencyMapInteraction('intermittency', target='map').z
        2.0
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'map',
        z: float = 2.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.z = z

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the next value of every state as the laws.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        x = context.target.x
        z = context.z
        self.laws_: list[BaseLaw] = [
            UpdateLaw(x, mod(x + x**z, 1)),
        ]


class IntermittencyMapSystem(BaseSystem):
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import IntermittencyMapSystem
        >>> IntermittencyMapSystem(z=1.5).prepare().system_.interactions[0].z
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.z = z

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the map and its laws."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[Element('map', values={'x': self.x})],
            interactions=[IntermittencyMapInteraction('intermittency', target='map', z=self.z)],
        )


class NewtonRaphsonMapInteraction(Interaction):
    """The laws of newton's method for `z³ = 1` in the complex plane, in real coordinates.

    Args:
        label:
            The label of the interaction.

        target:
            The element whose states `x` and `y` the map moves.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import NewtonRaphsonMapInteraction
        >>> NewtonRaphsonMapInteraction('newton_raphson', target='map').target
        'map'
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'map',
    ) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the next value of every state as the laws.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        x, y = context.target.x, context.target.y
        self.laws_: list[BaseLaw] = [
            UpdateLaw(x, 2 * x / 3 + (x**2 - y**2) / (3 * (x**2 + y**2) ** 2)),
            UpdateLaw(y, 2 * y / 3 - 2 * x * y / (3 * (x**2 + y**2) ** 2)),
        ]


class NewtonRaphsonMapSystem(BaseSystem):
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import NewtonRaphsonMapSystem
        >>> NewtonRaphsonMapSystem(x=1.5).prepare().system_.elements[0].values['x']
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.y = y

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the map and its laws."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[Element('map', values={'x': self.x, 'y': self.y})],
            interactions=[NewtonRaphsonMapInteraction('newton_raphson', target='map')],
        )


class CatMapInteraction(Interaction):
    """The laws of arnold's cat map `(x, y) → (2x + y, x + y) (mod 1)`.

    Args:
        label:
            The label of the interaction.

        target:
            The element whose states `x` and `y` the map moves.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import CatMapInteraction
        >>> CatMapInteraction('cat', target='map').target
        'map'
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'map',
    ) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the next value of every state as the laws.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        x, y = context.target.x, context.target.y
        self.laws_: list[BaseLaw] = [
            UpdateLaw(x, mod(2 * x + y, 1)),
            UpdateLaw(y, mod(x + y, 1)),
        ]


class CatMapSystem(BaseSystem):
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import CatMapSystem
        >>> CatMapSystem(x=0.5).prepare().system_.elements[0].values['x']
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.y = y

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the map and its laws."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[Element('map', values={'x': self.x, 'y': self.y})],
            interactions=[CatMapInteraction('cat', target='map')],
        )


class LoziMapInteraction(Interaction):
    """The laws of the Lozi map `x' = 1 + y - a |x|`, `y' = b x`.

    Args:
        label:
            The label of the interaction.

        target:
            The element whose states `x` and `y` the map moves.

        a:
            The slope of the fold.

        b:
            The contraction.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import LoziMapInteraction
        >>> LoziMapInteraction('lozi', target='map').a
        1.7
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'map',
        a: float = 1.7,
        b: float = 0.5,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.a = a
        self.b = b

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the next value of every state as the laws.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        x, y = context.target.x, context.target.y
        a, b = context.a, context.b
        self.laws_: list[BaseLaw] = [
            UpdateLaw(x, 1 + y - a * abs_symbolic(x)),
            UpdateLaw(y, b * x),
        ]


class LoziMapSystem(BaseSystem):
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import LoziMapSystem
        >>> LoziMapSystem(a=1.5).prepare().system_.interactions[0].a
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.y = y
        self.a = a
        self.b = b

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the map and its laws."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[Element('map', values={'x': self.x, 'y': self.y})],
            interactions=[LoziMapInteraction('lozi', target='map', a=self.a, b=self.b)],
        )


class ConservativeHenonMapInteraction(Interaction):
    """The laws of hénon's area-preserving quadratic map.

    Args:
        label:
            The label of the interaction.

        target:
            The element whose states `x` and `y` the map moves.

        nu:
            The rotation number of the linear part, in turns.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import ConservativeHenonMapInteraction
        >>> ConservativeHenonMapInteraction('conservative_henon', target='map').nu
        0.2114
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'map',
        nu: float = 0.2114,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.nu = nu

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the next value of every state as the laws.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        x, y = context.target.x, context.target.y
        nu = context.nu
        self.laws_: list[BaseLaw] = [
            UpdateLaw(x, cos(2 * pi * nu) * x + sin(2 * pi * nu) * (y + x**2)),
            UpdateLaw(y, -sin(2 * pi * nu) * x + cos(2 * pi * nu) * (y + x**2)),
        ]


class ConservativeHenonMapSystem(BaseSystem):
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import ConservativeHenonMapSystem
        >>> ConservativeHenonMapSystem(nu=0.25).prepare().system_.interactions[0].nu
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.y = y
        self.nu = nu

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the map and its laws."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[Element('map', values={'x': self.x, 'y': self.y})],
            interactions=[ConservativeHenonMapInteraction('conservative_henon', target='map', nu=self.nu)],
        )


class FermiUlamMapInteraction(Interaction):
    """The laws of the Fermi-Ulam map of a ball bouncing between two walls.

    Args:
        label:
            The label of the interaction.

        target:
            The element whose states `u` and `theta` the map moves.

        epsilon:
            The amplitude of the speed the oscillating wall gives the ball.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import FermiUlamMapInteraction
        >>> FermiUlamMapInteraction('fermi_ulam', target='map').epsilon
        0.1
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'map',
        epsilon: float = 0.1,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.epsilon = epsilon

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the next value of every state as the laws.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        u, theta = context.target.u, context.target.theta
        epsilon = context.epsilon
        self.laws_: list[BaseLaw] = [
            UpdateLaw(u, abs_symbolic(u + epsilon * sin(theta))),
            UpdateLaw(
                theta,
                mod(
                    theta + 2 * pi / abs_symbolic(u + epsilon * sin(theta)),
                    2 * pi,
                ),
            ),
        ]


class FermiUlamMapSystem(BaseSystem):
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import FermiUlamMapSystem
        >>> FermiUlamMapSystem(epsilon=0.2).prepare().system_.interactions[0].epsilon
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.u = u
        self.theta = theta
        self.epsilon = epsilon

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the map and its laws."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[Element('map', values={'u': self.u, 'theta': self.theta})],
            interactions=[FermiUlamMapInteraction('fermi_ulam', target='map', epsilon=self.epsilon)],
        )


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import CDNMapSystem
        >>> CDNMapSystem(beta=-0.3).prepare().system_.elements[0].values['beta']
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label=label, solver=solver, backend=backend)
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

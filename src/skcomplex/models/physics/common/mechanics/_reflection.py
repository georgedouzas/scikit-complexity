"""Define reflections, crossings of a wall that reverse the normal velocity of a moving point."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import SR, Expression
from sage.modules.free_module_element import FreeModuleElement
from sage.sets.real_set import RealSet

from .....core.interactions import CrossingInteraction
from .....core.laws import BaseLaw, CrossingLaw
from .....core.laws.utils import where
from .....errors import ValidationError
from .....specs import ParameterSpec, RoleSpec


class ReflectionInteraction(CrossingInteraction):
    """Reflection of a moving point off a wall `g(x) = 0`, with a coefficient of restitution `e`.

    When the guard `g` crosses zero, the velocity `v` of the point becomes `v - (1 + e) (v . n) n / |n|**2`, where `n`
    is the gradient of `g` with respect to the position. The normal velocity is reversed and scaled by `e`, and the
    tangential velocity is kept, so `e = 1` is a specular, elastic reflection. The position and the velocity are given
    by the names of their states, scalar or vector, whose components pair up in order. A `region` restricts the wall to
    the part of the surface where it holds, and the point crosses the rest of the surface freely.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the element that reflects.

        guard:
            The function `guard(namespace)` of the wall, an expression of the position that is zero on the wall.

        positions:
            The names of the states that hold the position.

        velocities:
            The names of the states that hold the velocity, the first derivatives of the position.

        direction:
            `1` for crossings from negative to positive, `-1` for the opposite, and `0` for both.

        restitution:
            The coefficient of restitution, the ratio of the normal speeds after and before the reflection.

        region:
            The function `region(namespace)` of the condition that holds on the part of the surface that is a wall, or
            `None` for a wall on the whole surface.

    Raises:
        ValidationError:
            The positions and the velocities have different numbers of names, or a function is not callable.

    Examples:
        >>> from skcomplex.core.elements import PointElement
        >>> from skcomplex.core.spaces import ContinuousSpace
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.models.physics.common.mechanics import ReflectionInteraction
        >>> wall = ReflectionInteraction('wall', target='ball', guard=lambda s: s.target.x[0] - 1, direction=1)
        >>> ball = PointElement('ball', x=[0.0, 0.0], x_d1=[1.0, 1.0])
        >>> system = System(elements=[ball], interactions=[wall], space=ContinuousSpace(n_dim=2))
        >>> trajectory = Simulation(system, dt=0.5).prepare().run(until=2.0).trajectory_
        >>> trajectory.values('ball.x')[-1].tolist()
        [0.0, 2.0]
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        guard: Callable[[SimpleNamespace], object],
        positions: Sequence[str] = ('x',),
        velocities: Sequence[str] = ('x_d1',),
        direction: int = 0,
        restitution: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 1.0,
        region: Callable[[SimpleNamespace], object] | None = None,
    ) -> None:
        super().__init__(label, target=target, guard=guard, direction=direction)
        self.positions = positions
        self.velocities = velocities
        self.restitution = restitution
        self.region = region

    def _get_components(self, member: SimpleNamespace, names: Sequence[str]) -> list[Expression]:
        """Return the components of the named states of a member, every component of a vector state."""
        components: list[Expression] = []
        for name in names:
            value = getattr(member, name)
            components += list(value) if isinstance(value, FreeModuleElement) else [SR(value)]
        return components

    def _check(self) -> None:
        """Raise an error for positions and velocities of different numbers of names, or a function not callable.

        Raises:
            ValidationError:
                The positions and the velocities have different numbers of names, or a function is not callable.
        """
        super()._check()
        if self.region is not None and not callable(self.region):
            error_msg = f'The region of reflection {self.label!r} must be a function of the namespace.'
            raise ValidationError(error_msg)
        if len(self.positions) != len(self.velocities):
            error_msg = (
                f'Reflection {self.label!r} pairs the positions {self.positions!r} with the velocities '
                f'{self.velocities!r}. Give one velocity per position.'
            )
            raise ValidationError(error_msg)

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the crossing law of the wall, which reflects the velocity about the normal of the wall.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with `namespace.target`.

        Returns:
            laws:
                The crossing law.
        """
        guard = SR(self.guard(namespace))
        positions = self._get_components(namespace.target, self.positions)
        velocities = self._get_components(namespace.target, self.velocities)
        normal = [guard.diff(position) for position in positions]
        speed = sum((v * n for v, n in zip(velocities, normal, strict=True)), SR(0))
        scale = (1 + namespace.restitution) * speed / sum((n**2 for n in normal), SR(0))
        reset = {v: v - scale * n for v, n in zip(velocities, normal, strict=True)}
        if self.region is not None:
            region = self.region(namespace)
            reset = {v: where(region, value, v) for v, value in reset.items()}
        return [CrossingLaw(guard, reset, self.direction, self.terminal, label=self.label)]

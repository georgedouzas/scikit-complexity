"""Define geodesic interactions.

They describe the free fall of a test particle along the geodesics of a pseudo-Riemannian space.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated, ClassVar

from sage.all__sagemath_symbolics import SR, vector
from sage.modules.free_module_element import FreeModuleElement

from ...errors import SpaceError, ValidationError
from ...specs import RoleSpec
from ...types import ElementKind
from ..base import BasePart
from ..laws import BaseLaw, RateLaw
from ..laws.utils import derivative
from ..spaces import BaseSpace, PseudoRiemannianSpace
from ._interaction import Interaction


class GeodesicInteraction(Interaction):
    """Geodesic equation `x''^mu = -Γ^mu_(alpha beta)(x) x'^alpha x'^beta` of a point.

    The point is a point of a pseudo-Riemannian space. The state is a vector of order 2 with one component per
    coordinate of the space, the position `x^mu` of a test particle, and its companion holds the tangent `x'^mu`. The
    time of the system is the affine parameter of the geodesic, the proper time of a timelike geodesic when the tangent
    starts with `g_(mu nu) x'^mu x'^nu = -1`. A tangent that starts null, with `g_(mu nu) x'^mu x'^nu = 0`, follows a
    light ray. Either norm is conserved along the geodesic, so it measures the error of a run. The Christoffel symbols
    are those the space derives from its metric.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the element or group whose state is the position.

        state:
            The name of the vector state of order 2 that is the position.

        space:
            The pseudo-Riemannian space whose geodesics the point follows, usually the space of the system too.

    Attributes:
        space_:
            The prepared space, whose Christoffel symbols the rate reads.

    Raises:
        ValidationError:
            The state is not a name.

        SpaceError:
            The space is not a pseudo-Riemannian space, raised by `prepare`.

    Examples:
        >>> from sage.all__sagemath_symbolics import SR, matrix
        >>> from skcomplex.core.components.elements import PointElement
        >>> from skcomplex.core.components.interactions import GeodesicInteraction
        >>> from skcomplex.core.components.spaces import PseudoRiemannianSpace
        >>> from skcomplex.core.components.systems import CompositeSystem
        >>> from skcomplex.core.engine.runtime import simulate
        >>> r, phi = SR.var('r phi')
        >>> plane = PseudoRiemannianSpace(matrix(SR, [[1, 0], [0, r**2]]), (r, phi))
        >>> line = GeodesicInteraction('line', target='particle', space=plane)
        >>> particle = PointElement('particle', x=[1.0, 0.0], x_d1=[0.0, 1.0])
        >>> system = CompositeSystem(elements=[particle], interactions=[line], space=plane)
        >>> trajectory = simulate(system, until=1.0, dt=0.001)
        >>> [round(float(value), 3) for value in trajectory.values('particle.x')[-1]]
        [1.414, 0.785]
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec(elements=(ElementKind.ANY,), states=('state',))],
        state: str = 'x',
        space: PseudoRiemannianSpace | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.space = space

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Set `space_`, the prepared pseudo-Riemannian space.

        It first raises an error for a state that is not a name.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            SpaceError:
                The space is not a pseudo-Riemannian space.
            ValidationError:
                The type declares states, or the state is not a name.
        """
        super()._prepare(context)
        if not isinstance(self.state, str) or not self.state.isidentifier():
            error_msg = (
                f'{type(self).__name__} {self.label!r} moves the state {self.state!r}. Give the name of a state.'
            )
            raise ValidationError(error_msg)
        if not isinstance(self.space, PseudoRiemannianSpace):
            error_msg = (
                f'{type(self).__name__} {self.label!r} moves a point along the geodesics of {self.space!r}. Give a '
                '`PseudoRiemannianSpace`.'
            )
            raise SpaceError(error_msg)
        self.space_ = self.space.prepare()

    def _get_laws(self, namespace: SimpleNamespace) -> list[BasePart]:
        """Return the rate law of the second derivative of the position.

        It is minus the connection on the tangent.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with the namespace of the target.

        Returns:
            laws:
                The rate law of the position.

        Raises:
            ValidationError:
                The position has not one component per coordinate of the space.
        """
        position = getattr(self._get_member(namespace, 'target'), self.state)
        components = list(position) if isinstance(position, FreeModuleElement) else [position]
        coordinates = self.space_.coordinates_
        if len(components) != len(coordinates):
            error_msg = (
                f'The position {self.state!r} of {self.label!r} has {len(components)} components for the '
                f'{len(coordinates)} coordinates of the space. Give a vector state of the dimension of the space.'
            )
            raise ValidationError(error_msg)
        at = dict(zip(coordinates, components, strict=True))
        tangent = [derivative(component, 1) for component in components]
        indices = range(len(components))
        rate = [
            -sum(
                (
                    SR(self.space_.christoffel_[mu][alpha][beta]).subs(at) * tangent[alpha] * tangent[beta]
                    for alpha in indices
                    for beta in indices
                    if not SR(self.space_.christoffel_[mu][alpha][beta]).is_zero()
                ),
                SR(0),
            )
            for mu in indices
        ]
        return [RateLaw(position, vector(SR, rate) if isinstance(position, FreeModuleElement) else rate[0])]

"""Define collisions, the binary collision integral of a density over the class axes of a phase space."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace
from typing import Annotated

from ...errors import ValidationError
from ...specs import RoleSpec
from ..base import BasePart
from ..laws import CollisionLaw
from ._interaction import Interaction


class CollisionInteraction(Interaction):
    """Binary collisions of a density over the continuous class axes of a phase space.

    The rate of the density at every position is the collision integral of a cross-section over two interval class
    axes, evaluated by the fast spectral method on a padded grid. The density is conserved exactly, and the momentum and
    the energy within the spectral accuracy of the method.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the field on a phase space whose density collides.

        state:
            The name of the state of the density.

        cross_section:
            The function `B(speed, cosine)` of arrays of the relative speed and the cosine of the deflection angle.

        factor:
            A function of the namespace that returns a factor of the collision frequency, such as a function of the
            density moment, or `None` for none.

    Raises:
        ValidationError:
            The cross-section or the factor is not callable, or the state is not a name.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.elements import DensityElement, FieldElement
        >>> from skcomplex.core.interactions import CollisionInteraction
        >>> from skcomplex.core.spaces import LatticeSpace, PhaseSpace
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.engine.solvers import RK4Solver
        >>> from skcomplex.specs import AxisSpec
        >>> axes = [AxisSpec('vx', 8, -6.0, 6.0), AxisSpec('vy', 8, -6.0, 6.0)]
        >>> space = PhaseSpace(LatticeSpace((1,)), axes).check().prepare()
        >>> vx, vy = np.meshgrid(space.coordinates_['vx'], space.coordinates_['vy'], indexing='ij')
        >>> gas = FieldElement('gas', element=DensityElement, space=space, values={'f': np.exp(-vx**2 - vy**2)[None]})
        >>> collide = CollisionInteraction('c', target='gas', cross_section=lambda speed, cosine: 1 / (2 * np.pi))
        >>> system = System(elements=[gas], interactions=[collide])
        >>> trajectory = Simulation(system, solver=RK4Solver(dt=0.1)).check().prepare().run(until=0.2).trajectory_
        >>> values = trajectory.values('gas.f')
        >>> abs(float(values[-1].sum() / values[0].sum()) - 1.0) < 1e-12
        True
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        state: str = 'f',
        cross_section: Callable[..., object],
        factor: Callable[[SimpleNamespace], object] | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.cross_section = cross_section
        self.factor = factor

    def _check(self) -> None:
        """Raise an error for a state that is not a name or a function that is not callable.

        Raises:
            ValidationError:
                The state is not a name, the cross-section is not callable, or the factor is not callable.
        """
        super()._check()
        if not isinstance(self.state, str) or not self.state.isidentifier():
            error_msg = f'{type(self).__name__} {self.label!r} collides the state {self.state!r}, which is no name.'
            raise ValidationError(error_msg)
        if not callable(self.cross_section):
            error_msg = f'The cross-section of {self.label!r} is {self.cross_section!r}. Give a function of two arrays.'
            raise ValidationError(error_msg)
        if self.factor is not None and not callable(self.factor):
            error_msg = f'The factor of {self.label!r} is {self.factor!r}. Give a function of the namespace.'
            raise ValidationError(error_msg)

    def _get_laws(self, namespace: SimpleNamespace) -> list[BasePart]:
        """Return the collision law of the density.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with the namespace of the field.

        Returns:
            laws:
                The collision law.
        """
        state = getattr(self._get_member(namespace, 'target'), self.state)
        factor = None if self.factor is None else self.factor(namespace)
        return [CollisionLaw(state, self.cross_section, factor, label=self.label)]

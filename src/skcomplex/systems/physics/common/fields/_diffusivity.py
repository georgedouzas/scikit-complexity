"""Define diffusion by a diffusivity.

The diffusivity spreads a state of a field as the heat kernel.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace

from .....core.components.interactions import RateInteraction
from .....core.components.laws.utils import laplacian
from .....core.errors import ValidationError


class DiffusivityInteraction(RateInteraction):
    """Diffusion `du/dt = D laplacian(u)` of a state `u` of a field by a diffusivity `D`.

    Physics names the coefficient a diffusivity, as the thermal diffusivity of heat conduction or the kinematic
    viscosity of the vorticity of a fluid. The diffusivity is uniform in space, a number or a function of the namespace
    that returns an expression of parameters and time. The rate is the Laplacian of the field, which its discretization
    evaluates, so the total of the state on a periodic field stays constant and a point mass spreads as the heat kernel.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the field whose state diffuses.

        state:
            The name of the state of the field that diffuses.

        coefficient:
            A function of the namespace that returns the diffusivity, or a number.

    Raises:
        ValidationError:
            The state is not a name, or the coefficient is neither callable nor a number.

    Examples:
        >>> from typing import Annotated
        >>> import numpy as np
        >>> from skcomplex.core.components.elements import Element, FieldElement
        >>> from skcomplex.core.components.spaces import LatticeSpace
        >>> from skcomplex.core.components.systems import CompositeSystem
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.core.specs import StateSpec
        >>> from skcomplex.systems.physics.common.fields import DiffusivityInteraction
        >>> class CellElement(Element):
        ...     def __init__(self, label, u: Annotated[float, StateSpec()] = 0.0):
        ...         super().__init__(label)
        ...         self.u = u
        >>> space = LatticeSpace((40,), spacing=0.1, boundary='periodic')
        >>> field = FieldElement('rod', element=CellElement, space=space, values={'u': np.eye(40)[20]})
        >>> heat = DiffusivityInteraction('heat', target='rod', state='u', coefficient=0.1)
        >>> trajectory = simulate(CompositeSystem(elements=[field], interactions=[heat]), until=1.0, dt=0.01)
        >>> round(float(trajectory.get_values('rod.u')[-1].sum()), 6)
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: str,
        state: str = 'u',
        coefficient: Callable[[SimpleNamespace], object] | float = 1.0,
    ) -> None:
        super().__init__(label, target=target, state=state)
        self.coefficient = coefficient

    def _check_functions(self) -> None:
        """Raise an error for a state that is not a name, or a coefficient that is not a function or a number."""
        super()._check_functions()
        if not (callable(self.coefficient) or isinstance(self.coefficient, int | float)):
            error_msg = f'The coefficient of {self.label!r} is {self.coefficient!r}. Give a function or a number.'
            raise ValidationError(error_msg)

    def _get_rate(self, namespace: SimpleNamespace) -> object:
        """Return the rate `D laplacian(u)` of the state.

        Args:
            namespace:
                The namespace of the symbols the rate can use, with the namespace of the field.

        Returns:
            rate:
                The rate of the state.
        """
        state = getattr(self._get_member(namespace, 'target'), self.state)
        return self._evaluate(self.coefficient, namespace) * laplacian(state)

"""Define diffusion by a diffusivity.

The diffusivity spreads a state of a field as the heat kernel.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace

from .....core.components.interactions import Interaction
from .....core.components.laws import RateLaw
from .....core.components.laws.utils import laplacian
from .....core.components.spaces import BaseSpace
from .....core.errors import ValidationError
from .....core.types import Role


class DiffusivityInteraction(Interaction):
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
        >>> import numpy as np
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.components.spaces import LatticeSpace
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.physics.common.fields import DiffusivityInteraction
        >>> space = LatticeSpace((40,), spacing=0.1, boundary='periodic')
        >>> rod = Element('rod', values={'u': np.eye(40)[20]}, space=space)
        >>> heat = DiffusivityInteraction('heat', target='rod', state='u', coefficient=0.1)
        >>> trajectory = System(elements=[rod], interactions=[heat]).prepare().simulate(until=1.0, dt=0.01)
        >>> round(float(trajectory.get_values('rod.u')[-1].sum()), 6)
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        state: str = 'u',
        coefficient: Callable[[SimpleNamespace], object] | float = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.coefficient = coefficient

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rate `D laplacian(u)` of the state.

        Args:
            context:
                The space the interaction inherits from its system, or the namespace of the field, the time `t`, and
                the parameters.

        Raises:
            ValidationError:
                The state is not a name, or the coefficient is neither callable nor a number.
        """
        if not isinstance(self.state, str) or not self.state.isidentifier():
            error_msg = f'{type(self).__name__} {self.label!r} diffuses the state {self.state!r}, which is no name.'
            raise ValidationError(error_msg)
        if not (callable(self.coefficient) or isinstance(self.coefficient, int | float)):
            error_msg = f'The coefficient of {self.label!r} is {self.coefficient!r}. Give a function or a number.'
            raise ValidationError(error_msg)
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            u = getattr(context.target, self.state)
            coefficient = self.coefficient(context) if callable(self.coefficient) else self.coefficient
            self.laws_ = [RateLaw(u, coefficient * laplacian(u))]

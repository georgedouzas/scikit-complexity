"""Define diffusion, the conservation law whose flux is minus a coefficient times the gradient of the state."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace
from typing import Annotated, ClassVar

from ...errors import ValidationError
from ...specs import RoleSpec, Tags
from ...types import ElementKind
from ..base import BasePart
from ..laws import BaseLaw, FluxLaw, RateLaw
from ..laws.utils import laplacian
from ._base import RATE_DOMAINS
from ._conservation import ConservationInteraction


class DiffusionInteraction(ConservationInteraction):
    """Diffusion `du/dt = D laplacian(u)` of a state `u` of a field, the conservation law of the flux `-D gradient(u)`.

    The coefficient `D` is uniform in space, a number or a function of the namespace that returns an expression of
    parameters and time. The interaction lowers to the Laplacian of the field, which its discretization evaluates, so
    the total of the state on a periodic field stays constant and a point mass spreads as the heat kernel.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the field whose state diffuses.

        state:
            The name of the state of the field that diffuses.

        coefficient:
            A function of the namespace that returns the diffusion coefficient, or a number.

    Raises:
        ValidationError:
            The state is not a name, or the coefficient is neither callable nor a number.

    Examples:
        >>> from typing import Annotated
        >>> import numpy as np
        >>> from skcomplex.core.components.elements import Element, FieldElement
        >>> from skcomplex.core.components.interactions import DiffusionInteraction
        >>> from skcomplex.core.components.spaces import LatticeSpace
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.core.engine.runtime import Simulation
        >>> from skcomplex.core.specs import StateSpec
        >>> class CellElement(Element):
        ...     def __init__(self, label, u: Annotated[float, StateSpec()] = 0.0):
        ...         super().__init__(label)
        ...         self.u = u
        >>> space = LatticeSpace((40,), spacing=0.1, boundary='periodic')
        >>> field = FieldElement('rod', element=CellElement, space=space, values={'u': np.eye(40)[20]})
        >>> heat = DiffusionInteraction('heat', target='rod', state='u', coefficient=0.1)
        >>> trajectory = Simulation(System(elements=[field], interactions=[heat]), dt=0.01).prepare()
        >>> round(float(trajectory.run(until=1.0).trajectory_.values('rod.u')[-1].sum()), 6)
        1.0
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (FluxLaw, RateLaw)

    def _get_tags(self) -> Tags:
        """Return the tags of the rate of diffusion."""
        return self._get_variant_tags((RateLaw,))

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec(elements=(ElementKind.ANY,), states=('state',), domains=RATE_DOMAINS)],
        state: str = 'u',
        coefficient: Callable[[SimpleNamespace], object] | float = 1.0,
    ) -> None:
        super().__init__(label, target=target, state=state)
        self.coefficient = coefficient

    def _check(self) -> None:
        """Raise an error for a state that is not a name, or a coefficient that is neither callable nor a number.

        Raises:
            ValidationError:
                The type declares states, the state is not a name, or the coefficient is neither callable nor a number.
        """
        super()._check()
        if not (callable(self.coefficient) or isinstance(self.coefficient, int | float)):
            error_msg = f'The coefficient of {self.label!r} is {self.coefficient!r}. Give a function or a number.'
            raise ValidationError(error_msg)

    def _get_transport_laws(self, namespace: SimpleNamespace) -> list[BasePart]:
        """Return the rate `D laplacian(u)` of the state, the divergence of the diffusive flux.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with the namespace of the field.

        Returns:
            laws:
                The rate law of the state.
        """
        coefficient = self.coefficient(namespace) if callable(self.coefficient) else self.coefficient
        state = getattr(self._get_member(namespace, 'target'), self.state)
        return [RateLaw(state, coefficient * laplacian(state))]

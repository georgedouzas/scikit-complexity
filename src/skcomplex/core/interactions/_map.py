"""Define maps, which set the next value of a state after one discrete time step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated

from ...errors import ValidationError
from ...specs import RoleSpec
from ..laws import BaseLaw, UpdateLaw
from ._interaction import Interaction

FORMS = ('replace', 'increment')


class MapInteraction(Interaction):
    """Map of a state of discrete time, whose next value is a function of the namespace, or the state plus it.

    In the form `'replace'` the next value of the state is `f(namespace)`, and in the form `'increment'` it is the state
    plus `f(namespace)`. A bound group updates every member at once, from the states at the start of the step. The
    function reads `namespace.target`, the states and parameters of the target, `namespace.sources`, a tuple of the
    namespaces of the sources, and the time `namespace.t`, and may hold draws such as `normal('noise')`.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the element or group whose state the map sets.

        state:
            The name of the state of the target.

        function:
            The function `f(namespace)` that returns a symbolic expression, or a vector of them for a vector state.

        sources:
            The paths of the elements, or members of groups such as `cells[0]`, that the function reads.

        form:
            `'replace'` or `'increment'`.

    Raises:
        ValidationError:
            The form is unknown, or the function is not callable.

    Examples:
        >>> from skcomplex.core.elements import Element
        >>> from skcomplex.core.interactions import MapInteraction
        >>> from skcomplex.core.spaces import DiscreteTime
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.specs import StateSpec
        >>> from typing import Annotated
        >>> class ValueElement(Element):
        ...     def __init__(self, label, x: Annotated[float, StateSpec()] = 0.5):
        ...         super().__init__(label)
        ...         self.x = x
        >>> logistic = MapInteraction(
        ...     'map', target='value', state='x', function=lambda s: 3 * s.target.x * (1 - s.target.x)
        ... )
        >>> system = System(elements=[ValueElement('value')], interactions=[logistic], time=DiscreteTime())
        >>> Simulation(system).check().prepare().run(until=2).trajectory_.values('value.x').tolist()
        [0.5, 0.75, 0.5625]
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        state: str,
        function: Callable[[SimpleNamespace], object],
        sources: Annotated[Sequence[str], RoleSpec(variadic=True)] = (),
        form: str = 'replace',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.function = function
        self.sources = sources
        self.form = form

    def _check(self) -> None:
        """Raise an error for an unknown form, or a function that cannot be called.

        Raises:
            ValidationError:
                The form is unknown, or the function is not callable.
        """
        super()._check()
        if self.form not in FORMS:
            error_msg = f'The form of map {self.label!r} is {self.form!r}. Use one of {", ".join(FORMS)}.'
            raise ValidationError(error_msg)
        if not callable(self.function):
            error_msg = f'The function of map {self.label!r} must be callable, but it is {self.function!r}.'
            raise ValidationError(error_msg)

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the update of the state to its next value.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with `namespace.target` and `namespace.sources`.

        Returns:
            laws:
                The update law of the state.
        """
        value = getattr(namespace.target, self.state)
        mapped = self.function(namespace)
        return [UpdateLaw(value, value + mapped if self.form == 'increment' else mapped)]

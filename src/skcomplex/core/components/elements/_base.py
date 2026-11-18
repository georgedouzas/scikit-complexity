"""Define the abstract element, the component of a system that holds its states."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import replace
from types import SimpleNamespace
from typing import ClassVar, Self

from ...errors import ValidationError
from ...specs import StateSpec, VisualSpec
from .._tags import Tags
from ..base import BaseComponent
from ..spaces import BaseSpace


class BaseElement(BaseComponent[BaseSpace, SimpleNamespace]):
    """An abstract element, the component of a system that holds its states.

    A state is a value that changes as the system runs, such as the size of a population or the position of a particle.
    An element names its states and parameters, may give laws of its own, and is drawn as its class attribute `visual`
    says, such as `VisualSpec('line')`, unless a visualisation observer draws it otherwise. A subclass takes
    each state and parameter as an argument of its `__init__`, annotated with `StateSpec` or `ParameterSpec` inside
    `Annotated`, and stores it under the same name, as a scikit-learn estimator does. It returns its laws from
    `_get_laws`, or an empty list when it has none.

    Unlike `BaseComponent`, it holds states, and the interactions and laws of a system read and change them through it.
    Choose it when a new kind of element stores its members in a way of its own, as `GroupElement` does. Choose
    `Element` for one entity with named states.

    It appears in every field through its subclasses, such as a population in ecology, a chemical species in a reactor,
    a bank account in economics, and a pendulum in mechanics.

    Args:
        label:
            The name of the element, unique among the components of its system, such as `'rabbits'`.

    Attributes:
        default_visual_:
            How the element is drawn by default, the `visual` of its class.

        laws_:
            The laws of the element for the namespace of its last run.

    Examples:
        A population with one state and one parameter. The constructor stores its arguments as given.

        >>> from typing import Annotated
        >>> from skcomplex.core.components.elements import BaseElement
        >>> from skcomplex.core.specs import ParameterSpec, StateSpec, VisualSpec
        >>> class PopulationElement(BaseElement):
        ...     visual = VisualSpec('line')
        ...
        ...     def __init__(
        ...         self, label, n: Annotated[float, StateSpec()] = 1.0, r: Annotated[float, ParameterSpec()] = 0.5
        ...     ):
        ...         super().__init__(label)
        ...         self.n = n
        ...         self.r = r
        ...
        ...     def _get_laws(self, namespace):
        ...         return []
        >>> PopulationElement('rabbits', n=40).get_params(deep=False)
        {'label': 'rabbits', 'n': 40, 'r': 0.5}
    """

    visual: ClassVar[VisualSpec] = VisualSpec()
    laws_: Sequence[BaseComponent]

    def _run(self, inputs: SimpleNamespace | None) -> None:
        """Set `laws_`, the laws of the element for a namespace of symbols.

        Args:
            inputs:
                The namespace of the symbols the laws can use.
        """
        self.laws_ = self._get_laws_of(inputs)

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Set `default_visual_`, the drawing of the element its class gives.

        Args:
            context:
                The space the element inherits from its system, which an element does not read.
        """
        super()._prepare(context)
        self.default_visual_ = self._get_visual()

    def _get_visual(self) -> VisualSpec:
        """Return how the element is drawn, the `visual` of its class by default."""
        return self.visual

    def _get_compiler(self) -> object:
        """Return the compiler a value of the element names, or `None` for the general one.

        A value names a compiler through its own `_get_compiler`, as the Hilbert space of a quantum state names the
        compiler of operator dynamics, also when it is given in `values`. A subclass may name one itself, as a word
        that rewrites does.
        """
        given = getattr(self, 'values', None)
        for value in [*self.get_params(deep=False).values(), *(given.values() if isinstance(given, dict) else ())]:
            named = getattr(value, '_get_compiler', None)
            if callable(named) and not isinstance(value, type):
                return named()
        return None

    def prepare(self, context: BaseSpace | None = None) -> Self:
        """Prepare the element on its space, checking that it holds values and no laws.

        Laws come from interactions, so an element whose `_prepare` sets `laws_` is rejected. Gravity on a ball, for
        example, is an interaction whose `_prepare` sets `self.laws_ = [RateLaw(s.target.x.diff(s.t, 2), -s.g)]`.

        Args:
            context:
                The space the element inherits from its system, or `None`.

        Returns:
            self:
                The prepared element.

        Raises:
            ValidationError:
                The `_prepare` of the element sets `laws_`.
        """
        if hasattr(self, 'laws_'):
            del self.laws_
        super().prepare(context)
        if hasattr(self, 'laws_'):
            error_msg = (
                f'Element {self.label!r} sets laws_ in _prepare, but an element holds values only. Write the law in '
                f'an interaction instead, such as a Gravity interaction whose _prepare sets '
                f'self.laws_ = [RateLaw(s.target.x.diff(s.t, 2), -s.g)].'
            )
            raise ValidationError(error_msg)
        return self

    def _get_domain_tags(self) -> frozenset[str]:
        """Return the domain tags of the states of the element."""
        domains = {spec._get_domain_tag() for spec in self.get_states().values()}
        return frozenset(domain for domain in domains if domain is not None)

    def _get_tags(self) -> Tags:
        """Return the tags of the element.

        The tags are the domains of its states and the laws its class lists.
        """
        return replace(self._get_law_tags(self._get_declared_laws()), domains=self._get_domain_tags())

    def get_states(self) -> dict[str, StateSpec]:
        """Return the states of the element with their specs.

        The states are those its class declares and those it takes as data.

        Returns:
            states:
                The spec of every state, by name.

        Examples:
            >>> from typing import Annotated
            >>> from skcomplex.core.components.elements import Element
            >>> from skcomplex.core.specs import StateSpec
            >>> class TankElement(Element):
            ...     def __init__(self, label, level: Annotated[float, StateSpec()] = 1.0):
            ...         super().__init__(label)
            ...         self.level = level
            >>> list(TankElement('tank').get_states())
            ['level']
        """
        return {name: spec for name, spec in self._get_component_specs().items() if isinstance(spec, StateSpec)}

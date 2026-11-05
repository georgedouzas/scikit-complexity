"""Define declared elements, which take their states and parameters as data at construction."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import inspect
from collections.abc import Mapping
from dataclasses import replace
from typing import Self

from ...errors import ValidationError
from ...specs import DeclaredVariableSpec, VisualSpec
from ._element import Element


class DeclaredElement(Element):
    """Element whose states and parameters are data, for a model that knows them only when it builds the element.

    The compiler treats a declared variable as it treats an annotated argument of the constructor: the spec gives the
    declaration and the value gives the default. A law reads a declared variable from its namespace, and `get_value`
    reads its value from the element. A subclass that derives its variables from other arguments overrides
    `_declare_variables`, and `variables` then holds the variables that replace the derived ones. The mappings the
    constructor takes are read as immutable, so `set_params` changes a variable.

    Args:
        label:
            The label of the element, unique among the parts of its system.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        variables:
            The declared states and parameters by name, or `None` for none.

    Attributes:
        default_visual_:
            The visual of the element when it sets none.

    Raises:
        ValidationError:
            A name is not an identifier or is an argument of the constructor, raised by `prepare`.

    Examples:
        >>> from skcomplex.core.elements import DeclaredElement
        >>> from skcomplex.specs import DeclaredVariableSpec, ParameterSpec, StateSpec
        >>> variables = {'x': DeclaredVariableSpec(1.0, StateSpec()), 'r': DeclaredVariableSpec(0.5, ParameterSpec())}
        >>> element = DeclaredElement('cell', variables=variables)
        >>> element.get_value('x'), element.set_params(r=0.25).get_value('r')
        (1.0, 0.25)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        variables: Mapping[str, DeclaredVariableSpec] | None = None,
    ) -> None:
        super().__init__(label, visual=visual)
        self.variables = variables

    def _declare_variables(self) -> Mapping[str, DeclaredVariableSpec]:
        """Return the variables the element declares from its arguments, by name."""
        return self.variables or {}

    def _get_declared(self) -> Mapping[str, DeclaredVariableSpec]:
        """Return the declared variables, read once per set of arguments, since compiling reads them often."""
        arguments = list(self.get_params(deep=False).values())
        known = getattr(self, '_declared', None)
        if known is not None and all(new is old for new, old in zip(arguments, known[0], strict=True)):
            return known[1]
        declared = self._declare_variables()
        self._declared = (arguments, declared)
        return declared

    def _check(self) -> None:
        """Raise an error for a name that a law cannot read or that an argument of the constructor takes.

        Raises:
            ValidationError:
                A name is not an identifier or is an argument of the constructor.
        """
        super()._check()
        arguments = inspect.signature(type(self)).parameters
        for name in self._get_declared():
            if not name.isidentifier() or name in arguments:
                error_msg = f'Element {self.label!r} declares {name!r}. Use an identifier that no argument takes.'
                raise ValidationError(error_msg)

    def get_value(self, name: str) -> object:
        """Return the value of a declared state or parameter.

        Args:
            name:
                The name of the variable.

        Returns:
            value:
                The value.

        Examples:
            >>> from skcomplex.core.elements import DeclaredElement
            >>> from skcomplex.specs import DeclaredVariableSpec, StateSpec
            >>> DeclaredElement('cell', variables={'x': DeclaredVariableSpec(1.0, StateSpec())}).get_value('x')
            1.0
        """
        return self._get_declared()[name].value

    def set_params(self, **params: object) -> Self:
        """Set arguments of the constructor and values of declared variables, which keep their specs.

        Args:
            **params:
                The new values by name.

        Returns:
            element:
                The element.

        Examples:
            >>> from skcomplex.core.elements import DeclaredElement
            >>> from skcomplex.specs import DeclaredVariableSpec, StateSpec
            >>> element = DeclaredElement('cell', variables={'x': DeclaredVariableSpec(1.0, StateSpec())})
            >>> element.set_params(label='other', x=2.0).get_value('x')
            2.0
        """
        arguments = self.get_params(deep=False)
        super().set_params(**{name: value for name, value in params.items() if name in arguments})
        declared = self._get_declared()
        rest = {name: value for name, value in params.items() if name not in arguments}
        changed = {name: replace(declared[name], value=value) for name, value in rest.items() if name in declared}
        if changed:
            super().set_params(variables={**(self.variables or {}), **changed})
        return super().set_params(**{name: value for name, value in rest.items() if name not in declared})

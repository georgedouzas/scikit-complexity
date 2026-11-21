"""Define the element whose states and parameters are given as data."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import inspect
from collections.abc import Mapping
from dataclasses import replace
from typing import Self

from ...components._declared_variable_spec import _DeclaredVariableSpec
from ...components._parameter_spec import _ParameterSpec
from ...components._state_spec import _StateSpec
from ...components.elements import Element, VisualSpec
from ...components.spaces import BaseSpace
from ...errors import ValidationError


class _DeclaredElement(Element):
    """An element whose states and parameters are given as data.

    The element takes a mapping from each name to its initial value and its spec, so a system can build it without a
    class of its own. A law reads a declared variable by name, as it reads a state that a subclass declares, and
    `get_value` returns its value. A state of higher order or with lags also has companion states, such as `x_d1` for
    its first derivative or `x_lag1` for its value one step back, which `get_value` reads and `set_params` changes. A
    subclass that derives its variables from other arguments overrides `_declare_variables`.

    Unlike `Element`, it needs no subclass, since it takes its states and parameters as data. Choose it when a system
    learns its variables only when it builds them, or when a class for one model would only repeat its arguments.

    It appears in many fields. The number of people infected in an epidemic, the moles of a chemical species in a
    reactor, the money in a bank account, and the tokens in a place of a Petri net are all declared states.

    Args:
        label:
            The name of the element, unique among the components of its system, such as `'infected'` or `'account'`.

        variables:
            The initial value and the spec of each state and parameter, by name, or `None` for none. For example,
            `{'x': _DeclaredVariableSpec(100.0, _StateSpec())}` declares a state `x` that starts at 100, in the units of
            the system, such as people or moles.

        drawing:
            How the element is drawn in place of the `visual` of its class, such as the drawing of the class of a
            plain element the system declares it from, or `None` for the `visual` of its class.

        view:
            The class of the views of the element's agents in place of the one of its class, such as the view of the
            class of a plain element the system declares it from, or `None` for the one of its class.

        defaults:
            The values a member that a rule creates takes when the rule gives none, by name, such as the defaults of
            the arguments of the class of a plain element the system declares it from, or `None` for none.

        origin:
            The class of a plain element the system declares it from, such as `PopulationElement`, whose name names
            the element in drawings, or `None` for its own class.

    Attributes:
        default_visual_:
            How the element is drawn by default, the `visual` of its class.

    Raises:
        ValidationError:
            A name is not an identifier or is an argument of the constructor, raised by `prepare`.
    """

    def __init__(
        self,
        label: str,
        *,
        variables: Mapping[str, _DeclaredVariableSpec] | None = None,
        drawing: VisualSpec | None = None,
        view: type | None = None,
        defaults: Mapping[str, object] | None = None,
        origin: type | None = None,
    ) -> None:
        super().__init__(label)
        self.variables = variables
        self.drawing = drawing
        self.view = view
        self.defaults = defaults
        self.origin = origin

    def _get_visual(self) -> VisualSpec:
        """Return how the element is drawn, its `drawing` or the `visual` of its class."""
        return self.drawing if self.drawing is not None else super()._get_visual()

    def _declare_variables(self) -> Mapping[str, _DeclaredVariableSpec]:
        """Return the variables the element declares from its arguments, by name."""
        return self.variables or {}

    def _get_declared(self) -> Mapping[str, _DeclaredVariableSpec]:
        """Return the declared variables.

        They are read once per set of arguments, since compiling reads them often.
        """
        arguments = list(self.get_params(deep=False).values())
        known = getattr(self, '_declared', None)
        if known is not None and all(new is old for new, old in zip(arguments, known[0], strict=True)):
            return known[1]
        declared = self._declare_variables()
        self._declared = (arguments, declared)
        return declared

    def _get_companions(self) -> dict[str, tuple[str, int]]:
        """Return the state and the initial value position of every companion, by name.

        A declared state of order `k` has the companions `<name>_d1` to `<name>_d<k-1>`, and one with `m` lags has the
        companions `<name>_lag1` to `<name>_lag<m>`.
        """
        companions: dict[str, tuple[str, int]] = {}
        for name, item in self._get_declared().items():
            spec = item.spec
            if not isinstance(spec, _StateSpec):
                continue
            derivatives = int(spec.order) - 1 if float(spec.order).is_integer() else 0
            companions.update({f'{name}_d{index}': (name, index - 1) for index in range(1, derivatives + 1)})
            companions.update({f'{name}_lag{index}': (name, index - 1) for index in range(1, spec.lags + 1)})
        return companions

    def _get_initial(self, name: str) -> list[object]:
        """Return the initial values of the companions of a declared state.

        A missing initial value of a derivative is zero and one of a lag is the value of the state, as the compiler
        reads them.
        """
        item = self._get_declared()[name]
        spec = item.spec
        if not isinstance(spec, _StateSpec):
            return []
        derivatives = int(spec.order) - 1 if float(spec.order).is_integer() else 0
        missing = 0.0 if derivatives else item.value
        initial = list(spec.initial)
        return initial + [missing] * (max(derivatives, spec.lags) - len(initial))

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Raise an error for a name a law cannot read or a constructor argument takes.

        Args:
            context:
                The space the element inherits from its system, which an element does not read.

        Raises:
            ValidationError:
                A name is not an identifier or is an argument of the constructor, or a variable is declared with a
                spec of a role, which only interactions take.
        """
        super()._prepare(context)
        arguments = inspect.signature(type(self)).parameters
        for name, item in self._get_declared().items():
            if not name.isidentifier() or name in arguments:
                error_msg = f'Element {self.label!r} declares {name!r}. Use an identifier that no argument takes.'
                raise ValidationError(error_msg)
            if not isinstance(item.spec, _StateSpec | _ParameterSpec):
                error_msg = (
                    f'Element {self.label!r} declares {name!r} with {item.spec!r}. An element declares a variable with '
                    'a state or a parameter, and a role belongs to an interaction.'
                )
                raise ValidationError(error_msg)

    def get_value(self, name: str) -> object:
        """Return the value of a declared state or parameter, or of a companion state.

        Args:
            name:
                The name of the variable, such as `'x'`, or of a companion state, such as `'x_d1'`.

        Returns:
            value:
                The value of the variable.
        """
        companions = self._get_companions()
        if name in companions:
            state, position = companions[name]
            return self._get_initial(state)[position]
        return self._get_declared()[name].value

    def set_params(self, **params: object) -> Self:
        """Set arguments of the constructor and values of declared variables.

        A declared variable keeps its spec and takes the new value. A companion state such as `x_d1` takes the new value
        as the initial value that the spec of its state holds.

        Args:
            **params:
                The new values by name, such as `x=2.0`.

        Returns:
            element:
                The element itself.
        """
        arguments = self.get_params(deep=False)
        super().set_params(**{name: value for name, value in params.items() if name in arguments})
        declared = self._get_declared()
        companions = self._get_companions()
        rest = {name: value for name, value in params.items() if name not in arguments}
        changed = {name: replace(declared[name], value=value) for name, value in rest.items() if name in declared}
        initial: dict[str, list[object]] = {}
        for name, value in rest.items():
            if name in companions:
                state, position = companions[name]
                initial.setdefault(state, self._get_initial(state))[position] = value
        for state, values in initial.items():
            item = changed.get(state, declared[state])
            if isinstance(item.spec, _StateSpec):
                changed[state] = replace(item, spec=replace(item.spec, initial=tuple(values)))
        if changed:
            super().set_params(variables={**(self.variables or {}), **changed})
        unknown = {name: value for name, value in rest.items() if name not in declared and name not in companions}
        return super().set_params(**unknown)

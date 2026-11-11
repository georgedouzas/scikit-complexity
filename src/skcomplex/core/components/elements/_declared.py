"""Define the element whose states and parameters are given as data."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import inspect
from collections.abc import Mapping
from dataclasses import replace
from typing import Self

from ...errors import ValidationError
from ...specs import DeclaredVariableSpec, StateSpec, VisualSpec
from ..spaces import BaseSpace
from ._element import Element


class DeclaredElement(Element):
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

        visual:
            How the element is drawn, such as `VisualSpec('line')`, or `None` for `default_visual_`.

        variables:
            The initial value and the spec of each state and parameter, by name, or `None` for none. For example,
            `{'x': DeclaredVariableSpec(100.0, StateSpec())}` declares a state `x` that starts at 100, in the units of
            the system, such as people or moles.

    Attributes:
        default_visual_:
            The visual of the element when it sets none.

    Raises:
        ValidationError:
            A name is not an identifier or is an argument of the constructor, raised by `prepare`.

    Examples:
        An epidemic compartment and a bank account are the same kind of element. Infected people recover at a rate of
        0.1 a day.

        >>> from skcomplex.core.components.elements import DeclaredElement
        >>> from skcomplex.core.components.interactions import RateInteraction
        >>> from skcomplex.core.components.systems import CompositeSystem
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.core.specs import DeclaredVariableSpec, StateSpec
        >>> infected = DeclaredElement('infected', variables={'x': DeclaredVariableSpec(100.0, StateSpec())})
        >>> recovery = RateInteraction('recovery', target='infected', state='x', rate=lambda s: -0.1 * s.target.x)
        >>> epidemic = CompositeSystem(elements=[infected], interactions=[recovery])
        >>> simulate(epidemic, until=10, dt=5.0).get_values('infected.x').round(1).tolist()
        [100.0, 60.7, 36.8]

        Interest at 5 percent a year grows the account continuously.

        >>> account = DeclaredElement('account', variables={'x': DeclaredVariableSpec(250.0, StateSpec())})
        >>> interest = RateInteraction('interest', target='account', state='x', rate=lambda s: 0.05 * s.target.x)
        >>> savings = CompositeSystem(elements=[account], interactions=[interest])
        >>> simulate(savings, until=1, dt=1.0).get_values('account.x').round(2).tolist()
        [250.0, 262.82]

        A state of second order, such as the position of a mass, has a companion state for its velocity.

        >>> mass = DeclaredElement('mass', variables={'x': DeclaredVariableSpec(1.0, StateSpec(order=2))})
        >>> mass.get_value('x'), mass.get_value('x_d1')
        (1.0, 0.0)
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
            if not isinstance(spec, StateSpec):
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
        if not isinstance(spec, StateSpec):
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
                A name is not an identifier or is an argument of the constructor.
        """
        super()._prepare(context)
        arguments = inspect.signature(type(self)).parameters
        for name in self._get_declared():
            if not name.isidentifier() or name in arguments:
                error_msg = f'Element {self.label!r} declares {name!r}. Use an identifier that no argument takes.'
                raise ValidationError(error_msg)

    def get_value(self, name: str) -> object:
        """Return the value of a declared state or parameter, or of a companion state.

        Args:
            name:
                The name of the variable, such as `'x'`, or of a companion state, such as `'x_d1'`.

        Returns:
            value:
                The value of the variable.

        Examples:
            >>> from skcomplex.core.components.elements import DeclaredElement
            >>> from skcomplex.core.specs import DeclaredVariableSpec, StateSpec
            >>> DeclaredElement('cell', variables={'x': DeclaredVariableSpec(1.0, StateSpec())}).get_value('x')
            1.0
            >>> element = DeclaredElement('mass', variables={'x': DeclaredVariableSpec(1.0, StateSpec(order=2))})
            >>> element.get_value('x_d1')
            0.0
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

        Examples:
            >>> from skcomplex.core.components.elements import DeclaredElement
            >>> from skcomplex.core.specs import DeclaredVariableSpec, StateSpec
            >>> element = DeclaredElement('cell', variables={'x': DeclaredVariableSpec(1.0, StateSpec())})
            >>> element.set_params(label='other', x=2.0).get_value('x')
            2.0
            >>> element = DeclaredElement('mass', variables={'x': DeclaredVariableSpec(1.0, StateSpec(order=2))})
            >>> element.set_params(x_d1=0.5).get_value('x_d1')
            0.5
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
            if isinstance(item.spec, StateSpec):
                changed[state] = replace(item, spec=replace(item.spec, initial=tuple(values)))
        if changed:
            super().set_params(variables={**(self.variables or {}), **changed})
        unknown = {name: value for name, value in rest.items() if name not in declared and name not in companions}
        return super().set_params(**unknown)

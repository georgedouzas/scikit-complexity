"""Define the element whose states and parameters its subclass declares."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import inspect
import numbers
from collections.abc import Mapping
from dataclasses import replace
from typing import Annotated, Any, ClassVar, Self, get_args, get_origin

import numpy as np
from sage.all__sagemath_symbolics import CC, RR, ZZ
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from ...errors import ValidationError
from .._matrix_group_spec import MatrixGroupSpec
from .._parameter_spec import _ParameterSpec
from .._power_series_spec import PowerSeriesSpec
from .._state_spec import _StateSpec
from .._subshift_spec import SubshiftSpec
from .._tags import Tags
from ..spaces import BaseSpace, NetworkSpace
from ._base import BaseElement

RESERVED = ('label', 'size', 'space', 'values', 'on')
BOOLEAN = FiniteEnumeratedSet([0, 1])
LINKS = ('edges', 'hyperedges')
RESERVED_MEANINGS = {
    'size': 'number of members',
    'space': 'space of the members',
    'values': 'mapping of values by name',
    'on': 'part of a network space that holds one member each',
}


class Element(BaseElement):
    """An element that holds named values of one thing or of a group of members.

    It holds its values by name, such as the height and mass of a ball in `Element('ball', values={'x': 10.0, 'm': 1.0})`,
    and the laws of the interactions decide which values change. With `size` it holds a group of members with values of
    their own, such as 1000 molecules of a gas, and with `space` one member per site, such as the cells of a grid. A
    subclass may take its values as arguments of `__init__` instead, such as a population class with a growth rate. Members
    whose values change by agent rules, Python functions run for each member at each step, act as agents, and their values
    may be any Python object. It holds a prey population, a firm of an economy, and the charge of a capacitor.

    Read more in the user guide, under Elements.

    Args:
        label:
            The name of the element, unique among the parts of its system, such as `'rabbits'`.

        values:
            Values by name, such as `{'x': 10.0, 'm': 1.0}`. A derivative or a past value that a law reads starts from the
            value named with primes or a past index, such as `{'x': 10.0, "x'": 3.0}` for a ball thrown upwards or
            `{'y': 1.0, 'y[-1]': 0.5}` for an economy. A group of members takes one value per member as an array.

        size:
            The number of members, each with its own values, such as `1000` molecules of a gas, or `None` for one.

        space:
            The space of the element, with one member per site, such as a lattice of cells, or `None` for none.

        on:
            The part of a network space that holds one member each: `'edges'` for every link, `'hyperedges'` for every
            hyperedge, the name of a layer for every link of that layer, such as `'immune'`, or `None` for every site.

    Attributes:
        default_visual_:
            How the element is drawn by default, the `visual` of its class.

    See Also:
        BaseElement: The base of every element.
        Interaction: States the laws that change the values of elements.

    Examples:
        >>> from skcomplex.core.components.elements import Element
        >>> ball = Element('ball', values={'x': 10.0, 'm': 1.0})
        >>> ball.get_params()['x']
        10.0
    """

    law_classes: ClassVar[tuple[type, ...]] = ()

    def __init__(
        self,
        label: str,
        *,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
        on: str | None = None,
    ) -> None:
        super().__init__(label)
        self.values = values
        self.size = size
        self.space = space
        self.on: object = on

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the parameters of the element, with each entry of `values` when deep.

        An entry of `values` whose name is a Python identifier is listed under its own name, so a system lists it as
        `<element>__<name>` and `set_params(ball__x=2.0)`, grid searches, and sweeps reach it. A primed or past value,
        such as `"x'"` or `'x[-1]'`, is no identifier and stays inside `values`. An entry that has the name of an
        argument is not listed twice.

        Args:
            deep:
                Whether to add the parameters of nested estimators and the entries of `values`.

        Returns:
            params:
                The parameters by name.

        Examples:
            >>> from skcomplex.core.components.elements import Element
            >>> params = Element('ball', values={'x': 10.0, "x'": 3.0}).get_params()
            >>> params['x'], "x'" in params
            (10.0, False)
        """
        params = super().get_params(deep=deep)
        if deep and isinstance(self.values, Mapping):
            entries = self._get_value_entries()
            params.update({name: value for name, value in entries.items() if name not in params})
        return params

    def set_params(self, **params: object) -> Self:
        """Set parameters of the element and entries of `values` by name.

        An entry of `values` takes its new value in a new mapping, so the mapping the user gave never changes, as
        `clone` of scikit-learn leaves the arguments of an estimator unchanged. A new `values` is set first, so the
        output of `get_params` is accepted as it is.

        Args:
            **params:
                The new values, by the name of a parameter or of an entry of `values`, such as `x=2.0`.

        Returns:
            self:
                The element.

        Examples:
            >>> from skcomplex.core.components.elements import Element
            >>> values = {'x': 10.0}
            >>> Element('ball', values=values).set_params(x=2.0).values, values
            ({'x': 2.0}, {'x': 10.0})
        """
        arguments = super().get_params(deep=True)
        if 'values' in params:
            super().set_params(values=params['values'])
        entries = self._get_value_entries()
        changed = {name: value for name, value in params.items() if name in entries and name not in arguments}
        rest = {name: value for name, value in params.items() if name not in changed and name != 'values'}
        super().set_params(**rest)
        if changed:
            self.values = {**(self.values or {}), **changed}
        return self

    def _get_value_entries(self) -> dict[str, object]:
        """Return the entries of `values` whose names are Python identifiers."""
        values = self.values if isinstance(self.values, Mapping) else {}
        return {name: value for name, value in values.items() if name.isidentifier()}

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Set `default_visual_`, checking the reserved names.

        Args:
            context:
                The space the element inherits from its system, which an element with `on` and no space of its own
                sits on.

        Raises:
            ValidationError:
                `size` is not a number of members, `space` is not a space, or `values` is not a mapping, as when a
                subclass stores a quantity of its own under a reserved name.
        """
        super()._prepare(context)
        size = self.size
        checks = {
            'size': size is None or (isinstance(size, numbers.Integral) and not isinstance(size, bool) and size >= 1),
            'space': self.space is None or isinstance(self.space, BaseSpace),
            'values': self.values is None or isinstance(self.values, Mapping),
        }
        for name, valid in checks.items():
            if not valid:
                error_msg = (
                    f'Element {self.label!r} has {name} = {getattr(self, name)!r}, but {name} is the '
                    f'{RESERVED_MEANINGS[name]}. The names {", ".join(RESERVED)} are reserved, so rename the value, '
                    f'such as half_side for the half side of a table.'
                )
                raise ValidationError(error_msg)
        self._check_on(self.space if self.space is not None else context)

    def _get_tags(self) -> Tags:
        """Return the tags of the element, with the tags of its space when it has one.

        A field on a phase space thus asks for a solver that supports phase spaces, as a group on it does.
        """
        tags = super()._get_tags()
        if not isinstance(self.space, BaseSpace):
            return tags
        return replace(tags, space=self.space.get_tags().space)

    def _check_on(self, space: BaseSpace | None) -> None:
        """Raise an error for an `on` that names no part of a network space.

        A value named `on` that the element declares as its own state or parameter names no place. The space is the
        element's own, or else that of its system.
        """
        if isinstance(type(self)._get_specs().get('on'), _StateSpec | _ParameterSpec):
            return
        on = getattr(self, 'on', None)
        if on is None or (isinstance(space, NetworkSpace) and (on in LINKS or on in space.prepare().graphs_)):
            return
        error_msg = (
            f'Element {self.label!r} has on = {on!r}, but on is the part of a network space that holds one member '
            "each, 'edges', 'hyperedges', or the name of a layer for its edges, such as "
            "Element('links', values={'sign': 1}, space=network, on='edges'). The names "
            f'{", ".join(RESERVED)} are reserved, so rename a value of the element named on, such as closed for a '
            'switch.'
        )
        raise ValidationError(error_msg)

    @staticmethod
    def _get_domain_of(annotation: object, value: object) -> object:
        """Return the domain of a value from its annotation or its Python type.

        A value without an annotation, such as one given through `values`, takes the domain of its type, so `3` is an
        integer and `0.5 + 1j` is complex. The annotation `object` says nothing of the kind, so a value annotated so,
        such as a point that may be one number or an array of complex numbers, also takes the domain of its type, and
        so does a value of one of several types, such as `float | Sequence[float]`. A subshift, such as
        `Annotated[int, SubshiftSpec(LatticeNeighborhood(((0,), (1,))), ((1, 1),))]`, is the domain of the cells of a
        lattice whose configurations hold no forbidden word.
        """
        if get_origin(annotation) is Annotated:
            for item in get_args(annotation)[1:]:
                if isinstance(item, set | frozenset):
                    return FiniteEnumeratedSet(sorted(item, key=str))
                allowed = hasattr(item, '__contains__') and not isinstance(item, str | MatrixGroupSpec)
                if allowed or isinstance(item, SubshiftSpec | PowerSeriesSpec):
                    return item
            annotation = get_args(annotation)[0]
        annotated = isinstance(annotation, type) and annotation not in {inspect.Parameter.empty, object}
        kind = annotation if annotated else type(value)
        if kind is bool:
            return BOOLEAN
        if kind is int:
            return ZZ
        if kind is complex or (isinstance(kind, type) and issubclass(kind, np.complexfloating)):
            return CC
        return RR

    def _get_domain_tags(self) -> frozenset[str]:
        """Return the domain tags of the states of the element.

        An element that declares no states has the domains of all its numeric values, since the laws of the system
        decide which of them are states, such as `boolean` for `opinion: Annotated[int, {0, 1}] = 0`.
        """
        if self._get_component_specs():
            return super()._get_domain_tags()
        params = self.get_params(deep=False)
        values = {name: params.get(name) for name in self._get_hints()}
        values |= {name: value for name, value in (self.values or {}).items() if name.isidentifier()}
        hints = self._get_hints()
        domains = {
            _StateSpec(domain=self._get_domain_of(hints.get(name, inspect.Parameter.empty), value))._get_domain_tag()
            for name, value in values.items()
            if isinstance(value, numbers.Number | list | np.ndarray)
        }
        return frozenset(domain for domain in domains if domain is not None)

    def _get_hints(self) -> dict[str, object]:
        """Return the type hint of every value argument by name, in their order.

        The hints are the annotations of the arguments of the class, such as `Annotated[float, Quantity('energy')]`,
        from which the kind, the allowed values, the quantity, and the period of each value are read. A generic
        subclass that takes its values through `values` gives their hints here instead, and the arguments it leaves
        out, such as the allowed values themselves, are no values.
        """
        parameters = inspect.signature(type(self)).parameters
        return {name: parameter.annotation for name, parameter in parameters.items() if name not in RESERVED}

    @classmethod
    def _get_view(cls) -> type | None:
        """Return the class of the views of the element's agents.

        The result is `None` for the engine's plain view. An agent element names a subclass of the engine's `AgentView`
        that annotates its states and parameters, such as `class WordView(AgentView): count: float`, so its rules read
        them with checked types.
        """
        return None

"""Define the element whose states and parameters its subclass declares."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import inspect
import numbers
from collections.abc import Mapping
from dataclasses import replace
from typing import Annotated, ClassVar, get_args, get_origin

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
    """An element whose states and parameters are declared by its own subclass.

    A subclass takes each state and parameter as a plain argument of its `__init__`, and the laws decide which values
    change. It lists the classes of the laws that interactions state for it in `law_classes`, so that `get_tags` can
    name their kinds before a run.

    An element whose laws are agent rules, Python functions that change one member at each step, is an agent. Under
    discrete time each member runs its rules with its states as attributes, and these states may hold any value, not
    only numbers. A member is removed only when its states tagged with a quantity reach zero. Its states and parameters
    cannot take the names that the agent view or Mesa reads, such as `location`, `alive`, `position`, or `system`.

    Unlike `BaseElement`, it declares its states by its own subclass, and it can act as an agent. Choose it
    when a kind of entity deserves a class of its own, such as a population with a growth law or a capacitor with two
    pins, and give `values` to a plain `Element` when the values are known only as data.

    It appears in many fields. A prey population in ecology, a capacitor in an electric circuit, a firm in an economy,
    and a pendulum in mechanics are each a subclass of it.

    Args:
        label:
            The name of the element, unique among the components of its system, such as `'rabbits'`.

        values:
            Values by name, such as `{'x': 10.0, 'm': 1.0}`, in place of a subclass that takes them as arguments. A
            derivative or a past value that a law reads starts from the value named with primes or a past index, such
            as `{'x': 10.0, "x'": 3.0}` for a ball thrown upwards or `{'y': 1.0, 'y[-1]': 0.5}` for an economy. An
            element with many members takes one starting value per member as an array, such as the velocities of a
            gas.

        size:
            The number of members, each with its own values, such as `1000` molecules of a gas, or `None` for one.

        space:
            The space of the element, with one member per site, such as a lattice of cells, or `None` for none.

        on:
            The part of a network space that holds one member each: `'edges'` for every link, such as the signed
            links of a network, `'hyperedges'` for every hyperedge, the name of a layer for every link of that layer,
            such as `'immune'` for the links between immune cells, or `None` for every site.

    Attributes:
        default_visual_:
            How the element is drawn by default, the `visual` of its class.

    Examples:
        A population and a bank deposit both grow at a rate proportional to their size, so one class serves both.

        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.components.interactions import Interaction
        >>> from skcomplex.core.components.laws import RateLaw
        >>> from skcomplex.core.systems import System
        >>> class Population(Element):
        ...     def __init__(self, label, x=1.0, r=0.5):
        ...         super().__init__(label)
        ...         self.x = x
        ...         self.r = r
        >>> def grow(s):
        ...     return [RateLaw(s.target.x, s.target.r * s.target.x)]
        >>> rabbits = Population('rabbits', x=40.0, r=0.5)
        >>> deposit = Population('deposit', x=250.0, r=0.05)
        >>> names = ['rabbits', 'deposit']
        >>> growths = [Interaction(f'{name}_growth', roles={'target': name}, laws=grow) for name in names]
        >>> system = System(elements=[rabbits, deposit], interactions=growths)
        >>> trajectory = system.prepare().simulate(until=1, dt=1.0)
        >>> trajectory.get_values('rabbits.x').round(1).tolist(), trajectory.get_values('deposit.x').round(2).tolist()
        ([40.0, 65.9], [250.0, 262.82])
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

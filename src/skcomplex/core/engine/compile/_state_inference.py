"""Infer the states and parameters of plain elements from the laws of their system."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Mapping
from typing import get_args, get_origin

import numpy as np
from sage.all__sagemath_symbolics import CC, RR, SR, ZZ, Expression
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from ...components.base import BaseComponent
from ...components.elements import BaseElement, DeclaredElement, Element, FieldElement, GroupElement
from ...components.laws import (
    BaseLaw,
    ConstraintLaw,
    JumpLaw,
    NoiseLaw,
    RateLaw,
    RuleLaw,
    TopologyRequestSpec,
    UpdateLaw,
)
from ...components.laws.utils import derivative, lag
from ...components.spaces import CellComplexSpace, ContinuousSpace, LatticeSpace, NetworkSpace
from ...components.systems import CompositeSystem
from ...enums import TopologyKind
from ...errors import ValidationError
from ...specs import DeclaredVariableSpec, ParameterSpec, StateSpec
from .._utils import bound_labels
from ._compilation import Compilation

RESERVED = ('label', 'visual', 'size', 'space', 'values')
BOOLEAN = FiniteEnumeratedSet([0, 1])
MEMBERSHIP = frozenset(
    {
        TopologyKind.ADD_NODE,
        TopologyKind.REMOVE_NODE,
        TopologyKind.REPLACE_NODE,
        TopologyKind.CONTRACT_EDGE,
        TopologyKind.DUPLICATE_NODE,
    },
)
STARTING = re.compile(r"(?P<name>[A-Za-z_]\w*)(?:(?P<primes>'+)|\[-(?P<lag>[1-9]\d*)\])")


class _StateInference:
    """Inference of the states and parameters of elements that declare neither.

    A value of such an element that some law changes is a state, of the order of the highest derivative its laws give,
    and any other value is a parameter. The kind of a value comes from its Python type hint or its value, and a
    constraint from the allowed values in `Annotated`. A value named after an axis of a continuous space, such as `x`,
    is the position of its member, and a value of a group whose members the laws add or remove is a state, even when no
    law changes it. An element with `size` or `space` is a group of members, and a value given as an
    array holds one value per member, such as the heights of `1000` molecules, while a single value is shared. On a
    lattice, a network, or a mesh, the members are the sites of a field, so laws can read `laplacian(u)` or a sum over
    neighbours.
    """

    def __init__(self, census: Callable[[CompositeSystem], Compilation]) -> None:
        self.census = census

    @staticmethod
    def _get_values(element: Element) -> dict[str, object]:
        """Return the values of a plain element by name, from its arguments and `values`.

        The starting values of derivatives and past values, such as `"x'"` or `'y[-1]'`, are left out.
        """
        given = {name: value for name, value in (element.values or {}).items() if not STARTING.fullmatch(name)}
        for name in given:
            if name in RESERVED or not name.isidentifier():
                error_msg = (
                    f'Element {element.label!r} cannot take a value named {name!r}. Values need Python names that are '
                    f'not one of {", ".join(RESERVED)}, such as values={{"x": 10.0}}, or the name of a value with '
                    "primes or a past index, such as values={\"x'\": 1.0} or values={'y[-1]': 0.5}."
                )
                raise ValidationError(error_msg)
        arguments = {name: value for name, value in element.get_params(deep=False).items() if name not in RESERVED}
        return arguments | given

    @staticmethod
    def _get_starting(element: Element, values: Mapping[str, object]) -> dict[str, dict[int, object]]:
        """Return the starting values of derivatives and past values, by value and index.

        A derivative named with `k` primes and a past value named with the index `-k` both have the index `k`.

        Raises:
            ValidationError:
                A starting value names no value of the element, or the element has many members.
        """
        starting: dict[str, dict[int, object]] = {}
        for name, value in (element.values or {}).items():
            found = STARTING.fullmatch(name)
            if found is None:
                continue
            base = found.group('name')
            if base not in values or element.size is not None or element.space is not None:
                error_msg = (
                    f'Element {element.label!r} takes the starting value {name!r}, but it has no single value named '
                    f'{base!r}. Give it with the value, such as values={{{base!r}: 0.0, {name!r}: 1.0}}.'
                )
                raise ValidationError(error_msg)
            index = len(found.group('primes')) if found.group('primes') else int(found.group('lag'))
            starting.setdefault(base, {})[index] = value
        return starting

    @staticmethod
    def _get_count(element: Element) -> int | None:
        """Return the number of members of an element, or `None` for one element.

        An element in a space without sites, such as a continuous one, is one member unless its size says otherwise.
        """
        if element.size is not None:
            return int(element.size)
        if element.space is not None and not element.space._has_sites():
            return 1
        if element.space is not None:
            return element.space.prepare()._count_parts('sites', None)
        return None

    @staticmethod
    def _get_axes(element: Element) -> tuple[str, ...]:
        """Return the axes of the continuous space of an element, or none."""
        space = element.space
        return space.prepare().axes_ if isinstance(space, ContinuousSpace) else ()

    @staticmethod
    def _get_member_value(element: Element, name: str, value: object, count: int | None) -> object:
        """Return the value of one member, checking an array holds one per member."""
        array = np.asarray(value, dtype=object)
        if count is None or array.ndim == 0:
            return value
        if array.shape[0] != count:
            error_msg = (
                f'Element {element.label!r} has {count} members but its value {name!r} has {array.shape[0]} entries. '
                f'Give one entry per member, or a single value that every member shares.'
            )
            raise ValidationError(error_msg)
        return array[0]

    @staticmethod
    def _is_plain(component: BaseComponent) -> bool:
        """Return whether an element has values but declares no states or laws."""
        if not isinstance(component, Element) or isinstance(component, DeclaredElement):
            return False
        if type(component)._get_laws is not Element._get_laws or not _StateInference._get_values(component):
            return False
        specs = type(component)._get_specs()
        return not any(isinstance(spec, StateSpec | ParameterSpec) for spec in specs.values())

    @staticmethod
    def _get_domain(annotation: object, value: object) -> object:
        """Return the domain of a value from its annotation or its Python type.

        A value without an annotation, such as one given through `values`, takes the domain of its type, so `3` is an
        integer and `0.5 + 1j` is complex. The annotation `object` says nothing of the kind, so a value annotated so,
        such as a point that may be one number or an array of complex numbers, also takes the domain of its type.
        """
        if get_origin(annotation) is not None and get_args(annotation):
            for item in get_args(annotation)[1:]:
                if isinstance(item, set | frozenset):
                    return FiniteEnumeratedSet(sorted(item, key=str))
                if hasattr(item, '__contains__') and not isinstance(item, str):
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

    def _get_spec(
        self,
        element: Element,
        name: str,
        value: object,
        order: float | None,
        *,
        lags: int = 0,
        starting: Mapping[int, object] | None = None,
        role: str | None = None,
    ) -> DeclaredVariableSpec:
        """Return the declaration of a value as a state or a parameter.

        The starting values of the derivatives or past values a law reads are given by their index, and a missing one
        is zero for a derivative and the value itself for a past value. A state with the role `'position'` is a
        position that walks, particle analyses, and drawings read.

        Raises:
            ValidationError:
                A starting value is given for a parameter, or for a derivative or past value no law reads.
        """
        parameter = inspect.signature(type(element)).parameters.get(name)
        annotation = inspect.Parameter.empty if parameter is None else parameter.annotation
        domain = self._get_domain(annotation, value)
        vector = np.ndim(np.asarray(value, dtype=object)) > 0
        starting = starting or {}
        count = lags or max(int(order or 1) - 1, 0)
        unread = sorted(index for index in starting if index > count)
        if unread and order is not None:
            error_msg = (
                f'Element {element.label!r} takes a starting value of {name!r} at index {unread[0]}, but its laws read '
                f'{count} derivatives or past values of it.'
            )
            raise ValidationError(error_msg)
        if order is None:
            return DeclaredVariableSpec(value, ParameterSpec(domain=domain, vector=vector))
        if order == 0:
            return DeclaredVariableSpec(value, StateSpec(domain=domain, vector=vector, algebraic=True, role=role))
        missing = value if lags else 0.0
        initial = tuple(starting.get(index, missing) for index in range(1, count + 1)) if starting else ()
        steps = 1 if lags else order
        spec = StateSpec(domain=domain, vector=vector, order=steps, lags=lags, initial=initial, role=role)
        return DeclaredVariableSpec(value, spec)

    def _replace(
        self,
        system: CompositeSystem,
        declare: Callable[[Element, str], list[BaseElement]],
        scope: str = '',
    ) -> CompositeSystem:
        """Return a copy of a system tree whose plain elements are declared elements."""
        elements = [
            declared
            for element in system.elements
            for declared in (declare(element, f'{scope}{element.label}') if self._is_plain(element) else [element])
        ]
        systems = [
            self._replace(nested, declare, f'{scope}{nested.label}/') if isinstance(nested, CompositeSystem) else nested
            for nested in system.systems
        ]
        return CompositeSystem(
            system.label,
            elements=elements,
            interactions=list(system.interactions),
            systems=systems,
            space=system.space,
            time=system.time,
        )

    def _has_plain(self, system: CompositeSystem) -> bool:
        """Return whether a system tree holds a plain element."""
        if any(self._is_plain(element) for element in system.elements):
            return True
        return any(isinstance(nested, CompositeSystem) and self._has_plain(nested) for nested in system.systems)

    @staticmethod
    def _get_lags(law: BaseComponent, named: Mapping[str, str]) -> dict[str, int]:
        """Return the number of past values a law reads of every value, by its dotted name."""
        lags: dict[str, int] = {}
        values = law._get_values() if isinstance(law, BaseLaw) else []
        for value in values:
            if not isinstance(value, Expression):
                continue
            for found in value.find(lag(SR.wild(0), SR.wild(1))):
                state, steps = found.operands()
                if str(state) in named and SR(steps).is_integer():
                    name = named[str(state)]
                    lags[name] = max(int(steps), lags.get(name, 0))
        return lags

    @staticmethod
    def _has_derivative(law: BaseLaw) -> bool:
        """Return whether a law reads a derivative of a state."""
        marker = derivative(SR.wild(0), SR.wild(1))
        return any(isinstance(value, Expression) and bool(value.find(marker)) for value in law._get_values())

    @staticmethod
    def _get_derivative_orders(law: BaseLaw, named: Mapping[str, str]) -> dict[str, int]:
        """Return the highest derivative a law reads of every value, by its dotted name.

        An equation such as `L q'' + R q' + q / C == 0` reads the second derivative of `q`, so `q` has order 2.
        """
        orders: dict[str, int] = {}
        for value in law._get_values():
            if not isinstance(value, Expression):
                continue
            for found in value.find(derivative(SR.wild(0), SR.wild(1))):
                state, steps = found.operands()
                if str(state) in named and SR(steps).is_integer():
                    name = named[str(state)]
                    orders[name] = max(int(steps), orders.get(name, 0))
        return orders

    @staticmethod
    def _get_order(name: str, order: float, known: float | None) -> float:
        """Return the higher of two orders of a value, both integers or equal.

        A fractional order matches only itself.

        Raises:
            ValidationError:
                The value has a fractional order and another order.
        """
        if known is None or known == order:
            return order
        if not float(order).is_integer() or not float(known).is_integer():
            fractional, other = (known, order) if float(order).is_integer() else (order, known)
            error_msg = (
                f'The value {name} has a fractional derivative of order {fractional} and a derivative of order '
                f'{other}. Give every rate law of {name} the same target, such as `fractional(x, {fractional})`.'
            )
            raise ValidationError(error_msg)
        return max(order, known)

    @staticmethod
    def _get_orders(compilation: Compilation) -> tuple[dict[str, float], dict[str, int]]:
        """Return the order and the past values of every value the laws use.

        A value that only constraints without derivatives determine has order 0, as the potential of a circuit node
        whose currents must balance, and a value that rate laws give a fractional derivative has its fractional order.
        A value whose derivative of order `k` a constraint reads has order `k` at least, as the charge of
        `L q'' + R q' + q / C == 0`.
        """
        census = Compilation._get_required(compilation.census, 'census')
        layout = Compilation._get_required(compilation.layout, 'layout')
        named = {
            str(block.symbols[variable]): entry.name
            for block in layout.blocks.values()
            for variable, entry in block.entries.items()
        }
        orders: dict[str, float] = {}
        lags: dict[str, int] = {}
        for record in census.records:
            law = record.law
            for name, steps in _StateInference._get_lags(law, named).items():
                lags[name] = max(steps, lags.get(name, 0))
            order = (law.order or 1) if isinstance(law, RateLaw | NoiseLaw) else 1
            names = [entry.name for entry in record.targets]
            if isinstance(law, ConstraintLaw) and law.target is not None and str(law.target) in named:
                names.append(named[str(law.target)])
                order = 1 if _StateInference._has_derivative(law) else 0
            for name in names:
                orders[name] = _StateInference._get_order(name, order, orders.get(name))
            if isinstance(law, ConstraintLaw):
                for name, steps in _StateInference._get_derivative_orders(law, named).items():
                    orders[name] = _StateInference._get_order(name, steps, orders.get(name))
        return orders, lags

    def get_declared(self, system: CompositeSystem) -> CompositeSystem:
        """Return a system tree whose plain elements declare what their laws show.

        Args:
            system:
                The system.

        Returns:
            system:
                The system, or a copy whose plain elements are declared elements.
        """
        if not self._has_plain(system):
            return system

        def probe(element: Element, path: str) -> list[BaseElement]:
            return self._get_declared_elements(element, lambda name: 1, lambda name: 0, probing=True)

        compilation = self.census(self._replace(system, probe))
        orders, lags = self._get_orders(compilation)
        kinds = self._get_topology_kinds(compilation)
        agents = self._get_agent_groups(compilation)
        steps = self._get_steps(compilation)

        def declare(element: Element, path: str) -> list[BaseElement]:
            return self._get_declared_elements(
                element,
                (lambda name: 1) if path in agents else (lambda name: orders.get(f'{path}.{name}')),
                lambda name: lags.get(f'{path}.{name}', 0),
                kinds,
                agents=path in agents,
                steps={
                    name.removeprefix(f'{path}.'): moved.removeprefix(f'{path}.')
                    for name, moved in steps.items()
                    if name.startswith(f'{path}.') and moved.startswith(f'{path}.')
                },
            )

        return self._replace(system, declare)

    @staticmethod
    def _get_steps(compilation: Compilation) -> dict[str, str]:
        """Return what every update law adds to its target, as `v` in `x <- x + v`.

        The dotted name of each such value maps to that of the value it moves, so the step of a position in discrete
        time is its velocity.
        """
        census = Compilation._get_required(compilation.census, 'census')
        layout = Compilation._get_required(compilation.layout, 'layout')
        named = {
            str(block.symbols[variable]): entry.name
            for block in layout.blocks.values()
            for variable, entry in block.entries.items()
        }
        steps: dict[str, str] = {}
        for record in census.records:
            law = record.law
            if not isinstance(law, UpdateLaw) or not isinstance(law.value, Expression) or str(law.target) not in named:
                continue
            step = str((law.value - SR(law.target)).expand())
            if step in named:
                steps[named[step]] = named[str(law.target)]
        return steps

    @staticmethod
    def _get_topology_kinds(compilation: Compilation) -> frozenset[str]:
        """Return the kinds of the changes the laws make to a network."""
        census = Compilation._get_required(compilation.census, 'census')
        kinds: set[str] = set()
        for record in census.records:
            law = record.law
            if isinstance(law, UpdateLaw) and isinstance(law.target, TopologyRequestSpec):
                kinds.add(law.target.kind)
            if isinstance(law, JumpLaw):
                kinds.update(item.kind for item in law.changes or {} if isinstance(item, TopologyRequestSpec))
        return frozenset(kinds)

    @staticmethod
    def _get_agent_groups(compilation: Compilation) -> set[str]:
        """Return the paths of the groups whose members run agent rules."""
        census = Compilation._get_required(compilation.census, 'census')
        nodes = {node.path: node for node in Compilation._get_required(compilation.nodes, 'nodes')}
        groups: set[str] = set()
        for record in census.records:
            node = nodes.get(record.path)
            if not isinstance(record.law, RuleLaw) or record.law.members is None or node is None:
                continue
            groups.update(f'{node.scope}{label}' for label in bound_labels(getattr(node.component, record.law.members)))
        return groups

    def _get_declared_elements(
        self,
        element: Element,
        order: Callable[[str], float | None],
        lags: Callable[[str], int],
        kinds: frozenset[str] = frozenset(),
        probing: bool = False,
        *,
        agents: bool = False,
        steps: Mapping[str, str] | None = None,
    ) -> list[BaseElement]:
        """Return the declared element of a plain element, or a group of declared members.

        On a network whose edges the laws change, the members are a variable group of nodes, with a variable group on
        the edges named after the element, such as `voters_edges`. When the laws also add or remove nodes, every value
        of a node is a state, since it comes and goes with its node, and the group has room for twice its nodes. The
        members of a group that runs agent rules are agents, whose rules may change every value and add or remove
        members, so every value is a state and the group is variable, and a single element that runs them is a group of
        one agent. A value named after an axis of a continuous space is a position, also when no law changes it, and
        the value an update law adds to a position at every step, as `v` in `x <- x + v`, is its velocity. The probe
        that finds the orders reads no starting values of derivatives or past values.
        """
        values = self._get_values(element)
        starting = {} if probing else self._get_starting(element, values)
        count = self._get_count(element) or (1 if agents else None)
        network = bool(kinds) and isinstance(element.space, NetworkSpace)
        growing = network and bool(kinds & MEMBERSHIP)
        axes = self._get_axes(element)
        velocities = {name for name, moved in (steps or {}).items() if moved in axes}
        variables = {}
        for name, value in values.items():
            member = self._get_member_value(element, name, value, count)
            kept = growing or name in axes
            variables[name] = self._get_spec(
                element,
                name,
                member,
                1 if order(name) is None and kept else order(name),
                lags=lags(name),
                starting=starting.get(name),
                role='position' if name in axes else 'velocity' if name in velocities else None,
            )
        if count is None:
            return [DeclaredElement(element.label, variables=variables)]
        arrays = {name: value for name, value in values.items() if np.ndim(np.asarray(value, dtype=object)) > 0}
        values = {'variables': variables, **arrays}
        if network:
            nodes = GroupElement(
                element.label,
                element=DeclaredElement,
                space=element.space,
                on='sites',
                variable=True,
                capacity=None if growing else count,
                values=values,
            )
            edges = GroupElement(
                f'{element.label}_edges',
                element=Element,
                space=element.space,
                on='edges',
                variable=True,
            )
            return [nodes, edges]
        if isinstance(element.space, LatticeSpace | NetworkSpace):
            mesh = isinstance(element.space, CellComplexSpace) and element.space._is_mesh()
            discretization = 'finite-element' if mesh else 'finite-difference'
            field = FieldElement(
                element.label,
                element=DeclaredElement,
                space=element.space,
                values=values,
                discretization=discretization,
            )
            return [field]
        group = GroupElement(
            element.label,
            element=DeclaredElement,
            size=count,
            space=element.space,
            on=None if element.space is None else 'sites',
            values=values,
            variable=agents,
        )
        return [group]

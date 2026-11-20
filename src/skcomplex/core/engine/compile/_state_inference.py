"""Infer the states and parameters of plain elements from the laws of their system."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Collection, Mapping
from typing import Annotated, get_args, get_origin

import numpy as np
from sage.all__sagemath_symbolics import RR, SR, Expression
from sage.sets.real_set import RealSet
from shapely import Polygon

from ...components.base import BaseComponent
from ...components.elements import (
    BaseElement,
    DeclaredElement,
    Element,
    FieldElement,
    GroupElement,
    Member,
    Period,
    Quantity,
    Tensor,
)
from ...components.laws import (
    BaseLaw,
    ConstraintLaw,
    JumpLaw,
    LinearProgramLaw,
    LocalRuleLaw,
    NoiseLaw,
    ObjectiveLaw,
    RateLaw,
    RuleLaw,
    TopologyRequestSpec,
    UpdateLaw,
)
from ...components.laws.utils import delayed, derivative, lag
from ...components.spaces import BaseSpace, CellComplexSpace, ContinuousSpace, LatticeSpace, NetworkSpace
from ...components.systems import CompositeSystem
from ...enums import TopologyKind, VariableKind
from ...errors import SpaceError, ValidationError
from ...specs import (
    BaseSpec,
    DeclaredVariableSpec,
    EntrySpec,
    MatrixGroupSpec,
    ParameterSpec,
    StateSpec,
    VisualSpec,
)
from .._utils import binding_of, bound_labels, declarations_of
from ..problems._bilinear_law import BilinearLaw
from ._compilation import Compilation

RESERVED = ('label', 'visual', 'size', 'space', 'values', 'on')
REGION = 'region'
LINKS = ('edges', 'hyperedges')
MIN_RANK = 2
OTHER = '__other'
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
    neighbours. With `on='edges'` or `on='hyperedges'` the members sit on the links of a network instead, such as the
    signed links of triadic percolation.
    """

    def __init__(self, census: Callable[[CompositeSystem], Compilation]) -> None:
        self.census = census

    @staticmethod
    def _get_values(element: Element) -> dict[str, object]:
        """Return the values of a plain element by name, from its arguments and `values`.

        The values keep the order in which they are given, the arguments of the class first and then the entries of
        `values`, so the states and parameters come out in that order. The starting values of derivatives and past
        values, such as `"x'"` or `'y[-1]'`, are left out, and so are the arguments that hold a spec, such as the
        structure a chain of angles is drawn from, which are settings of the element.
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
        params = element.get_params(deep=False)
        arguments = {
            name: params[name]
            for name in element._get_hints()
            if name in params and name not in RESERVED and not isinstance(params[name], BaseSpec)
        }
        return arguments | given

    @staticmethod
    def _get_starting(
        element: Element,
        values: Mapping[str, object],
        skipped: Collection[str] = (),
    ) -> dict[str, dict[int, object]]:
        """Return the starting values of derivatives and past values, by value and index.

        A derivative named with `k` primes and a past value named with the index `-k` both have the index `k`. An
        element with many members takes one starting value per member as an array, such as the velocities of a gas. The
        names in `skipped`, such as the history of a delayed value, are left out.

        Raises:
            ValidationError:
                A starting value names no value of the element.
        """
        starting: dict[str, dict[int, object]] = {}
        for name, value in (element.values or {}).items():
            found = STARTING.fullmatch(name)
            if found is None or name in skipped:
                continue
            base = found.group('name')
            if base not in values:
                error_msg = (
                    f'Element {element.label!r} takes the starting value {name!r}, but it has no value named '
                    f'{base!r}. Give it with the value, such as values={{{base!r}: 0.0, {name!r}: 1.0}}.'
                )
                raise ValidationError(error_msg)
            index = len(found.group('primes')) if found.group('primes') else int(found.group('lag'))
            starting.setdefault(base, {})[index] = value
        return starting

    @staticmethod
    def _get_histories(element: Element, delays: Collection[str]) -> dict[str, tuple[str, object]]:
        """Return the history of every value the laws read with a delay.

        The history is the value before the start, given as the past value with the index `-1`, a number or a
        function of time, such as `values={'I': 0.01, 'I[-1]': 0.0}` for infected people who were none before an
        outbreak. Each comes with the name it is given under.
        """
        histories = {}
        for name, value in (element.values or {}).items():
            found = STARTING.fullmatch(name)
            if found is not None and found.group('lag') == '1' and found.group('name') in delays:
                histories[found.group('name')] = (name, value)
        return histories

    @staticmethod
    def _get_rule_derivatives(element: Element) -> dict[str, tuple[str, str, object]]:
        """Return the starting derivatives of the values of agents, by engine name.

        The engine name of `"x'"` is `x_d1`. Rules change such a derivative as a value of its own, such as the
        velocity `"x'"` of a boid that its steering turns and its flight adds to `x`, and read it as `agent.x_d1`.
        Each comes with the name it is given under and the value it derives.
        """
        derivatives = {}
        for name, value in (element.values or {}).items():
            found = STARTING.fullmatch(name)
            if found is not None and found.group('primes'):
                base = found.group('name')
                derivatives[f'{base}_d{len(found.group("primes"))}'] = (name, base, value)
        return derivatives

    @staticmethod
    def _get_free_order(element: Element, name: str) -> int | None:
        """Return the order of a value that moves freely, or `None`.

        A value that no law changes but that starts with a derivative, such as a point given `values={"x'": 1.0}`, moves
        at that derivative, as Newton's first law says, so it has the order one above its highest starting derivative,
        whose own derivative is zero.
        """
        primes = [
            len(found.group('primes'))
            for given in (element.values or {})
            if (found := STARTING.fullmatch(given)) is not None
            and found.group('name') == name
            and found.group('primes')
        ]
        return max(primes) + 1 if primes else None

    @staticmethod
    def _get_link(element: Element) -> tuple[str, str | None] | None:
        """Return the part of a network an element sits on and its layer.

        It is `None` for the sites. An element on the edges of a layer, such as
        `Element('immune_links', values={...}, space=network, on='immune')`, sits on the edges of that layer, and
        `on='edges'` on those of the graph of the network.
        """
        if element.on in LINKS:
            return str(element.on), None
        if isinstance(element.space, NetworkSpace) and isinstance(element.on, str):
            return 'edges', element.on
        return None

    @staticmethod
    def _get_count(element: Element) -> int | None:
        """Return the number of members of an element, or `None` for one element.

        An element in a space without sites, such as a continuous one, is one member unless its size says otherwise,
        and an element on the hyperedges of a network without any has no members until the laws add hyperedges.
        """
        if element.size is not None:
            return int(element.size)
        if isinstance(element.space, NetworkSpace) and element.on == 'hyperedges':
            return len(element.space.prepare().hyperedges_)
        link = _StateInference._get_link(element)
        if element.space is not None and link is not None:
            return element.space.prepare()._count_parts(*link)
        if element.space is not None and not element.space._has_sites():
            return 1
        if element.space is not None:
            return element.space.prepare()._count_parts('sites', None)
        return None

    @staticmethod
    def _get_axes(element: Element, agents: bool = False, space: BaseSpace | None = None) -> tuple[str, ...]:
        """Return the axes of the continuous space of an element, or none.

        The space is that of the element, or else the continuous space of its system. The agents of a lattice also have
        the axes of its cells, such as `x` and `y`, and the agents of a network its axis `node`, which place them.
        """
        space = element.space or (space if isinstance(space, ContinuousSpace) else None)
        if isinstance(space, ContinuousSpace) or (agents and isinstance(space, LatticeSpace | NetworkSpace)):
            return getattr(space.prepare(), 'axes_', ())
        return ()

    @staticmethod
    def _is_single(element: Element) -> bool:
        """Return whether an element is one member whose values, even arrays, are its own.

        Such an element has no size and lies in no space or in a space without sites, such as a ball in the plane or
        a walker on a line, whose position `[0.25]` is a vector of one component and not one value per member.
        """
        space = element.space
        no_sites = space is None or not space._has_sites()
        return element.size is None and no_sites and _StateInference._get_link(element) is None

    @staticmethod
    def _get_rows(element: Element, value: object) -> object:
        """Return the values of the members of an element, one row for a single element."""
        return [value] if _StateInference._is_single(element) else value

    @staticmethod
    def _holds_objects(value: object) -> bool:
        """Return whether a value holds one object per member that is not a number.

        Such an array holds, for example, the queue of orders of every member, `[(), (7,)]`.
        """
        if not isinstance(value, np.ndarray) or value.dtype != object:
            return False
        return not all(isinstance(item, int | float | complex) for item in value.ravel())

    @staticmethod
    def _get_member_value(element: Element, name: str, value: object, count: int | None) -> object:
        """Return the value of one member, checking an array holds one per member.

        A single element in a space without sites, such as a ball in the plane, is one member whose value is the array
        itself, such as its velocity `[1.0, 0.0]`.

        On a lattice, an array of the shape of the lattice holds one value per site, such as the concentration of every
        cell of a plate.
        """
        array = np.asarray(value, dtype=object)
        if count is None or array.ndim == 0 or _StateInference._is_single(element):
            return value
        space = element.space
        if isinstance(space, LatticeSpace) and element.size is None and len(space.prepare().shape_) > 1:
            shape = tuple(space.shape_)
            if array.shape[: len(shape)] == shape:
                array = array.reshape(count, *array.shape[len(shape) :])
        if array.shape[0] != count and count == 1 and element.size is None:
            return value
        if array.shape[0] != count:
            error_msg = (
                f'Element {element.label!r} has {count} members but its value {name!r} has {array.shape[0]} entries. '
                f'Give one entry per member, or a single value that every member shares.'
            )
            raise ValidationError(error_msg)
        return array[0]

    @staticmethod
    def _get_new_member_default(
        element: Element,
        name: str,
        declared: DeclaredVariableSpec,
        member: object,
    ) -> DeclaredVariableSpec:
        """Return the declaration of a value per member, defaulting to the class default.

        A member that the laws add to a group, such as a new species or a new edge, takes the default of the argument
        of the class, as a new instance of the class would, and not the value of the first member. A value without a
        default of its class keeps the value of the first member.
        """
        parameter = inspect.signature(type(element)).parameters.get(name)
        if parameter is None or parameter.default is inspect.Parameter.empty:
            return declared
        default = parameter.default
        if np.ndim(np.asarray(default, dtype=object)) != np.ndim(np.asarray(member, dtype=object)):
            return declared
        return DeclaredVariableSpec(default, declared.spec)

    @staticmethod
    def _get_class_defaults(element: Element, names: Collection[str]) -> dict[str, object]:
        """Return the default of every value the element class gives a default.

        A member that a rule creates takes them for the values the rule does not give. A plain `Element(values=...)`
        takes its values in `values`, so its class gives no default.
        """
        parameters = inspect.signature(type(element)).parameters
        return {
            name: parameters[name].default
            for name in names
            if name in parameters and parameters[name].default is not inspect.Parameter.empty
        }

    @staticmethod
    def _is_plain(component: BaseComponent) -> bool:
        """Return whether an element has values or members but declares no states or laws.

        A group with no values, such as workers whose rules change other agents, is plain too. An element that names a
        compiler of its own, through its class as a word that productions rewrite does, or through a value as the
        Hilbert space of a spin does, is read by that compiler as it is, so it is not plain.
        """
        if not isinstance(component, Element) or isinstance(component, DeclaredElement):
            return False
        if component._get_compiler() is not None:
            return False
        if type(component)._get_laws is not Element._get_laws:
            return False
        valueless = not _StateInference._get_values(component) and component.size is None
        if valueless and not isinstance(component.space, NetworkSpace):
            return False
        specs = type(component)._get_specs()
        return not any(isinstance(spec, StateSpec | ParameterSpec) for spec in specs.values())

    @staticmethod
    def _get_quantity(annotation: object) -> str | None:
        """Return the name of the quantity in the `Annotated` of a value, or `None`.

        A value annotated as `Annotated[float, Quantity('energy')]` holds energy, which the conservation monitor reads.
        """
        for item in get_args(annotation)[1:] if get_origin(annotation) is Annotated else ():
            if isinstance(item, Quantity):
                item._check()
                return item.name
        return None

    @staticmethod
    def _get_group(annotation: object) -> MatrixGroupSpec | None:
        """Return the matrix group in the `Annotated` of a value, or `None`.

        A value annotated as `Annotated[object, MatrixGroupSpec('SU', 3)]` holds the real components of a matrix of
        that group, such as the link of a lattice gauge field.
        """
        for item in get_args(annotation)[1:] if get_origin(annotation) is Annotated else ():
            if isinstance(item, MatrixGroupSpec):
                return item
        return None

    @staticmethod
    def _get_period(annotation: object) -> float | None:
        """Return the period in the `Annotated` of a value, or `None`.

        A value annotated as `Annotated[float, Period(2 * pi)]` wraps around after every step.
        """
        for item in get_args(annotation)[1:] if get_origin(annotation) is Annotated else ():
            if isinstance(item, Period):
                item._check()
                return float(item.length)
        return None

    @staticmethod
    def _get_symmetry(annotation: object) -> str | None:
        """Return the symmetry of the tensor in the `Annotated` of a value, or `None`.

        A value annotated as `Annotated[object, Tensor('traceless')]` holds the independent components of a symmetric
        traceless matrix, which laws read as a Sage matrix.
        """
        for item in get_args(annotation)[1:] if get_origin(annotation) is Annotated else ():
            if isinstance(item, Tensor):
                item._check()
                return item.symmetry
        return None

    @staticmethod
    def _get_member(annotation: object) -> str | None:
        """Return the label of the group in the `Annotated` of a value, or `None`.

        A value annotated as `Annotated[int, Member('firms')]` names a member of the group `firms`, or none with -1.
        """
        for item in get_args(annotation)[1:] if get_origin(annotation) is Annotated else ():
            if isinstance(item, Member):
                item._check()
                return item.group
        return None

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
        moving: bool = False,
        history: object = None,
        control: bool = False,
        objects: bool = False,
        probing: bool = False,
    ) -> DeclaredVariableSpec:
        """Return the declaration of a value as a state or a parameter.

        A control of an objective law is an adaptive parameter, bounded by the interval of its allowed values, such as
        `u: Annotated[float, RealSet.closed(0, 1)]`. A value of objects, such as the queue of orders `(3, 7)` every
        member of a group holds, is a state of objects.

        The starting values of the derivatives or past values a law reads are given by their index, and a missing one
        is zero for a derivative and the value itself for a past value. A state with the role `'position'` is a
        position that walks, particle analyses, and drawings read. A periodic value wraps around after every step, and
        analyses read it as a heading when its element moves in a space and as a phase otherwise. The probe of the laws
        reads a value outside its interval as a real number, so a control that starts outside its bounds is reported
        as an issue of its bounds.

        Raises:
            ValidationError:
                A starting value is given for a parameter, or for a derivative or past value no law reads.
        """
        annotation = element._get_hints().get(name, inspect.Parameter.empty)
        domain = Element._get_domain_of(annotation, value)
        if probing and isinstance(domain, RealSet) and not EntrySpec._contains(domain, value):
            domain = RR
        quantity = self._get_quantity(annotation)
        period = self._get_period(annotation)
        member = self._get_member(annotation)
        group = self._get_group(annotation)
        if period is not None and role is None:
            role = 'direction' if moving else 'phase'
        if objects:
            domain = object
        symmetry = self._get_symmetry(annotation)
        vector, n_dim, tensor = (False, None, symmetry) if symmetry or objects else element._get_shape_of(value)
        if symmetry is None and not objects:
            value = element._get_matrix_entries(value, tensor)
        starting = {
            index: item if symmetry else element._get_matrix_entries(item, tensor)
            for index, item in (starting or {}).items()
        }
        count = lags or max(int(order or 1) - 1, 0)
        unread = sorted(index for index in starting if index > count)
        if unread and order is not None:
            error_msg = (
                f'Element {element.label!r} takes a starting value of {name!r} at index {unread[0]}, but its laws read '
                f'{count} derivatives or past values of it.'
            )
            raise ValidationError(error_msg)
        if order is None:
            declared = ParameterSpec(domain=domain, vector=vector, n_dim=n_dim, tensor=tensor)
            return DeclaredVariableSpec(value, declared)
        if control:
            bounds = (float(domain.inf()), float(domain.sup())) if isinstance(domain, RealSet) else None
            return DeclaredVariableSpec(value, ParameterSpec(domain=domain, adaptive=True, bounds=bounds))
        if order == 0:
            spec = StateSpec(
                domain=domain,
                vector=vector,
                n_dim=n_dim,
                tensor=tensor,
                algebraic=True,
                role=role,
                quantity=quantity,
            )
            return DeclaredVariableSpec(value, spec)
        if group is not None:
            return DeclaredVariableSpec(value, StateSpec(group=group, role=role, quantity=quantity))
        missing = value if lags else 0.0
        initial = tuple(starting.get(index, missing) for index in range(1, count + 1)) if starting else ()
        steps = 1 if lags else order
        spec = StateSpec(
            domain=domain,
            vector=vector,
            n_dim=n_dim,
            tensor=tensor,
            order=steps,
            lags=lags,
            initial=initial,
            period=period,
            role=role,
            quantity=quantity,
            reference=member,
            history=history,
        )
        return DeclaredVariableSpec(value, spec)

    def _replace(
        self,
        system: CompositeSystem,
        declare: Callable[[Element, str, BaseSpace | None], list[BaseElement]],
        scope: str = '',
        space: BaseSpace | None = None,
    ) -> CompositeSystem:
        """Return a copy of a system tree whose plain elements are declared elements.

        Every element is declared in the space of its nearest system. The copy keeps the observables of every system.
        """
        space = system.space or space
        elements = [
            declared
            for element in system.elements
            for declared in (
                declare(element, f'{scope}{element.label}', space) if self._is_plain(element) else [element]
            )
        ]
        systems = [
            (
                self._replace(nested, declare, f'{scope}{nested.label}/', space)
                if isinstance(nested, CompositeSystem)
                else nested
            )
            for nested in system.systems
        ]
        replaced = CompositeSystem(
            system.label,
            elements=elements,
            interactions=list(system.interactions),
            systems=systems,
            space=system.space,
            time=system.time,
            solver=system.solver,
        )
        if hasattr(system, 'observables'):
            replaced.observables = list(system.observables)
        return replaced

    def _has_plain(self, system: CompositeSystem) -> bool:
        """Return whether a system tree holds a plain element."""
        if any(self._is_plain(element) for element in system.elements):
            return True
        return any(isinstance(nested, CompositeSystem) and self._has_plain(nested) for nested in system.systems)

    @staticmethod
    def _get_lags(law: BaseComponent, named: Mapping[str, str]) -> dict[str, int]:
        """Return the number of past values a law reads of every value, by its dotted name.

        A reduction over neighbours that reads past values of the other member, such as the delayed headings of a flock,
        counts too.
        """
        lags: dict[str, int] = {}
        values = law._get_values() if isinstance(law, BaseLaw) else []
        for value in values:
            if not isinstance(value, Expression):
                continue
            for found in value.find(lag(SR.wild(0), SR.wild(1))):
                state, steps = found.operands()
                read = str(state).removesuffix(OTHER)
                if read in named and SR(steps).is_integer():
                    name = named[read]
                    lags[name] = max(int(steps), lags.get(name, 0))
        return lags

    @staticmethod
    def _get_controls(compilation: Compilation) -> set[str]:
        """Return the dotted names of the values an objective law steers as its controls.

        A control, such as the input `u` of `ObjectiveLaw(x**2 + u**2, controls=[u])`, is an adaptive parameter.
        """
        census = Compilation._get_required(compilation.census, 'census')
        return {
            entry.name
            for record in census.records
            if isinstance(record.law, ObjectiveLaw) and record.law.controls is not None
            for entry in record.targets
        }

    @staticmethod
    def _get_delays(compilation: Compilation) -> set[str]:
        """Return the dotted names of the values the laws read with a delay.

        A law reads such a value as `x(t=t - tau)`.
        """
        census = Compilation._get_required(compilation.census, 'census')
        layout = Compilation._get_required(compilation.layout, 'layout')
        named = {
            str(block.symbols[variable]): entry.name
            for block in layout.blocks.values()
            for variable, entry in block.entries.items()
        }
        patterns = (delayed(SR.wild(0), SR.wild(1)), delayed(SR.wild(0), SR.wild(1), SR.wild(2)))
        delays = set()
        for record in census.records:
            values = record.law._get_values() if isinstance(record.law, BaseLaw) else []
            for value in values:
                if not isinstance(value, Expression):
                    continue
                found = [item for pattern in patterns for item in value.find(pattern)]
                delays |= {named[str(item.operands()[0])] for item in found if str(item.operands()[0]) in named}
        return delays

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
    def _check_rate_order(name: str, order: float, known: float | None) -> None:
        """Raise an error for rate laws of two integer orders on one state.

        Rates of the same derivative add up, but a rate of `x` beside a rate of `x''` states the motion twice.

        Raises:
            ValidationError:
                The state has rate laws of two different integer orders.
        """
        if known is None or known == order or not float(order).is_integer() or not float(known).is_integer():
            return
        low, high = sorted((int(order), int(known)))
        error_msg = (
            f'The state {name} has rate laws of order {low} and of order {high}, but the rates of a state add up '
            f'only on one derivative. Write each law on the same derivative, such as both on the derivative of '
            f'order {high}, or derive the rates from one interaction.'
        )
        raise ValidationError(error_msg)

    @staticmethod
    def _get_densities(law: BaseComponent, named: Mapping[str, str]) -> list[str]:
        """Return the names of the densities a bilinear form reads.

        A density that two populations meet through, such as the field of an encounter, is a state also when no law
        changes it, since the form reads it from the states as it reads the density it changes.
        """
        if not isinstance(law, BilinearLaw):
            return []
        return [named[str(item)] for item in (law.first, law.second) if item is not None and str(item) in named]

    @staticmethod
    def _get_orders(compilation: Compilation) -> tuple[dict[str, float], dict[str, int]]:
        """Return the order and the past values of every value the laws use.

        A value that only constraints determine has order 0, as the potential of a circuit node whose currents must
        balance, also when a constraint reads derivatives of other values, as the back electromotive force of a motor
        reads the speed of its rotor. A value that rate laws give a fractional derivative has its fractional order.
        A value whose derivative of order `k` a constraint reads has order `k` at least, as the charge of
        `L q'' + R q' + q / C == 0`.

        Raises:
            ValidationError:
                A value has rate laws of two different integer orders, or a fractional order beside another order.
        """
        census = Compilation._get_required(compilation.census, 'census')
        layout = Compilation._get_required(compilation.layout, 'layout')
        named = {
            str(block.symbols[variable]): entry.name
            for block in layout.blocks.values()
            for variable, entry in block.entries.items()
        }
        orders: dict[str, float] = {}
        rates: dict[str, float] = {}
        lags: dict[str, int] = {}
        for record in census.records:
            law = record.law
            for name in _StateInference._get_densities(law, named):
                orders[name] = _StateInference._get_order(name, 1, orders.get(name))
            for name, steps in _StateInference._get_lags(law, named).items():
                lags[name] = max(steps, lags.get(name, 0))
            order = (
                (law.order or 1)
                if isinstance(law, RateLaw | NoiseLaw)
                else 0 if isinstance(law, LinearProgramLaw) else 1
            )
            names = [entry.name for entry in record.targets]
            if isinstance(law, ConstraintLaw) and law.target is not None and str(law.target) in named:
                names.append(named[str(law.target)])
                order = 0
            for name in names:
                if isinstance(law, RateLaw):
                    _StateInference._check_rate_order(name, order, rates.get(name))
                    rates[name] = order
                orders[name] = _StateInference._get_order(name, order, orders.get(name))
            if isinstance(law, ConstraintLaw):
                for name, steps in _StateInference._get_derivative_orders(law, named).items():
                    orders[name] = _StateInference._get_order(name, steps, orders.get(name))
        return orders, lags

    def get_declared(self, system: CompositeSystem) -> CompositeSystem:
        """Return a system tree whose plain elements declare what their laws show.

        A probe declares every value as a state and collects the laws. When the probe fails on a space, its system tree,
        or the system itself when the elements cannot even be probed, is returned, so the compiler reports the failure
        with every other issue, such as a phase space above its limit or a tensor value without a space.

        Args:
            system:
                The system.

        Returns:
            system:
                The system, or a copy whose plain elements are declared elements.
        """
        if not self._has_plain(system):
            return system

        def probe(element: Element, path: str, space: BaseSpace | None) -> list[BaseElement]:
            return self._get_declared_elements(element, lambda name: 1, lambda name: 0, probing=True, space=space)

        try:
            probed = self._replace(system, probe)
        except SpaceError:
            return system
        try:
            compilation = self.census(probed)
        except SpaceError:
            return probed
        orders, lags = self._get_orders(compilation)
        orders = dict.fromkeys(self._get_rule_reads(compilation), 1) | orders
        kinds = self._get_topology_kinds(compilation)
        sizes = self._get_hyperedge_sizes(compilation)
        links = self._get_edge_spaces(system)
        agents = self._get_agent_groups(compilation)
        steps = self._get_steps(compilation)
        delays = self._get_delays(compilation)
        controls = self._get_controls(compilation)

        def declare(element: Element, path: str, space: BaseSpace | None) -> list[BaseElement]:
            agent = path in agents and not self._is_field(element, kinds)
            return self._get_declared_elements(
                element,
                (lambda name: 1) if agent else (lambda name: orders.get(f'{path}.{name}')),
                lambda name: lags.get(f'{path}.{name}', 0),
                kinds,
                agents=path in agents,
                sizes=sizes,
                links=links,
                delays=frozenset(name.removeprefix(f'{path}.') for name in delays if name.startswith(f'{path}.')),
                controls=frozenset(name.removeprefix(f'{path}.') for name in controls if name.startswith(f'{path}.')),
                steps={
                    name.removeprefix(f'{path}.'): moved.removeprefix(f'{path}.')
                    for name, moved in steps.items()
                    if name.startswith(f'{path}.') and moved.startswith(f'{path}.')
                },
                space=space,
            )

        return self._replace(system, declare)

    @staticmethod
    def _is_field(element: Element, kinds: frozenset[str]) -> bool:
        """Return whether an element is a field on the sites or the links of a space.

        Such an element is a field, such as the pheromone on the cells of a lattice, and not a group of agents, even
        when rules read it through a role. Rules change a field only through deposits into the values that its laws
        change, so it keeps as states only those values, as every other element does, and its other values, such as a
        rate of evaporation, stay parameters. The nodes and links of a network whose laws change it are agents.
        """
        space = element.space
        if isinstance(space, NetworkSpace) and kinds:
            return False
        if _StateInference._get_link(element) is not None:
            return True
        return isinstance(space, LatticeSpace | NetworkSpace) and element.size is None

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
    def _get_rule_reads(compilation: Compilation) -> set[str]:
        """Return the dotted names of the values whose neighbours a local rule reads.

        Such a value is a state even when no law changes it, as the trapping times of the sites of a walk, since a
        local rule reads the states of the neighbours of every site.
        """
        census = Compilation._get_required(compilation.census, 'census')
        layout = Compilation._get_required(compilation.layout, 'layout')
        named = {
            str(block.symbols[variable]): entry.name
            for block in layout.blocks.values()
            for variable, entry in block.entries.items()
        }
        reads: set[str] = set()
        for record in census.records:
            law = record.law
            local = isinstance(law, LocalRuleLaw) or (isinstance(law, UpdateLaw) and law.neighborhood is not None)
            if local and law.reads is not None:
                reads.update(named[str(item)] for item in law.reads if str(item) in named)
        return reads

    @staticmethod
    def _get_hyperedge_sizes(compilation: Compilation) -> frozenset[int]:
        """Return the sizes of the hyperedges the laws add.

        A new good made from three parents forms a hyperedge of four members, its parents and itself.
        """
        census = Compilation._get_required(compilation.census, 'census')
        sizes: list[int] = []
        for record in census.records:
            law = record.law
            requests = (
                [law.target]
                if isinstance(law, UpdateLaw)
                else list(law.changes or {}) if isinstance(law, JumpLaw) else []
            )
            sizes.extend(
                int(item.hyperedge)
                for item in requests
                if isinstance(item, TopologyRequestSpec) and item.hyperedge is not None
            )
        return frozenset(sizes)

    def _get_edge_spaces(self, system: CompositeSystem) -> frozenset[int]:
        """Return the identities of the networks whose edges a plain element holds.

        A network whose edges an element such as `Element('catalysis', space=network, on='edges')` holds takes no
        other group on its edges.
        """
        spaces = {
            id(element.space)
            for element in system.elements
            if self._is_plain(element) and element.on == 'edges' and isinstance(element.space, NetworkSpace)
        }
        for nested in system.systems:
            if isinstance(nested, CompositeSystem):
                spaces |= self._get_edge_spaces(nested)
        return frozenset(spaces)

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
        """Return the paths of the groups whose members are agents.

        Their members run agent rules, or a rule reads them through the roles of its interaction, such as the traders a
        market clears with `UpdateLaw(s, clear)` or the balls an urn draws from with `UpdateLaw(s.target, draw)`. An
        element whose values rate or noise laws change flows in continuous time, so it is no agent even when a rule of
        the same interaction reads it, as the soil that farmers tend.
        """
        census = Compilation._get_required(compilation.census, 'census')
        nodes = {node.path: node for node in Compilation._get_required(compilation.nodes, 'nodes')}
        groups: set[str] = set()
        flowing: set[str] = set()
        for record in census.records:
            if isinstance(record.law, RateLaw | NoiseLaw):
                flowing.update(entry.name.rpartition('.')[0] for entry in record.targets)
            node = nodes.get(record.path)
            if not isinstance(record.law, RuleLaw) or node is None:
                continue
            for role in declarations_of(node.component, VariableKind.ROLE):
                groups.update(f'{node.scope}{label}' for label in bound_labels(binding_of(node.component, role)))
        return groups - flowing

    @staticmethod
    def _get_field(label: str, space: LatticeSpace | NetworkSpace, values: Mapping[str, object]) -> FieldElement:
        """Return the field of declared members on a space, discretized as the space says.

        A lattice gives its discretization, the edge of each state, the symmetry, and the finite-volume scheme, such as
        `LatticeSpace(64, discretization='spectral')`. A mesh is solved by finite elements with the edge of each state
        it gives, and a network by finite differences.
        """
        if isinstance(space, LatticeSpace):
            return FieldElement(
                label,
                element=DeclaredElement,
                space=space,
                values=values,
                discretization=space.discretization,
                boundary=space.boundary if isinstance(space.boundary, Mapping) else None,
                symmetry=space.symmetry,
                scheme=space.scheme,
            )
        mesh = isinstance(space, CellComplexSpace) and space._is_mesh()
        boundary = space.boundary if isinstance(space, CellComplexSpace) else None
        discretization = 'finite-element' if mesh else 'finite-difference'
        return FieldElement(
            label,
            element=DeclaredElement,
            space=space,
            values=values,
            discretization=discretization,
            boundary=boundary,
        )

    @staticmethod
    def _get_node_edges(element: Element, links: frozenset[int]) -> list[GroupElement]:
        """Return the variable group on the changing edges of a network, after its nodes.

        A network whose edges an element holds, such as `Element('catalysis', space=network, on='edges')`, takes no
        such group, since that element is the group on its edges.
        """
        if id(element.space) in links:
            return []
        return [GroupElement(f'{element.label}_edges', element=Element, space=element.space, on='edges', variable=True)]

    @staticmethod
    def _get_link_group(
        element: Element,
        link: tuple[str, str | None],
        count: int | None,
        values: Mapping[str, object],
        *,
        changing: bool,
        sizes: frozenset[int],
    ) -> GroupElement:
        """Return the group with one member on every edge or hyperedge of a network.

        When the laws change the network, the group grows and shrinks with its edges or hyperedges. A group on
        hyperedges of several sizes holds the largest hyperedge, its rank, and a group whose hyperedges, those it starts
        with and those the laws add, have one size has no rank, so laws can run along them. A group on a network
        without hyperedges starts empty, as the production functions of goods that are yet to be made. A group on the
        edges of a layer has one member on each edge of that layer.
        """
        on, layer = link
        empty = on == 'hyperedges' and count == 0
        space = element.space
        variable = isinstance(space, NetworkSpace) and (changing or empty)
        rank = None
        if isinstance(space, NetworkSpace) and variable and on == 'hyperedges':
            starting = {len(item) for item in space.prepare().hyperedges_}
            every = starting | sizes
            rank = None if starting and len(every) == 1 else max(every, default=MIN_RANK)
        return GroupElement(
            element.label,
            element=DeclaredElement,
            size=0 if empty else None,
            space=element.space,
            on=on,
            layer=layer,
            values=values,
            variable=variable,
            rank=rank,
        )

    @staticmethod
    def _get_probed(value: object) -> object:
        """Return a value for the probe of the laws, with zeros in place of `nan`.

        The probe reads only the shapes of the values, and an agent may hold `nan` until a rule sets it.
        """
        if isinstance(value, np.ndarray | list | tuple | float):
            array = np.asarray(value)
            if array.dtype.kind == 'f' and np.isnan(array).any():
                return np.nan_to_num(array, nan=0.0)
        return value

    @staticmethod
    def _get_start_regions(
        element: Element,
        region: object,
        count: int | None,
        agents: bool,
        space: BaseSpace | None = None,
    ) -> list[Polygon]:
        """Return the region of its continuous space that every member starts in.

        The value `region` names one region of the space for every member, or one region for all of them, such as
        `Element('residents', values={'region': 'tract_1'}, size=100, space=space)`. The run draws the starting point
        of every member inside its region.

        Args:
            element:
                The element.

            region:
                The name of a region, or one name per member.

            count:
                The number of members, or `None` for a single element.

            agents:
                Whether the members run agent rules.

            space:
                The space of the system, which an element without a space of its own lives in.

        Returns:
            regions:
                The polygon every member starts in.

        Raises:
            ValidationError:
                The element is not on a continuous space, its members run no agent rules, or a name is not one of the
                regions of the space.
        """
        space = element.space if element.space is not None else space
        if not isinstance(space, ContinuousSpace) or not agents:
            error_msg = (
                f'Element {element.label!r} starts in a region, which members that run agent rules on a continuous '
                "space do, such as Element('residents', values={'region': 'tract_1'}, size=100, space=space)."
            )
            raise ValidationError(error_msg)
        named = dict(space.regions or {})
        names = np.broadcast_to(np.asarray(region, dtype=object), (count or 1,)).tolist()
        regions = []
        for name in names:
            polygon = named.get(name) if isinstance(name, str) else None
            if not isinstance(polygon, Polygon):
                found = [key for key, value in named.items() if isinstance(value, Polygon)]
                error_msg = (
                    f'Element {element.label!r} starts in the region {name!r}, which the space does not name. Its '
                    f'regions are: {", ".join(found) or "none"}.'
                )
                raise ValidationError(error_msg)
            regions.append(polygon)
        return regions

    @staticmethod
    def _get_group_on(element: Element, count: int | None, agents: bool) -> str | None:
        """Return where the members of a group sit, `'sites'` or `None`.

        The members of a group in a continuous space, or one per node of a network, sit on its sites. Agents of a
        network and another number of members, such as the balls in the urns of its nodes, sit on no sites and are
        placed by their value `node`.
        """
        space = element.space
        if space is None or isinstance(space, LatticeSpace):
            return None
        if isinstance(space, NetworkSpace) and (agents or count != space.graph.number_of_nodes()):
            return None
        return 'sites'

    @staticmethod
    def _get_rule_derivative_specs(
        element: Element,
        derivatives: Mapping[str, tuple[str, str, object]],
        count: int | None,
        axes: Collection[str],
    ) -> dict[str, DeclaredVariableSpec]:
        """Return the declarations of the derivatives that the rules of agents change.

        They are keyed by engine name, and the first derivative of a position, such as `"x'"`, is a velocity.
        """
        declared = {}
        for name, (_, base, given) in derivatives.items():
            role = 'velocity' if name == f'{base}_d1' and base in axes else None
            member = _StateInference._get_member_value(element, base, given, count)
            declared[name] = DeclaredVariableSpec(member, StateSpec(domain=RR, role=role))
        return declared

    @staticmethod
    def _get_past_arrays(name: str, values: object, lags: int) -> dict[str, object]:
        """Return the past values of many members, each its value before the start."""
        return {f'{name}_lag{index}': values for index in range(1, lags + 1)}

    def _get_declared_elements(
        self,
        element: Element,
        order: Callable[[str], float | None],
        lags: Callable[[str], int],
        kinds: frozenset[str] = frozenset(),
        probing: bool = False,
        *,
        agents: bool = False,
        sizes: frozenset[int] = frozenset(),
        links: frozenset[int] = frozenset(),
        steps: Mapping[str, str] | None = None,
        space: BaseSpace | None = None,
        delays: frozenset[str] = frozenset(),
        controls: frozenset[str] = frozenset(),
    ) -> list[BaseElement]:
        """Return the declared element of a plain element, or a group of declared members.

        On a network whose edges the laws change, the members are a variable group of nodes, with a variable group on
        the edges named after the element, such as `voters_edges`. When the laws also add or remove nodes, every value
        of a node is a state, since it comes and goes with its node, and the group grows as a Python list does. The
        members of a group that runs agent rules are agents, whose rules may change every value and add or remove
        members, so every value is a state and the group is variable, and a single element that runs them is a group of
        one agent. A value named after an axis of a continuous space, or of the cells of a lattice for agents, is a
        position, also when no law changes it, and the value an update law adds to a position at every step, as `v` in
        `x <- x + v`, is its velocity. An element on the edges or hyperedges of a network is a group with one member on
        each, which grows and shrinks with them when the laws change the network. The probe that finds the orders
        reads no starting values of derivatives or past values. The element keeps the drawing of its class, except a
        drawing of agents for members that run no agent rules, and the view of its agents.
        """
        values = self._get_values(element)
        region = values.pop(REGION, None)
        values = {name: self._get_probed(value) for name, value in values.items()} if probing else values
        histories = self._get_histories(element, delays)
        derivatives = self._get_rule_derivatives(element) if agents else {}
        skipped = [name for name, _ in histories.values()] + [name for name, _, _ in derivatives.values()]
        starting = {} if probing else self._get_starting(element, values, skipped)
        count = self._get_count(element)
        count = 1 if count is None and agents else count
        regions = None if region is None or probing else self._get_start_regions(element, region, count, agents, space)
        link = self._get_link(element)
        network = bool(kinds) and isinstance(element.space, NetworkSpace) and link is None
        growing = network and bool(kinds & MEMBERSHIP)
        axes = self._get_axes(element, agents, space)
        velocities = {name for name, moved in (steps or {}).items() if moved in axes}
        variables = {}
        arrays = {}
        for name, value in values.items():
            member = self._get_member_value(element, name, value, count)
            kept = growing or name in axes
            free = None if probing else self._get_free_order(element, name)
            variables[name] = self._get_spec(
                element,
                name,
                member,
                (
                    free
                    if order(name) is None and free is not None
                    else 1 if order(name) is None and kept else order(name)
                ),
                lags=lags(name),
                starting={
                    index: self._get_member_value(element, name, given, count)
                    for index, given in starting.get(name, {}).items()
                },
                role='position' if name in axes else 'velocity' if name in velocities else None,
                moving=any(axis in values for axis in axes),
                history=histories[name][1] if name in histories else None,
                control=name in controls,
                objects=self._holds_objects(value),
                probing=probing,
            )
            annotation = element._get_hints().get(name, inspect.Parameter.empty)
            symmetric = self._get_symmetry(annotation) or self._holds_objects(value)
            tensor = None if symmetric else element._get_shape_of(member)[2]
            if count is not None and np.ndim(np.asarray(value, dtype=object)) > 0:
                variables[name] = self._get_new_member_default(element, name, variables[name], member)
                arrays[name] = self._get_rows(element, element._get_matrix_entries(value, tensor))
                arrays |= self._get_past_arrays(name, arrays[name], 0 if count is None or probing else lags(name))
            for index, given in starting.get(name, {}).items():
                if count is not None and np.ndim(np.asarray(given, dtype=object)) > 0:
                    arrays[f'{name}_lag{index}' if lags(name) else f'{name}_d{index}'] = self._get_rows(
                        element,
                        element._get_matrix_entries(given, tensor),
                    )
        if not probing:
            variables |= self._get_rule_derivative_specs(element, derivatives, count, axes)
            arrays |= {
                name: self._get_rows(element, given)
                for name, (_, _, given) in derivatives.items()
                if np.ndim(np.asarray(given, dtype=object))
            }
        visual = element._get_visual()
        drawing = None if visual == VisualSpec() or (visual.kind == 'agents' and not agents) else visual
        view = element._get_view()
        if count is None:
            return [
                DeclaredElement(element.label, variables=variables, drawing=drawing, view=view, origin=type(element)),
            ]
        defaults = self._get_class_defaults(element, variables) if agents else None
        values = {
            'variables': variables,
            'drawing': drawing,
            'view': view,
            'defaults': defaults,
            'origin': type(element),
            **arrays,
        }
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
            return [nodes, *self._get_node_edges(element, links)]
        if link is not None:
            return [self._get_link_group(element, link, count, values, changing=bool(kinds), sizes=sizes)]
        if isinstance(element.space, LatticeSpace | NetworkSpace) and element.size is None:
            return [self._get_field(element.label, element.space, values)]
        place = self._get_group_on(element, count, agents)
        off_nodes = place is None and isinstance(element.space, NetworkSpace) and not agents
        group = GroupElement(
            element.label,
            element=DeclaredElement,
            size=count,
            space=None if off_nodes else element.space,
            on=place,
            values=values,
            location=regions,
            variable=agents,
        )
        return [group]

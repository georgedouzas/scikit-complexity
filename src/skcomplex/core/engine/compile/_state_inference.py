"""Infer the states and parameters of plain elements from the laws of their system."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import get_args, get_origin

import numpy as np
from sage.all__sagemath_symbolics import RR, SR, ZZ, Expression
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from ...components.base import BaseComponent
from ...components.elements import DeclaredElement, Element
from ...components.laws import BaseLaw, ConstraintLaw, NoiseLaw, RateLaw
from ...components.laws.utils import lag
from ...components.systems import CompositeSystem
from ...specs import DeclaredVariableSpec, ParameterSpec, StateSpec
from ._compilation import Compilation

RESERVED = ('label', 'visual', 'size', 'space')
BOOLEAN = FiniteEnumeratedSet([0, 1])


class _StateInference:
    """Inference of the states and parameters of elements that declare neither.

    A value of such an element that some law changes is a state, of the order of the highest derivative its laws give,
    and any other value is a parameter. The kind of a value comes from its Python type hint, and a constraint from the
    allowed values in `Annotated`.
    """

    def __init__(self, census: Callable[[CompositeSystem], Compilation]) -> None:
        self.census = census

    @staticmethod
    def _get_values(element: Element) -> dict[str, object]:
        """Return the values of a plain element by name, without its settings."""
        return {name: value for name, value in element.get_params(deep=False).items() if name not in RESERVED}

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
        """Return the domain of a value from its annotation or its Python type."""
        if get_origin(annotation) is not None and get_args(annotation):
            for item in get_args(annotation)[1:]:
                if isinstance(item, set | frozenset):
                    return FiniteEnumeratedSet(sorted(item, key=str))
                if hasattr(item, '__contains__') and not isinstance(item, str):
                    return item
            annotation = get_args(annotation)[0]
        kind = annotation if isinstance(annotation, type) else type(value)
        if kind is bool:
            return BOOLEAN
        if kind is int:
            return ZZ
        return RR

    def _get_spec(
        self,
        element: Element,
        name: str,
        value: object,
        order: int | None,
        lags: int = 0,
    ) -> DeclaredVariableSpec:
        """Return the declaration of a value as a state or a parameter."""
        annotation = inspect.signature(type(element)).parameters[name].annotation
        domain = self._get_domain(annotation, value)
        vector = np.ndim(np.asarray(value, dtype=object)) > 0
        spec = (
            ParameterSpec(domain=domain, vector=vector)
            if order is None
            else StateSpec(domain=domain, vector=vector, order=1 if lags else order, lags=lags)
        )
        return DeclaredVariableSpec(value, spec)

    def _replace(
        self,
        system: CompositeSystem,
        declare: Callable[[Element, str], DeclaredElement],
        scope: str = '',
    ) -> CompositeSystem:
        """Return a copy of a system tree whose plain elements are declared elements."""
        elements = [
            declare(element, f'{scope}{element.label}') if self._is_plain(element) else element
            for element in system.elements
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
    def _get_orders(compilation: Compilation) -> tuple[dict[str, int], dict[str, int]]:
        """Return the order and the past values of every value the laws use."""
        census = Compilation._get_required(compilation.census, 'census')
        layout = Compilation._get_required(compilation.layout, 'layout')
        named = {
            str(block.symbols[variable]): entry.name
            for block in layout.blocks.values()
            for variable, entry in block.entries.items()
        }
        orders: dict[str, int] = {}
        lags: dict[str, int] = {}
        for record in census.records:
            law = record.law
            for name, steps in _StateInference._get_lags(law, named).items():
                lags[name] = max(steps, lags.get(name, 0))
            order = (law.order or 1) if isinstance(law, RateLaw | NoiseLaw) else 1
            names = [entry.name for entry in record.targets]
            if isinstance(law, ConstraintLaw) and law.target is not None and str(law.target) in named:
                names.append(named[str(law.target)])
            for name in names:
                orders[name] = max(order, orders.get(name, 1))
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

        def probe(element: Element, path: str) -> DeclaredElement:
            values = self._get_values(element)
            variables = {name: self._get_spec(element, name, value, 1) for name, value in values.items()}
            return DeclaredElement(element.label, visual=element.visual, variables=variables)

        orders, lags = self._get_orders(self.census(self._replace(system, probe)))

        def declare(element: Element, path: str) -> DeclaredElement:
            values = self._get_values(element)
            variables = {
                name: self._get_spec(element, name, value, orders.get(f'{path}.{name}'), lags.get(f'{path}.{name}', 0))
                for name, value in values.items()
            }
            return DeclaredElement(element.label, visual=element.visual, variables=variables)

        return self._replace(system, declare)

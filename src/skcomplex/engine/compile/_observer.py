"""Values of the observables that laws read at the current state, built from the blocks of a system."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from sage.all__sagemath_symbolics import Expression
from sage.modules.free_module_element import FreeModuleElement

from ...core.errors import ValidationError
from ...core.laws.utils import count, maximum, mean, minimum, total, variance
from ...core.specs import ObservableSpec
from ..problems import (
    Problem,
    Reference,
    Source,
    vector,
)
from ._block import Block
from ._declarations import Declarations
from ._observables import Declared, ObservableEvaluator
from ._observed import Observed
from ._tree import Node

_REDUCERS: dict[str, Callable[..., Expression]] = {
    'sum': total,
    'mean': mean,
    'max': maximum,
    'min': minimum,
    'var': variance,
}


@dataclass
class Observer:
    """Values of the observables that laws read at the current state, built from the blocks of a system."""

    declarations: Declarations
    live: Declared
    values: dict[str, tuple[object, dict[Expression, Reference]]] = field(default_factory=dict)

    def _system_value(
        self,
        path: str,
        scope: str,
        observable: ObservableSpec,
    ) -> tuple[object, dict[Expression, Reference]]:
        """Return an expression of the symbols of a system's elements, with earlier observables by label."""
        prefix = f'{scope}/' if scope else ''
        items: dict[str, Any] = {}
        references: dict[Expression, Reference] = {}
        for block in self.declarations.blocks.values():
            for variable, entry in block.entries.items():
                if entry.members is None and entry.name.startswith(prefix):
                    items[entry.name[len(prefix) :]] = block.symbols[variable]
                    references.update(
                        {item: block.references[item] for item in Problem._real_components_of(block.symbols[variable])},
                    )
        namespace = ObservableEvaluator._tree(items)
        for other, (other_scope, earlier) in self.live.items():
            if other == path:
                break
            if other_scope == scope:
                value, other_references = self.value(other, other_scope, earlier)
                setattr(namespace, earlier.label, value)
                references.update(other_references)
        expression = observable.expression(namespace) if callable(observable.expression) else observable.expression
        return expression, references

    def value(
        self,
        path: str,
        scope: str,
        observable: ObservableSpec,
    ) -> tuple[object, dict[Expression, Reference]]:
        """Return the value of an observable with the references of the symbols it reads."""
        if path in self.values:
            return self.values[path]
        if observable.group is None:
            self.values[path] = self._system_value(path, scope, observable)
            return self.values[path]
        block = self.declarations.blocks[ObservableEvaluator._join(scope, observable.group)]
        _, references = block.namespace(self._source_of(block))
        self.values[path] = (self._reduced_value(path, observable, block), references)
        return self.values[path]

    def evaluate(self, node: Node, label: object) -> Observed | None:
        """Return the namespace of the observable a role binds to, or `None` for a role bound to no live observable."""
        path = f'{node.scope}{label}' if isinstance(label, str) else None
        if path not in self.live:
            return None
        scope, observable = self.live[path]
        value, references = self.value(path, scope, observable)
        return Observed(SimpleNamespace(value=value), dict(references))

    @staticmethod
    def _source_of(block: Block) -> Source:
        """Return how laws read the variables of a block from outside it, per member for a group."""
        return Source.MEMBER if block.node.members is not None else Source.SINGLE

    @staticmethod
    def _reduced_value(path: str, observable: ObservableSpec, block: Block) -> object:
        """Return an observable that reduces a group as a reduction over the group's member symbols."""
        namespace, _ = block.namespace(Observer._source_of(block))
        if observable.reduce == 'count':
            return count(over=namespace)
        if observable.variable is not None:
            if observable.variable not in block.symbols:
                error_msg = (
                    f'ObservableSpec {path} reduces {observable.variable}, which {block.node.path} does not declare.'
                )
                raise ValidationError(error_msg)
            value: Any = block.symbols[observable.variable]
        elif observable.expression is None:
            error_msg = (
                f'ObservableSpec {path} reduces {block.node.path} with {observable.reduce!r} but names no variable.'
            )
            raise ValidationError(error_msg)
        else:
            value = observable.expression(namespace) if callable(observable.expression) else observable.expression
        reducer = _REDUCERS[str(observable.reduce)]
        if isinstance(value, FreeModuleElement):
            return vector(reducer(item, over=namespace) for item in value)
        return reducer(value, over=namespace)

"""Values of the observables that laws read at the current state, built from the blocks of a system."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import replace
from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression
from sage.modules.free_module_element import FreeModuleElement

from ...base import BaseConfig
from ...core.laws.utils import count, maximum, mean, minimum, total, variance
from ...core.specs import NodeSpec
from ...errors import ValidationError
from ...specs import ObservableSpec
from .._utils import vector
from ..problems import BlockSpec, DeclarationsSpec, ObservedSpec, ProblemSpec, ReferenceSpec
from ..types import ReadKind
from ._observables import Declared, ObservableEvaluator

_REDUCERS: dict[str, Callable[..., Expression]] = {
    'sum': total,
    'mean': mean,
    'max': maximum,
    'min': minimum,
    'var': variance,
}


class Observer(BaseConfig[None, tuple[NodeSpec, object]]):
    """Values of the observables that laws read at the current state, built from the blocks of a system.

    `prepare()` starts with no observable evaluated, and `run((node, label))` sets `observed_`, the namespace of the
    observable a role of a component binds to, evaluating every observable once.

    Args:
        declarations:
            The declarations of the components of the system.

        live:
            The scope and the observable of every observable that laws read at the current state, by path.

    Attributes:
        values_:
            The value of every evaluated observable with the references of the symbols it reads, by path.

        observed_:
            The namespace of the observable the last run's role binds to with the references of its symbols, or `None`
            for a role bound to no live observable.
    """

    values_: dict[str, tuple[object, dict[Expression, ReferenceSpec]]]
    observed_: ObservedSpec | None

    def __init__(self, declarations: DeclarationsSpec, live: Declared) -> None:
        self.declarations = declarations
        self.live = live

    def _prepare(self, context: None) -> None:
        """Start with no observable evaluated.

        Args:
            context:
                Nothing: the observer serves its declarations.
        """
        self.values_ = {}

    def _run(self, inputs: tuple[NodeSpec, object] | None) -> None:
        """Set the namespace of the observable a role binds to, or `None` for a role bound to no live observable.

        Args:
            inputs:
                The component that holds the role and the label the role binds to.

        Raises:
            ValidationError:
                No component and label are passed.
        """
        if inputs is None:
            error_msg = 'An observer runs on a component and the label its role binds to. Pass them to `run`.'
            raise ValidationError(error_msg)
        node, label = inputs
        path = f'{node.scope}{label}' if isinstance(label, str) else None
        if path not in self.live:
            self.observed_ = None
            return
        scope, observable = self.live[path]
        value, references = self._value(path, scope, observable)
        self.observed_ = ObservedSpec(SimpleNamespace(value=value), dict(references))

    def _system_value(
        self,
        path: str,
        scope: str,
        observable: ObservableSpec,
    ) -> tuple[object, dict[Expression, ReferenceSpec]]:
        """Return an expression of the symbols of a system's elements, with earlier observables by label."""
        prefix = f'{scope}/' if scope else ''
        items: dict[str, object] = {}
        references: dict[Expression, ReferenceSpec] = {}
        for block in self.declarations.blocks.values():
            for variable, entry in block.entries.items():
                if entry.members is None and entry.name.startswith(prefix):
                    items[entry.name[len(prefix) :]] = block.symbols[variable]
                    references.update(
                        {
                            item: block.references[item]
                            for item in ProblemSpec._real_components_of(block.symbols[variable])
                        },
                    )
        namespace = ObservableEvaluator._tree(items)
        for other, (other_scope, earlier) in self.live.items():
            if other == path:
                break
            if other_scope == scope:
                value, other_references = self._value(other, other_scope, earlier)
                setattr(namespace, earlier.label, value)
                references.update(other_references)
        expression = observable.expression(namespace) if callable(observable.expression) else observable.expression
        return expression, references

    def _value(
        self,
        path: str,
        scope: str,
        observable: ObservableSpec,
    ) -> tuple[object, dict[Expression, ReferenceSpec]]:
        """Return the value of an observable with the references of the symbols it reads."""
        if path in self.values_:
            return self.values_[path]
        if observable.group is None:
            self.values_[path] = self._system_value(path, scope, observable)
            return self.values_[path]
        block = self.declarations.blocks[ObservableEvaluator._join(scope, observable.group)]
        source = self._source_of(block)
        references = {item: replace(reference, source=source) for item, reference in block.references.items()}
        self.values_[path] = (self._reduced_value(path, observable, block), references)
        return self.values_[path]

    @staticmethod
    def _source_of(block: BlockSpec) -> ReadKind:
        """Return how laws read the variables of a block from outside it, per member for a group."""
        return ReadKind.MEMBER if block.node.members is not None else ReadKind.SINGLE

    @staticmethod
    def _namespace_of(block: BlockSpec) -> SimpleNamespace:
        """Return the namespace of the variables of a block, with tuples for per-entry parameters."""
        return SimpleNamespace(
            **{name: tuple(value) if name in block.per else value for name, value in block.symbols.items()},
        )

    @staticmethod
    def _reduced_value(path: str, observable: ObservableSpec, block: BlockSpec) -> object:
        """Return an observable that reduces a group as a reduction over the group's member symbols."""
        namespace = Observer._namespace_of(block)
        if observable.reduce == 'count':
            return count(over=namespace)
        if observable.variable is not None:
            if observable.variable not in block.symbols:
                error_msg = (
                    f'ObservableSpec {path} reduces {observable.variable}, which {block.node.path} does not declare.'
                )
                raise ValidationError(error_msg)
            value: object = block.symbols[observable.variable]
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

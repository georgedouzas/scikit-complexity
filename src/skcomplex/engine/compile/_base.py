"""Define the bases of compilation: the compiler of a system, the passes of a compilation, and the lowerings of laws."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import copy
import dataclasses
import functools
from collections.abc import Callable, Iterator, Mapping, Sequence
from types import SimpleNamespace
from typing import TYPE_CHECKING, Self, cast

from sage.all__sagemath_symbolics import Expression

from ...base import BaseConfig
from ...core.laws import BaseLaw
from ...core.system import System
from ...errors import LabelError, ParameterError, RoleError, SkcomplexError, SpaceError, ValidationError
from ...specs import BaseProblemSpec, IssueSpec, ObservableSpec, ValidationReportSpec
from ..backends import BaseBackend, NumbaBackend, NumpyBackend
from ..backends._base import _AUTO_MEMBERS

if TYPE_CHECKING:
    from ._compilation import Compilation

_FIXES: dict[type[SkcomplexError], tuple[str, str]] = {
    LabelError: ('label', 'Give every child of a system a unique label of letters, digits, and underscores.'),
    RoleError: ('role', 'Bind every role to the path of an element, group, or member of the system.'),
    SpaceError: ('space-missing', 'Give the system a space whose kind the component supports.'),
    ParameterError: ('parameter', 'Give the parameter a value inside its domain.'),
}


class BaseCompiler(BaseConfig):
    """Compiler of a system into a problem of one mathematical form.

    `check()` validates the settings, `prepare(system)` validates the system and sets `claims_`, whether the compiler
    compiles the system, and `report_`, the issues it finds, and `run(initial_state)` raises the errors of the report
    and returns the problem. A subclass implements the hooks `_claims`, `_issues`, and `_compile`.

    Args:
        backend:
            The backend instance, or `None` for the backend that suits the size of the system.

        solver:
            The solver the problem runs on, or `None` for the default solver of its form, whose limits compiling
            checks.

        observables:
            The observables to compute at every frame, labelled by their path from the root, such as
            `village/mean_wealth` for an observable of the nested system `village`.

    Attributes:
        system_:
            The prepared system, with the observables of the run.

        claims_:
            Whether the compiler compiles the prepared system.

        report_:
            The issues of the prepared system.
    """

    system_: System
    claims_: bool
    report_: ValidationReportSpec

    def __init__(
        self,
        backend: BaseBackend | None = None,
        solver: object = None,
        observables: Sequence[ObservableSpec] = (),
    ) -> None:
        self.backend = backend
        self.solver = solver
        self.observables = observables

    def _check(self) -> Self:
        """Raise an error for a backend that is not a backend or an observable that is not an observable spec.

        Returns:
            self:
                The compiler.

        Raises:
            ValidationError:
                A setting has the wrong type.
        """
        if self.backend is not None and not isinstance(self.backend, BaseBackend):
            error_msg = f'The backend is a {type(self.backend).__name__}. Pass a backend instance or `None`.'
            raise ValidationError(error_msg)
        wrong = [item for item in self.observables if not isinstance(item, ObservableSpec)]
        if wrong:
            error_msg = f'The observables hold a {type(wrong[0]).__name__}. Pass observable specs.'
            raise ValidationError(error_msg)
        return self

    def _prepare(self, context: object) -> Self:
        """Attach the observables to a system, decide whether the compiler claims it, and set the issues it finds.

        Args:
            context:
                The system.

        Returns:
            self:
                The prepared compiler.

        Raises:
            ValidationError:
                The context is not a system.
        """
        if not isinstance(context, System):
            error_msg = f'A compiler prepares on a system, not on a {type(context).__name__}.'
            raise ValidationError(error_msg)
        system = context
        try:
            self.system_ = self._with_observables(system, self.observables)
        except ValidationError as error:
            self.system_ = system
            self.claims_ = self._claims(system)
            self.report_ = ValidationReportSpec(
                (IssueSpec('observable-path', system.label, str(error), 'Label observables by their path.'),),
            )
            return self
        self.claims_ = self._claims(self.system_)
        self.report_ = ValidationReportSpec(tuple(self._issues(self.system_)))
        return self

    def _claims(self, system: System) -> bool:
        """Check whether the compiler compiles a system.

        Args:
            system:
                The system, with the observables of the run.

        Raises:
            NotImplementedError:
                The subclass does not implement the hook.
        """
        raise NotImplementedError

    def _issues(self, system: System) -> list[IssueSpec]:
        """Return the issues of a system that the compiler finds before compiling it.

        Args:
            system:
                The system, with the observables of the run.

        Returns:
            issues:
                The issues, in the order they were found.
        """
        return []

    def _compile(self, initial_state: Mapping[str, object]) -> BaseProblemSpec:
        """Compile the prepared system into a problem.

        Args:
            initial_state:
                Initial values that replace those of the components, by the dotted name of the state.

        Raises:
            NotImplementedError:
                The subclass does not implement the hook.
        """
        raise NotImplementedError

    def _run(self, inputs: object) -> BaseProblemSpec:
        """Raise the errors of the report and compile the prepared system into a problem.

        Args:
            inputs:
                The initial values that replace those of the components, by the dotted name of the state, such as
                `{'rabbits.n': 20}`, or `None`.

        Returns:
            problem:
                The compiled problem.

        Raises:
            ValidationError:
                The report holds an error, the compiler does not claim the system, or the initial values are not a
                mapping.
        """
        self.report_.check()
        if not self.claims_:
            error_msg = f'{type(self).__name__} does not compile the system {self.system_.label}. Use compile_system.'
            raise ValidationError(error_msg)
        if inputs is not None and not isinstance(inputs, Mapping):
            error_msg = f'The initial state is a {type(inputs).__name__}. Pass a mapping of dotted state names.'
            raise ValidationError(error_msg)
        problem = self._compile(dict(inputs or {}))
        observables = tuple(self._observables_of(problem.system)) if problem.system is not None else ()
        return dataclasses.replace(problem, observables=observables)

    @staticmethod
    def _resolve_backend(backend: BaseBackend | None, members: int) -> BaseBackend:
        """Return the backend passed, or the backend that suits the size of a system when none is passed.

        Without a backend, `NumbaBackend` runs systems whose groups hold at least 1000 members in total, and
        `NumpyBackend` the others.

        Args:
            backend:
                The backend instance, or `None`.

            members:
                The total number of members of the system's groups.

        Returns:
            backend:
                The checked backend passed, or a new checked backend.

        Examples:
            >>> from skcomplex.engine.compile import BaseCompiler
            >>> type(BaseCompiler._resolve_backend(None, 10)).__name__
            'NumpyBackend'
        """
        if backend is not None:
            return backend.check()
        return (NumbaBackend() if members >= _AUTO_MEMBERS else NumpyBackend()).check()

    @staticmethod
    def _element_value(element: str, expression: Callable[[object], object], namespace: SimpleNamespace) -> object:
        """Return the value of an expression of an element's namespace and the time, read from its system."""
        return expression(SimpleNamespace(**vars(getattr(namespace, element)), t=namespace.t))

    @staticmethod
    def _of_root(observable: ObservableSpec) -> ObservableSpec:
        """Return an observable of an element as the observable of the root system labelled `<element>.<label>`."""
        if observable.element is None or not callable(observable.expression):
            return observable
        expression = functools.partial(BaseCompiler._element_value, observable.element, observable.expression)
        return dataclasses.replace(
            observable,
            label=f'{observable.element}.{observable.label}',
            expression=expression,
            element=None,
        )

    @staticmethod
    def _observables_of(system: System, scope: str = '') -> Iterator[tuple[str, ObservableSpec]]:
        """Yield every observable of a system tree, with the path of the system that declares it.

        Args:
            system:
                The root system.

            scope:
                The path of the system, empty for the root.

        Yields:
            observable:
                The path of the declaring system and the observable.
        """
        for observable in getattr(system, 'observables', ()):
            yield scope, observable
        for child in system.systems:
            yield from BaseCompiler._observables_of(child, f'{scope}/{child.label}' if scope else child.label)

    @staticmethod
    def _system_paths(system: System, scope: str = '') -> Iterator[str]:
        """Yield the path of every nested system."""
        for child in system.systems:
            path = f'{scope}/{child.label}' if scope else child.label
            yield path
            yield from BaseCompiler._system_paths(child, path)

    @staticmethod
    def _attached(system: System, by_scope: dict[str, list[ObservableSpec]], scope: str) -> System:
        """Return a copy of a system that holds the observables of its scope, with its nested systems attached."""
        copied = copy.copy(system)
        copied.systems = [
            BaseCompiler._attached(child, by_scope, f'{scope}/{child.label}' if scope else child.label)
            for child in system.systems
        ]
        copied.observables = [*getattr(system, 'observables', ()), *by_scope.get(scope, ())]
        return copied

    @staticmethod
    def _with_observables(system: System, observables: Sequence[ObservableSpec]) -> System:
        """Return a copy of a system tree that declares observables in the systems their labels name.

        Args:
            system:
                The system.

            observables:
                The observables, labelled by their path from the root.

        Returns:
            system:
                The system tree with the observables, or the system itself without observables.

        Raises:
            ValidationError:
                An observable names a nested system that does not exist.
        """
        by_scope: dict[str, list[ObservableSpec]] = {}
        for observable in observables:
            checked = BaseCompiler._of_root(observable.check())
            scope, _, label = checked.label.rpartition('/')
            by_scope.setdefault(scope, []).append(dataclasses.replace(checked, label=label))
        if not by_scope:
            return system
        paths = {''} | set(BaseCompiler._system_paths(system))
        unknown = sorted(set(by_scope) - paths)
        if unknown:
            error_msg = (
                f'The observables name the nested systems {", ".join(unknown)}, which do not exist. Label an '
                'observable by its path from the root, such as `village/mean_wealth`.'
            )
            raise ValidationError(error_msg)
        return BaseCompiler._attached(system, by_scope, '')

    @staticmethod
    def _issue_of(error: SkcomplexError, path: str, code: str, fix: str) -> IssueSpec:
        """Return the issue of an error, with the code and fix of its class or the ones given.

        Args:
            error:
                The error.

            path:
                The path of the component or system it is in.

            code:
                The code of the issue when the class of the error has none.

            fix:
                The fix of the issue when the class of the error has none.

        Returns:
            issue:
                The issue.
        """
        code, fix = _FIXES.get(type(error), (code, fix))
        return IssueSpec(code, path, str(error), fix, error=type(error))


class BasePass(BaseConfig):
    """Step of a compilation that reads fields of the compilation and writes others.

    A pass declares the fields it reads and writes through the hooks `_get_reads` and `_get_writes`, which order the
    passes of a compilation. `prepare(compilation)` takes the compilation the pass serves, and `run()` calls the hook
    `_run`, which reads the fields the pass reads from `compilation_` and writes the fields it writes. A pass takes no
    input to `run`: the compilation it changes is its prepared target.

    Attributes:
        compilation_:
            The compilation the pass reads and writes.
    """

    compilation_: 'Compilation'

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads; none by default."""
        return ()

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes; none by default."""
        return ()

    def _prepare(self, context: object) -> Self:
        """Take the compilation the pass reads and writes.

        Args:
            context:
                The compilation.

        Returns:
            self:
                The prepared pass.
        """
        self.compilation_ = cast('Compilation', context)
        return self

    def _run(self, inputs: object = None) -> None:
        """Read the fields of the compilation the pass reads and write the fields it writes.

        Args:
            inputs:
                Nothing: the pass changes its prepared compilation.

        Raises:
            NotImplementedError:
                The subclass does not implement the hook.
        """
        raise NotImplementedError

    @staticmethod
    def _replaced[L: BaseLaw](law: L, **changes: object) -> L:
        """Return a new law of the class of a law, with some of its arguments changed.

        Args:
            law:
                The law.

            **changes:
                The new values of arguments, by name.

        Returns:
            law:
                The new law, which the compiler reads without checking it again.

        Examples:
            >>> from sage.all__sagemath_symbolics import var
            >>> from skcomplex.core.laws import RateLaw
            >>> from skcomplex.engine.compile import BasePass
            >>> x = var('x')
            >>> BasePass._replaced(RateLaw(x, -x), value=-2 * x).value
            -2*x
        """
        return type(law)(**{**law.get_params(deep=False), **changes})

    @staticmethod
    def _calls_of(expression: Expression, name: str) -> list[Expression]:
        """Return the calls of a symbolic function by name inside an expression.

        Args:
            expression:
                The expression.

            name:
                The name of the function.

        Returns:
            calls:
                The calls, nested calls included.
        """
        found, stack = [], [expression]
        while stack:
            item = stack.pop()
            if item.operator() is None:
                continue
            if getattr(item.operator(), 'name', lambda: '')() == name:
                found.append(item)
            stack.extend(item.operands())
        return found


class BaseLowering(BaseConfig):
    """Lowering of one family of laws, which one pass creates, later passes finish, and the compilation keeps.

    The constructor stores what the lowering reads, `prepare(compilation)` sets the prepared attributes the lowering
    gathers, and `run(laws)` returns the laws lowered. A subclass implements the hooks `_prepare` and `_run`.

    Attributes:
        compilation_:
            The compilation the lowering serves.
    """

    compilation_: 'Compilation'

    def _prepare(self, context: object) -> Self:
        """Take the compilation the lowering serves.

        Args:
            context:
                The compilation.

        Returns:
            self:
                The prepared lowering.
        """
        self.compilation_ = cast('Compilation', context)
        return self

    def _run(self, inputs: object) -> list[BaseLaw]:
        """Return laws lowered.

        Args:
            inputs:
                The laws.

        Raises:
            NotImplementedError:
                The subclass does not implement the hook.
        """
        raise NotImplementedError

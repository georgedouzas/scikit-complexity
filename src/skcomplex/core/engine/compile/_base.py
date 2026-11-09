"""Define the bases of compilation.

They are the compiler of a system, the passes of a compilation, and the lowerings of laws.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import copy
import dataclasses
import functools
from collections.abc import Callable, Iterator, Mapping, Sequence
from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression

from ...base import BaseConfig
from ...components.laws import BaseLaw
from ...components.systems import CompositeSystem
from ...errors import (
    LabelError,
    ParameterError,
    RoleElementError,
    RoleError,
    SkcomplexError,
    SpaceError,
    TensorShapeError,
    TensorSymmetryError,
    ValidationError,
)
from ...specs import IssueSpec, ObservableSpec, SupportSpec, ValidationReportSpec
from ...types import FormKind
from ..backends import BaseBackend, NumbaBackend, NumpyBackend
from ..backends._base import _AUTO_MEMBERS
from ..problems import BaseProblemSpec
from ..problems._kernel_spec import KernelSpec
from ..problems._reference_spec import ReferenceSpec
from ._compilation import Compilation

_FIXES: dict[type[SkcomplexError], tuple[str, str]] = {
    LabelError: ('label', 'Give every child of a system a unique label of letters, digits, and underscores.'),
    RoleError: ('role', 'Bind every role to the path of an element, group, or member of the system.'),
    RoleElementError: (
        'role-element',
        (
            'Bind the role to an element that carries the state, declare the state on the element, or read another '
            'state. `get_requirements()` of an interaction and `get_states()` of an element show what they need and '
            'carry.'
        ),
    ),
    SpaceError: ('space-missing', 'Give the system a space whose kind the component supports.'),
    ParameterError: ('parameter', 'Give the parameter a value inside its domain.'),
    TensorShapeError: ('law-domain', 'Give the tensor operation a matrix or a vector of the shape it reads.'),
    TensorSymmetryError: ('tensor-symmetry', 'Give the state a rate of its declared symmetry.'),
}


class BaseCompiler(BaseConfig[CompositeSystem, Mapping[str, object]]):
    """Compiler of a system into a problem of one mathematical form.

    `prepare(system)` validates the settings and the system and sets `report_`, the issues it finds,
    and `run(initial_state)` raises the errors of the report and sets `problem_`, the compiled problem, and `solver_`,
    the solver the problem runs on. A subclass implements the hooks `_issues` and `_compile`, and a compiler of a form
    the engine does not know names the solver of its problems in `_get_solver`.

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

        report_:
            The issues of the prepared system.

        problem_:
            The compiled problem.

        solver_:
            The solver the compiled problem runs on: the solver setting, the solver the compiler names, or `None` for
            the default solver of the problem's form.
    """

    system_: CompositeSystem
    report_: ValidationReportSpec
    problem_: BaseProblemSpec
    solver_: object

    def __init__(
        self,
        backend: BaseBackend | None = None,
        solver: object = None,
        observables: Sequence[ObservableSpec] = (),
    ) -> None:
        self.backend = backend
        self.solver = solver
        self.observables = observables

    def _prepare(self, context: CompositeSystem | None) -> None:
        """Attach the observables to a system and set the issues it finds.

        It first raises an error for a wrong backend or observable. The backend must be a backend and every observable
        an observable spec.

        Args:
            context:
                The system.

        Raises:
            ValidationError:
                The context is not a system. A setting has the wrong type.
        """
        if self.backend is not None and not isinstance(self.backend, BaseBackend):
            error_msg = f'The backend is a {type(self.backend).__name__}. Pass a backend instance or `None`.'
            raise ValidationError(error_msg)
        wrong = [item for item in self.observables if not isinstance(item, ObservableSpec)]
        if wrong:
            error_msg = f'The observables hold a {type(wrong[0]).__name__}. Pass observable specs.'
            raise ValidationError(error_msg)
        if not isinstance(context, CompositeSystem):
            error_msg = f'A compiler prepares on a system, not on a {type(context).__name__}.'
            raise ValidationError(error_msg)
        system = context.prepare().system_ if context._nests_classes() else context
        try:
            self.system_ = self._with_observables(system, self.observables)
        except ValidationError as error:
            self.system_ = system
            self.report_ = ValidationReportSpec(
                (IssueSpec('observable-path', system.label, str(error), 'Label observables by their path.'),),
            )
            return
        self.report_ = ValidationReportSpec(tuple(self._issues(self.system_)))

    def _issues(self, system: CompositeSystem) -> list[IssueSpec]:
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

        Returns:
            problem:
                The compiled problem.

        Raises:
            NotImplementedError:
                The subclass does not implement the hook.
        """
        raise NotImplementedError

    def _get_solver(self, problem: BaseProblemSpec) -> object:
        """Return the solver of a compiled problem.

        It is `None` for the default solver of its form. A compiler of a form the engine does not know, such as one
        defined in a system package, names its solver here.

        Args:
            problem:
                The compiled problem.

        Returns:
            solver:
                `None` here.
        """
        del problem
        return None

    def _run(self, inputs: Mapping[str, object] | None) -> None:
        """Raise the errors of the report and compile the prepared system.

        The compiled problem goes to `problem_` and its solver to `solver_`.

        Args:
            inputs:
                The initial values that replace those of the components, by the dotted name of the state, such as
                `{'rabbits.n': 20}`, or `None`.

        Raises:
            ValidationError:
                The report holds an error, or the initial values are not a mapping.
        """
        self.report_._check()
        if inputs is not None and not isinstance(inputs, Mapping):
            error_msg = f'The initial state is a {type(inputs).__name__}. Pass a mapping of dotted state names.'
            raise ValidationError(error_msg)
        problem = self._compile(dict(inputs or {}))
        observables = tuple(self._observables_of(problem.system)) if problem.system is not None else ()
        self.problem_ = dataclasses.replace(problem, observables=observables)
        self.solver_ = self.solver if self.solver is not None else self._get_solver(self.problem_)

    @staticmethod
    def _support_of(runner: type) -> SupportSpec:
        """Return the support of a solver or backend class.

        The support names what the class runs.

        Args:
            runner:
                The class of the solver or the backend.

        Returns:
            support:
                The times, spaces, laws, noise, and orders the class runs.

        Raises:
            ValidationError:
                The class is not a solver or a backend of the engine.
        """
        get_support = getattr(runner, '_get_support', None)
        support = get_support() if callable(get_support) else None
        if not isinstance(support, SupportSpec):
            error_msg = f'{runner.__name__} is not a solver or a backend of the engine, which name what they run.'
            raise ValidationError(error_msg)
        return support

    @staticmethod
    def _form_of(solver: type) -> FormKind:
        """Return the form a solver class solves.

        Args:
            solver:
                The class of the solver.

        Returns:
            form:
                The form the class solves.

        Raises:
            ValidationError:
                The class is not a solver of the engine.
        """
        get_form = getattr(solver, '_get_form', None)
        form = get_form() if callable(get_form) else None
        if not isinstance(form, FormKind):
            error_msg = f'{solver.__name__} is not a solver of the engine, which names the form it solves.'
            raise ValidationError(error_msg)
        return form

    @staticmethod
    def _resolve_backend(backend: BaseBackend | None, members: int) -> BaseBackend:
        """Return the backend passed, or the one that suits the size of a system.

        The second applies when no backend is passed. Without a backend, `NumbaBackend` runs systems whose groups hold
        at least 1000 members in total, and `NumpyBackend` the others.

        Args:
            backend:
                The backend instance, or `None`.

            members:
                The total number of members of the system's groups.

        Returns:
            backend:
                The backend passed, or a new backend, which checks its settings when it is prepared.

        Examples:
            >>> from skcomplex.core.engine.compile import BaseCompiler
            >>> type(BaseCompiler._resolve_backend(None, 10)).__name__
            'NumpyBackend'
        """
        if backend is not None:
            return backend
        return NumbaBackend() if members >= _AUTO_MEMBERS else NumpyBackend()

    @staticmethod
    def _element_value(element: str, expression: Callable[[object], object], namespace: SimpleNamespace) -> object:
        """Return the value of an expression of an element's namespace and the time.

        The value is read from the system of the element.
        """
        return expression(SimpleNamespace(**vars(getattr(namespace, element)), t=namespace.t))

    @staticmethod
    def _of_root(observable: ObservableSpec) -> ObservableSpec:
        """Return an observable of an element as an observable of the root system.

        The observable of the root system is labelled `<element>.<label>`.
        """
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
    def _observables_of(system: CompositeSystem, scope: str = '') -> Iterator[tuple[str, ObservableSpec]]:
        """Yield every observable of a system tree.

        Each comes with the path of the system that declares it.

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
    def _system_paths(system: CompositeSystem, scope: str = '') -> Iterator[str]:
        """Yield the path of every nested system."""
        for child in system.systems:
            path = f'{scope}/{child.label}' if scope else child.label
            yield path
            yield from BaseCompiler._system_paths(child, path)

    @staticmethod
    def _attached(system: CompositeSystem, by_scope: dict[str, list[ObservableSpec]], scope: str) -> CompositeSystem:
        """Return a copy of a system that holds the observables of its scope.

        The copy has its nested systems attached.
        """
        copied = copy.copy(system)
        copied.systems = [
            BaseCompiler._attached(child, by_scope, f'{scope}/{child.label}' if scope else child.label)
            for child in system.systems
        ]
        copied.observables = [*getattr(system, 'observables', ()), *by_scope.get(scope, ())]
        return copied

    @staticmethod
    def _with_observables(system: CompositeSystem, observables: Sequence[ObservableSpec]) -> CompositeSystem:
        """Return a copy of a system tree with observables in the systems they name.

        Every observable is declared in the system its label names.

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
            observable._check()
            checked = BaseCompiler._of_root(observable)
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
        """Return the issue of an error.

        The issue holds the code and fix of the error class, or the ones given.

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


class BasePass(BaseConfig[Compilation, None]):
    """Step of a compilation that reads fields of the compilation and writes others.

    A pass declares the fields it reads and writes through the hooks `_get_reads` and `_get_writes`, which order the
    passes of a compilation. `prepare(compilation)` takes the compilation the pass serves, and `run()` calls the hook
    `_run`, which reads the fields the pass reads from `compilation_` and writes the fields it writes. A pass takes no
    input to `run`: the compilation it changes is its prepared target.

    Attributes:
        compilation_:
            The compilation the pass reads and writes.
    """

    compilation_: Compilation

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads.

        By default, none.
        """
        return ()

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes.

        By default, none.
        """
        return ()

    def _prepare(self, context: Compilation | None) -> None:
        """Take the compilation the pass reads and writes.

        Args:
            context:
                The compilation.

        Raises:
            ValidationError:
                The context is not a compilation.
        """
        if not isinstance(context, Compilation):
            error_msg = f'A pass prepares on a compilation, not on a {type(context).__name__}.'
            raise ValidationError(error_msg)
        self.compilation_ = context

    def _run(self, inputs: None = None) -> None:
        """Read and write the fields of the compilation the pass uses.

        The pass reads some fields of its compilation and writes others.

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
            >>> from skcomplex.core.components.laws import RateLaw
            >>> from skcomplex.core.engine.compile import BasePass
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


class BaseLowering[LoweredT, InputT](BaseConfig[Compilation, InputT]):
    """Lowering of one family of laws.

    One pass creates the lowering, later passes finish it, and the compilation keeps it. The constructor stores what the
    lowering reads, `prepare(compilation)` sets the prepared attributes the lowering gathers, and `run(laws)` sets
    `lowered_`, the laws lowered. A subclass implements the hooks `_prepare` and `_run`, and names the type of the laws
    it lowers and of the laws it runs on, which are the same for most lowerings.

    A lowering that reads new symbols into laws notes their references with `_add_reference`, reads them back with
    `_reference_of`, and `_apply_references` replaces every spec it noted references for by a copy that holds them, in
    every field of the compilation.

    Attributes:
        compilation_:
            The compilation the lowering serves.

        added_:
            The spec and the references the lowering added to it, by the identity of the spec.

        lowered_:
            The laws the last run lowered, in the shape of its input.
    """

    compilation_: Compilation
    added_: dict[int, tuple[KernelSpec, dict[Expression, ReferenceSpec]]]
    lowered_: LoweredT

    def _prepare(self, context: Compilation | None) -> None:
        """Take the compilation the lowering serves, with no reference added yet.

        Args:
            context:
                The compilation.

        Raises:
            ValidationError:
                The context is not a compilation.
        """
        if not isinstance(context, Compilation):
            error_msg = f'A lowering prepares on a compilation, not on a {type(context).__name__}.'
            raise ValidationError(error_msg)
        self.compilation_ = context
        self.added_ = {}

    def _add_reference(self, spec: KernelSpec, item: Expression, reference: ReferenceSpec) -> None:
        """Note the reference of a symbol the lowering adds to a spec.

        `_apply_references` gives the spec the references noted.
        """
        self.added_.setdefault(id(spec), (spec, {}))[1][item] = reference

    def _reference_of(self, item: object, spec: KernelSpec) -> ReferenceSpec | None:
        """Return the reference of a symbol of a spec, or `None`.

        The references the lowering added are included.
        """
        _, added = self.added_.get(id(spec), (spec, {}))
        return added[item] if item in added else spec.references.get(item)

    def _apply_references(self) -> None:
        """Replace every spec the lowering added references to by a copy with them.

        The copies replace the specs in the compilation.
        """
        replaced = {
            key: dataclasses.replace(spec, references={**spec.references, **added})
            for key, (spec, added) in self.added_.items()
        }
        self.added_ = {}
        self._replace_specs(replaced)
        self.compilation_._replace_specs(replaced)

    def _replace_specs(self, specs: Mapping[int, KernelSpec]) -> None:
        """Replace the kernel specs the lowering holds, none here.

        Args:
            specs:
                The new spec of every replaced spec, by the identity of the replaced spec.
        """
        del specs

    def _run(self, inputs: InputT | None) -> None:
        """Lower laws into `lowered_`, in the shape of the input.

        Args:
            inputs:
                The laws, alone or by the kernel spec that holds them.

        Raises:
            NotImplementedError:
                The subclass does not implement the hook.
        """
        raise NotImplementedError

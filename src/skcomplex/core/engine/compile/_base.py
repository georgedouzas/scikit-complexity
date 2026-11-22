"""Define the bases of compilers, passes, and lowerings."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import copy
import dataclasses
import functools
from collections.abc import Callable, Iterator, Mapping, Sequence
from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression

from ...base import BaseConfig
from ...components import Observable
from ...components.laws import BaseLaw
from ...enums import FormKind
from ...errors import (
    LabelError,
    ParameterError,
    ParameterShapeError,
    RoleElementError,
    RoleError,
    SkcomplexError,
    SpaceError,
    TensorShapeError,
    TensorSymmetryError,
    ValidationError,
)
from ..backends import BaseBackend, NumbaBackend, NumpyBackend, SupportSpec
from ..backends._base import AUTO_MEMBERS
from ..problems import BaseProblemSpec
from ..problems._composite_system import _CompositeSystem
from ..problems._kernel_spec import KernelSpec
from ..problems._reference_spec import ReferenceSpec
from ._compilation import Compilation
from ._issue_spec import IssueSpec
from ._validation_report_spec import ValidationReportSpec

FIXES: dict[type[SkcomplexError], tuple[str, str]] = {
    LabelError: ('label', 'Give every child of a system a unique label of letters, digits, and underscores.'),
    RoleError: ('role', 'Bind every role to the path of an element, group, or member of the system.'),
    RoleElementError: (
        'role-element',
        (
            'Bind the role to an element that has the value, give the element the value, such as '
            "`Element('ball', values={'x': 0.0})`, or read another value. `get_requirements()` of an interaction shows "
            'the values its roles read.'
        ),
    ),
    SpaceError: ('space-missing', 'Give the system a space whose kind the component supports.'),
    ParameterError: ('parameter', 'Give the parameter a value inside its domain.'),
    ParameterShapeError: ('parameter-shape', 'Bind the roles of the axes of the array to groups of their lengths.'),
    TensorShapeError: ('law-domain', 'Give the tensor operation a matrix or a vector of the shape it reads.'),
    TensorSymmetryError: ('tensor-symmetry', 'Give the state a rate of its declared symmetry.'),
}

type _Composite = _CompositeSystem


class BaseCompiler(BaseConfig[_Composite, Mapping[str, object]]):
    """An abstract compiler that turns a system into a problem that a solver runs.

    `prepare(system)` checks the settings and the system and lists what is wrong in `report_`, and `run(initial_state)`
    raises the errors of the report and sets `problem_`, the compiled problem, and `solver_`, the solver that runs it.
    A subclass fills in the hooks `_issues` and `_compile`, and a compiler of a kind of problem that the engine does not
    know names the solver of its problems in `_get_solver`. `PassCompiler` compiles every system of the catalog, and a
    field package writes its own compiler on this base, as the quantum package does for operators.

    Read more in the user guide, under Extend the library.

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

    See Also:
        PassCompiler: Compiles a system through a sequence of passes.
        OperatorCompiler: Compiles a system of operator elements.

    Examples:
        >>> from skcomplex.core.engine.compile import BaseCompiler
        >>> from skcomplex.core.engine.compile._pass_compiler import PassCompiler
        >>> issubclass(PassCompiler, BaseCompiler)
        True
    """

    system_: _Composite
    report_: ValidationReportSpec
    problem_: BaseProblemSpec
    solver_: object

    def __init__(
        self,
        backend: BaseBackend | None = None,
        solver: object = None,
        observables: Sequence[Observable] = (),
    ) -> None:
        self.backend = backend
        self.solver = solver
        self.observables = observables

    def _prepare(self, context: _Composite | None) -> None:
        """Attach the observables to a system and set the issues it finds.

        It first raises an error for a wrong backend or observable. The backend must be a backend and every observable
        an observable spec. A `System` or a ready-made system of the catalog compiles the composite of its
        components.

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
        wrong = [item for item in self.observables if not isinstance(item, Observable)]
        if wrong:
            error_msg = f'The observables hold a {type(wrong[0]).__name__}. Pass observable specs.'
            raise ValidationError(error_msg)
        get_composite = getattr(context, '_get_composite', None)
        if not isinstance(context, _CompositeSystem) and callable(get_composite):
            context = get_composite()
        if not isinstance(context, _CompositeSystem):
            error_msg = f'A compiler prepares on a system, not on a {type(context).__name__}.'
            raise ValidationError(error_msg)
        system = context.prepare().system_ if context._nests_classes() else context
        try:
            system = self._copy_with_observables(system, self.observables)
        except ValidationError as error:
            self.system_ = self._get_declared_system(system)
            self.report_ = ValidationReportSpec(
                (IssueSpec('observable-path', system.label, str(error), 'Label observables by their path.'),),
            )
            return
        self.system_ = self._get_declared_system(system)
        roles = self._get_domain_issues(self.system_)
        self.report_ = ValidationReportSpec(tuple(roles or self._get_issues(self.system_)))

    def _get_domain_issues(self, system: _CompositeSystem, scope: str = '') -> list[IssueSpec]:
        """Return the issue of a role bound to a value of a domain it does not accept.

        The values of plain elements have their domains only once the laws show which are states, so a role such as
        `Annotated[Role, 'boolean']` is checked on the declared system, whatever kind of law the interaction sets.
        """
        try:
            system._check_roles()
        except RoleElementError as error:
            return [self._get_issue_of(error, scope or system.label, 'role', 'Bind the role to values of its domains.')]
        for item in system.systems:
            found = self._get_domain_issues(item, f'{scope}{item.label}/') if isinstance(item, _CompositeSystem) else []
            if found:
                return found
        return []

    def _get_declared_system(self, system: _CompositeSystem) -> _CompositeSystem:
        """Return the system with declarations for its plain elements."""
        return system

    def _get_issues(self, system: _Composite) -> list[IssueSpec]:
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
        observables = tuple(self._get_observables_of(problem.system)) if problem.system is not None else ()
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
    def _get_form_of(solver: type) -> FormKind:
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
        """
        if backend is not None:
            return backend
        return NumbaBackend() if members >= AUTO_MEMBERS else NumpyBackend()

    @staticmethod
    def _get_element_value(element: str, expression: Callable[[object], object], namespace: SimpleNamespace) -> object:
        """Return the value of an expression of an element's namespace and the time.

        The value is read from the system of the element.
        """
        return expression(SimpleNamespace(**vars(getattr(namespace, element)), t=namespace.t))

    @staticmethod
    def _get_of_root(observable: Observable) -> Observable:
        """Return an observable of an element as an observable of the root system.

        The observable of the root system is labelled `<element>.<label>`.
        """
        if observable.element is None or not callable(observable.expression):
            return observable
        expression = functools.partial(BaseCompiler._get_element_value, observable.element, observable.expression)
        return dataclasses.replace(
            observable,
            label=f'{observable.element}.{observable.label}',
            expression=expression,
            element=None,
        )

    @staticmethod
    def _get_observables_of(system: _CompositeSystem, scope: str = '') -> Iterator[tuple[str, Observable]]:
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
            yield from BaseCompiler._get_observables_of(child, f'{scope}/{child.label}' if scope else child.label)

    @staticmethod
    def _get_system_paths(system: _CompositeSystem, scope: str = '') -> Iterator[str]:
        """Yield the path of every nested system."""
        for child in system.systems:
            path = f'{scope}/{child.label}' if scope else child.label
            yield path
            yield from BaseCompiler._get_system_paths(child, path)

    @staticmethod
    def _get_attached(
        system: _CompositeSystem,
        by_scope: dict[str, list[Observable]],
        scope: str,
    ) -> _CompositeSystem:
        """Return a copy of a system that holds the observables of its scope.

        The copy has its nested systems attached.
        """
        copied = copy.copy(system)
        copied.systems = [
            BaseCompiler._get_attached(child, by_scope, f'{scope}/{child.label}' if scope else child.label)
            for child in system.systems
        ]
        copied.observables = [*getattr(system, 'observables', ()), *by_scope.get(scope, ())]
        return copied

    @staticmethod
    def _copy_with_observables(system: _CompositeSystem, observables: Sequence[Observable]) -> _CompositeSystem:
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
        by_scope: dict[str, list[Observable]] = {}
        for observable in observables:
            observable._check()
            checked = BaseCompiler._get_of_root(observable)
            scope, _, label = checked.label.rpartition('/')
            by_scope.setdefault(scope, []).append(dataclasses.replace(checked, label=label))
        if not by_scope:
            return system
        paths = {''} | set(BaseCompiler._get_system_paths(system))
        unknown = sorted(set(by_scope) - paths)
        if unknown:
            error_msg = (
                f'The observables name the nested systems {", ".join(unknown)}, which do not exist. Label an '
                'observable by its path from the root, such as `village/mean_wealth`.'
            )
            raise ValidationError(error_msg)
        return BaseCompiler._get_attached(system, by_scope, '')

    @staticmethod
    def _get_issue_of(error: SkcomplexError, path: str, code: str, fix: str) -> IssueSpec:
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
        code, fix = FIXES.get(type(error), (code, fix))
        return IssueSpec(code, path, str(error), fix, error=type(error))


class BasePass(BaseConfig[Compilation, None]):
    """An abstract pass, one step of a compilation that reads fields and writes others.

    A compilation turns a system into a problem in small steps, and each step is a pass, such as the pass that finds
    the states of every element or the pass that writes the equations. A pass names the fields of the compilation it
    reads and writes in the hooks `_get_reads` and `_get_writes`, which set the order in which the passes run.
    `prepare(compilation)` takes the compilation, and `run()` calls the hook `_run`, which reads its fields from
    `compilation_` and writes the new ones. A field package adds a pass on this base to compile laws of its own.

    Read more in the user guide, under Extend the library.

    Attributes:
        compilation_:
            The compilation the pass reads and writes.

    See Also:
        Compilation: Holds the fields that passes read and write.
        PassCompiler: Runs the passes over a compilation.

    Examples:
        >>> from skcomplex.core.engine.compile import BasePass, FlattenPass
        >>> issubclass(FlattenPass, BasePass)
        True
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
    def _get_replaced[L: BaseLaw](law: L, **changes: object) -> L:
        """Return a new law of the class of a law, with some of its arguments changed.

        Args:
            law:
                The law.

            **changes:
                The new values of arguments, by name.

        Returns:
            law:
                The new law, which the compiler reads without checking it again.
        """
        return dataclasses.replace(law, **changes)

    @staticmethod
    def _get_checked_law[L: BaseLaw](law: L) -> L:
        """Return a law once its fields are checked.

        Args:
            law:
                The law.

        Returns:
            law:
                The same law.

        Raises:
            ValidationError:
                A field of the law is invalid, named in the message.
        """
        law._check()
        return law

    @staticmethod
    def _get_calls_of(expression: Expression, name: str) -> list[Expression]:
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
    """An abstract lowering, a helper that rewrites one family of laws.

    To lower a law is to rewrite it in simpler terms that the numeric functions can evaluate, such as a random draw read
    as a number stored after the parameters, or a Laplacian read as the values of the neighbouring sites. One pass
    creates the lowering, later passes may finish it, and the compilation keeps it. The constructor stores what the
    lowering reads, `prepare(compilation)` gathers what it needs from the compilation, and `run(laws)` sets `lowered_`,
    the rewritten laws. A subclass fills in the hooks `_prepare` and `_run`. A lowering that adds new symbols to the
    laws notes where they read their values with `_add_reference`, and `_apply_references` gives every changed component
    the new references, in every field of the compilation.

    Read more in the user guide, under Extend the library.

    Attributes:
        compilation_:
            The compilation the lowering serves.

        added_:
            The spec and the references the lowering added to it, by the identity of the spec.

        lowered_:
            The laws the last run lowered, in the shape of its input.

    See Also:
        BasePass: The base of the steps of a compilation.
        DrawLowering: Rewrites random draws as stored numbers.

    Examples:
        >>> from skcomplex.core.engine.compile import BaseLowering
        >>> from skcomplex.core.engine.compile._draw_lowering import DrawLowering
        >>> issubclass(DrawLowering, BaseLowering)
        True
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

    def _get_reference_of(self, item: object, spec: KernelSpec) -> ReferenceSpec | None:
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

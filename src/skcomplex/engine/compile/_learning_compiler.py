"""Compile a system that trains a nested system into a learning problem around the problem of that system."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import dataclasses
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
from sage.all__sagemath_symbolics import SR, Expression
from sklearn.base import clone

from ...core.interactions import TrainingInteraction
from ...core.laws import EquationLaw, GradientStepLaw, RateLaw, UpdateLaw
from ...core.spaces import DiscreteTime
from ...core.system import System
from ...errors import SkcomplexError, SolverError, ValidationError
from ...specs import EntrySpec, IssueSpec, LayoutSpec, ObservableSpec, ParameterSpec
from ...types import FormKind
from .._utils import symbol, symbols_of
from ..backends import BaseBackend, JaxBackend
from ..problems import BaseProblemSpec, ExpressionPrinter, LearningProblemSpec, ProblemSpec
from ..types import TracedArray, TracedFunction
from ._base import BaseCompiler
from ._compilation import Compilation
from ._observables import ObservableEvaluator
from .passes._functions import FunctionsPass

DIFFERENTIABLE = (EquationLaw, RateLaw, UpdateLaw)
INNER_FORMS = (FormKind.DISCRETE, FormKind.ODE)
REDUCTIONS = ('sum', 'mean', 'max', 'min', 'var', 'count')

type TracedRead = Callable[[TracedArray, TracedArray], TracedArray]


class _LearningCompiler(BaseCompiler):
    """Compiler of a system whose training interaction trains a nested system, into a `LearningProblemSpec`.

    It compiles the nested system with the compiler `inner` on the JAX backend, checks the trained parameters, the
    dataset, the loss, and the metrics against the problem of that system, and lowers the loss and the metrics to pure
    JAX functions of its states and parameters.

    Args:
        inner:
            The compiler of the nested system, whose backend it sets to `JaxBackend`.

        backend:
            The backend of the run, which a system that trains does not read, since its nested system always compiles
            on `JaxBackend` to be differentiated.

        solver:
            The solver, which must be `None` or a solver of the learning form.

        observables:
            The observables of the run, which a system that trains does not record.

    Attributes:
        training_:
            The training interaction of the prepared system.

        law_:
            The gradient-step law it returns.

        inner_:
            The problem of the nested system, or `None` before it compiles.

        backend_:
            The JAX backend that compiled the nested system, which keeps its pure function.

        step_:
            The pure JAX function of the nested system, its update or its drift.

        measures_:
            The pure JAX functions of the loss and the metrics, by `'loss'` and the label of every metric.
    """

    training_: TrainingInteraction | None
    law_: GradientStepLaw
    inner_: ProblemSpec | None
    backend_: JaxBackend
    step_: TracedFunction
    measures_: dict[str, TracedFunction]

    def __init__(
        self,
        inner: BaseCompiler,
        backend: BaseBackend | None = None,
        solver: object = None,
        observables: Sequence[ObservableSpec] = (),
    ) -> None:
        super().__init__(backend, solver, observables)
        self.inner = inner

    @staticmethod
    def _find_training(system: System) -> TrainingInteraction | None:
        """Return the training interaction of a system, or `None` for a system that does not train."""
        return next((item for item in system.interactions if isinstance(item, TrainingInteraction)), None)

    @staticmethod
    def _find_system(system: System, path: str) -> System | None:
        """Return the nested system at a path such as `net` or `forest/net`, or `None`."""
        found: System | None = system
        for label in path.split('/'):
            found = next((item for item in found.systems if item.label == label), None) if found is not None else None
        return found

    def _setting_issues(self, system: System, training: TrainingInteraction) -> list[IssueSpec]:
        """Return the issues of the time and the solver of a system that trains."""
        issues = []
        if not isinstance(system.time, DiscreteTime):
            given = type(system.time).__name__ if system.time is not None else 'no time'
            message = f'The GradientStepLaw of {training.label} needs DiscreteTime, one step per epoch, not {given}.'
            issues.append(IssueSpec('law-time', training.label, message, 'Declare `DiscreteTime()` for the system.'))
        if self.solver is not None and self._form_of(type(self.solver)) != FormKind.LEARNING:
            message = (
                f'The solver {type(self.solver).__name__} does not run GradientStepLaw. Use GradientDescentSolver.'
            )
            fix = 'Pass `solver=GradientDescentSolver()`, or no solver.'
            issues.append(IssueSpec('solver-unsupported', system.label, message, fix, error=SolverError))
        return issues

    def _differentiable_issues(self, model: System, training: TrainingInteraction) -> list[IssueSpec]:
        """Return the laws of the nested system that the JAX backend cannot differentiate."""
        compiler = clone(self.inner).check().prepare(model)
        compilation = getattr(compiler, 'compilation_', None)
        census = getattr(compilation, 'census', None)
        records = census.records if census is not None else ()
        issues = []
        for record in records:
            law = record.law
            callable_values = any(callable(item) and not isinstance(item, Expression) for item in law._get_values())
            if isinstance(law, DIFFERENTIABLE) and not callable_values:
                continue
            message = (
                f'The {type(law).__name__} of {record.path} in the nested system {training.model} cannot be '
                'differentiated. Learning needs a discrete map or ordinary differential equations written as symbolic '
                'laws.'
            )
            fix = 'Write the nested system with rate, update, or equation laws of symbolic expressions.'
            issues.append(IssueSpec('learning-not-differentiable', training.label, message, fix))
        return issues

    def _compile_inner(self, model: System, training: TrainingInteraction) -> list[IssueSpec]:
        """Compile the nested system into `inner_`, and return the issues that stop it."""
        self.inner_, self.backend_ = None, JaxBackend()
        compiler = clone(self.inner).set_params(backend=self.backend_).check().prepare(model)
        if not compiler.report_.ok:
            return list(compiler.report_.issues)
        try:
            problem = compiler.run().problem_
        except SkcomplexError as error:
            return [self._issue_of(error, training.label, 'learning-not-differentiable', 'Write symbolic laws.')]
        if not isinstance(problem, ProblemSpec) or problem.form not in INNER_FORMS:
            message = f'The nested system {training.model} compiles to a {problem.form}, which learning cannot unroll.'
            fix = 'Train a discrete map or ordinary differential equations.'
            return [IssueSpec('learning-not-differentiable', training.label, message, fix)]
        self.inner_, self.step_ = problem, self._phased_step(compiler)
        return []

    def _phased_step(self, compiler: BaseCompiler) -> TracedFunction:
        """Return the pure JAX function of the inner problem, running the phases of a step of a discrete map in order.

        A phase runs in the steps `k`, counted from one, for which `k - offset` is a multiple of its period, and reads
        the states the phases before it set, as the update of the problem does.
        """
        compilation = getattr(compiler, 'compilation_', None)
        lowered = compilation.kernels if isinstance(compilation, Compilation) else None
        phases = FunctionsPass._phases_of(compilation, lowered) if compilation is not None and lowered else None
        if compilation is None or lowered is None or phases is None:
            return self.backend_.traced_
        layout = Compilation._required(compilation.layout, 'layout')
        complete = dataclasses.replace(
            lowered,
            form=FormKind.DISCRETE,
            size=layout.ends[True],
            reductions=compilation.reductions,
            draws=compilation.draws,
        )
        updates = []
        for phase, kernels in phases:
            backend = (
                JaxBackend()
                .check()
                .prepare()
                .run(dataclasses.replace(complete, kernels=defaultdict(list, {UpdateLaw: kernels})))
            )
            updates.append((phase.period, phase.offset, backend.traced_))

        def phased(t: object, x: TracedArray, p: TracedArray) -> TracedArray:
            step = jnp.round(jnp.asarray(t)) + 1
            for period, offset, update in updates:
                x = update(t, x, p) if period == 1 else jnp.where((step - offset) % period == 0, update(t, x, p), x)
            return x

        return phased

    @staticmethod
    def _adaptive_names(problem: ProblemSpec) -> set[str]:
        """Return the names of the adaptive parameters of the elements and interactions of a problem."""
        names = set(problem.adaptive_parameters)
        for compiled in problem.interactions:
            for name, spec in compiled.interaction._get_specs().items():
                if isinstance(spec, ParameterSpec) and spec.adaptive:
                    names.add(f'{compiled.path}.{name}')
        return names

    def _trained_issues(self, problem: ProblemSpec, training: TrainingInteraction) -> list[IssueSpec]:
        """Return the trained paths that are not parameters, or not adaptive parameters, of the nested system."""
        adaptive = self._adaptive_names(problem)
        parameters = sorted(adaptive | set(problem.parameters.names))
        issues = []
        for path in training.trained:
            if path in adaptive:
                continue
            if path in problem.parameters.names:
                message = f'The trained parameter {path} of {training.model} is not adaptive, so it cannot change.'
                fix = f'Declare {path} `ParameterSpec(adaptive=True)`.'
                issues.append(IssueSpec('learning-not-adaptive', training.label, message, fix))
                continue
            message = (
                f'{training.label} trains {path}, which is not a parameter of {training.model}. Its parameters are '
                f'{", ".join(parameters)}.'
            )
            fix = 'Name an adaptive parameter of the nested system by its dotted path.'
            issues.append(IssueSpec('learning-trained-path', training.label, message, fix))
        return issues

    @staticmethod
    def _shape_issue(training: TrainingInteraction, path: str, expected: object, given: object) -> IssueSpec:
        """Return the issue of a dataset array whose shape does not fit."""
        message = f'The dataset of {training.label} gives {path} the shape {given}, but it needs {expected}.'
        fix = 'Give every array the shape (sequences, steps) followed by the shape of its parameter.'
        return IssueSpec('learning-data-shape', training.label, message, fix)

    def _data_issues(self, problem: ProblemSpec, training: TrainingInteraction) -> list[IssueSpec]:
        """Return the dataset arrays whose path, number of sequences and steps, or trailing shape do not fit."""
        data = training.data
        sequences, steps = data.sequences, data.steps
        issues = []
        for path, values in data.values.items():
            if not problem.parameters.has(path):
                message = f'The dataset of {training.label} sets {path}, which is no parameter of {training.model}.'
                fix = f'Set parameters of the nested system, such as {", ".join(problem.parameters.names[:3])}.'
                issues.append(IssueSpec('learning-data-shape', training.label, message, fix))
                continue
            expected = (sequences, steps, *problem.parameters.entry(path).shape)
            if np.shape(values) != expected:
                issues.append(self._shape_issue(training, path, expected, np.shape(values)))
        weights = data.weights if isinstance(data.weights, Mapping) else {'loss': data.weights}
        labels = {'loss', *(item.label for item in training.metrics)}
        for label, array in weights.items():
            if label not in labels:
                message = f'The dataset of {training.label} weights {label}, which is neither the loss nor a metric.'
                issues.append(IssueSpec('learning-data-shape', training.label, message, 'Weight the loss or a metric.'))
            elif array is not None and np.shape(array) != (sequences, steps):
                issues.append(
                    self._shape_issue(training, f'the weights of {label}', (sequences, steps), np.shape(array)),
                )
        if data.times is not None and (np.shape(data.times) != (steps,) or not np.all(np.diff(data.times) > 0)):
            issues.append(self._shape_issue(training, 'the times', f'({steps},) increasing', np.shape(data.times)))
        return issues

    @staticmethod
    def _reader(is_state: bool, indices: int | np.ndarray) -> TracedRead:
        """Return the function that reads entries of the state or the parameter vector."""
        return lambda x, p: (x if is_state else p)[indices]

    @staticmethod
    def _scalar_reads(problem: ProblemSpec) -> dict[Expression, TracedRead]:
        """Return the readers of the symbols of the single elements and interactions of a problem."""
        reads: dict[Expression, TracedRead] = {}
        for is_state, layout in ((True, problem.states), (False, problem.parameters)):
            for entry in layout.entries:
                if entry.members is not None or entry.name not in problem.symbols or entry.shape:
                    continue
                reads[problem.symbols[entry.name]] = _LearningCompiler._reader(is_state, entry.start)
        return reads

    @staticmethod
    def _member_reads(problem: ProblemSpec, names: Mapping[str, str]) -> dict[Expression, TracedRead]:
        """Return the readers of the scalar variables of the members of a group, one array of members each."""
        reads: dict[Expression, TracedRead] = {}
        for name in names.values():
            entry = ObservableEvaluator._entry_of(problem, name)
            if entry is None or entry.columns != 1:
                continue
            is_state, members = problem.states.has(name), np.arange(entry.start, entry.start + entry.size)
            reads[problem.symbols[name]] = _LearningCompiler._reader(is_state, members)
        return reads

    @staticmethod
    def _printed(
        path: str,
        expression: object,
        reads: Mapping[Expression, TracedRead],
        time: Expression,
    ) -> TracedFunction:
        """Return the pure JAX function of an expression whose symbols the readers read."""
        expression = SR(expression)
        arguments = sorted(symbols_of([expression]) - {time}, key=str)
        missing = [str(item) for item in arguments if item not in reads]
        if missing:
            error_msg = f'The observable {path} reads {", ".join(missing)}, which the nested system does not have.'
            raise ValidationError(error_msg)
        printer = ExpressionPrinter('jax', time=time, compiled=True).check().prepare(arguments)
        function = ExpressionPrinter._function_of(printer.run([expression]).code_)
        readers = [reads[item] for item in arguments]

        def value(t: object, x: TracedArray, p: TracedArray) -> TracedArray:
            return jnp.asarray(function(t, *[read(x, p) for read in readers])[0], dtype=float)

        return value

    @staticmethod
    def _evaluated(observable: ObservableSpec, namespace: SimpleNamespace) -> object:
        """Return the expression of an observable in a namespace, or raise naming the variable it misses."""
        if not callable(observable.expression):
            return observable.expression
        try:
            return observable.expression(namespace)
        except AttributeError as error:
            error_msg = f'The observable {observable.label} reads {error.name}, which the nested system does not have.'
            raise ValidationError(error_msg) from error

    def _measure_of(self, problem: ProblemSpec, observable: ObservableSpec) -> TracedFunction:
        """Return the pure JAX function of an observable of the nested system, a value of its states and parameters."""
        if observable.group is None:
            namespace = ObservableEvaluator._namespace_of(problem)
            expression = self._evaluated(observable, namespace)
            return self._printed(observable.label, expression, self._scalar_reads(problem), problem.time)
        element = next((item for item in problem.elements if item.path == observable.group), None)
        if element is None or observable.reduce not in REDUCTIONS:
            error_msg = (
                f'The observable {observable.label} reduces {observable.group} with {observable.reduce!r}, but '
                f'learning reduces a group of the nested system with one of {", ".join(REDUCTIONS)}.'
            )
            raise ValidationError(error_msg)
        names = {**element.states, **element.parameters}
        reads = {**self._scalar_reads(problem), **self._member_reads(problem, names)}
        if observable.variable is not None:
            if observable.variable not in names:
                error_msg = f'The observable {observable.label} reads {observable.variable}, which is not in the group.'
                raise ValidationError(error_msg)
            member: object = problem.symbols[names[observable.variable]]
        else:
            members = SimpleNamespace(**{key: problem.symbols[name] for key, name in names.items()}, t=problem.time)
            member = self._evaluated(observable, members)
        value, kind, size = (
            self._printed(observable.label, member, reads, problem.time),
            observable.reduce,
            element.members or 1,
        )

        def reduced(t: object, x: TracedArray, p: TracedArray) -> TracedArray:
            argument = jnp.broadcast_to(value(t, x, p), (size,))
            return JaxBackend._reduced(str(kind), argument, jnp.ones(size, dtype=bool))

        return reduced

    def _measure_issues(self, problem: ProblemSpec, training: TrainingInteraction) -> list[IssueSpec]:
        """Lower the loss and the metrics into `measures_`, and return the observables that read missing variables."""
        self.measures_ = {}
        issues = []
        for label, observable in (('loss', training.loss), *((item.label, item) for item in training.metrics)):
            try:
                self.measures_[label] = self._measure_of(problem, observable)
            except ValidationError as error:
                fix = 'Read states and parameters of the nested system in the loss and the metrics.'
                issues.append(IssueSpec('learning-unknown-observable', training.label, str(error), fix))
        return issues

    def _issues(self, system: System) -> list[IssueSpec]:
        """Return the issues of a system that trains, compiling its nested system as far as it can.

        Args:
            system:
                The system, with the observables of the run.

        Returns:
            issues:
                The issues, in the order they were found.
        """
        self.training_, self.inner_ = self._find_training(system), None
        training = self.training_
        if training is None:
            message = f'The system {system.label} holds no training interaction.'
            return [IssueSpec('compiler', system.label, message, 'Compile it with PassCompiler.')]
        try:
            training.check()
        except ValidationError as error:
            return [self._issue_of(error, training.label, 'value', 'Correct the training interaction.')]
        self.law_ = next(
            item for item in training.prepare().run(SimpleNamespace()).laws_ if isinstance(item, GradientStepLaw)
        )
        issues = self._setting_issues(system, training)
        model = self._find_system(system, training.model)
        if model is None:
            message = f'{training.label} trains {training.model}, which is not a nested system of {system.label}.'
            fix = f'Bind `model` to one of {", ".join(item.label for item in system.systems) or "no nested system"}.'
            return [*issues, IssueSpec('learning-model', training.label, message, fix)]
        stops = self._differentiable_issues(model, training) or self._compile_inner(model, training)
        if stops or self.inner_ is None:
            return [*issues, *stops]
        issues += self._trained_issues(self.inner_, training)
        issues += self._data_issues(self.inner_, training)
        return issues + self._measure_issues(self.inner_, training)

    def _layouts(
        self,
        problem: ProblemSpec,
        training: TrainingInteraction,
    ) -> tuple[LayoutSpec, LayoutSpec, tuple[str, ...]]:
        """Return the layouts of the trained parameters and of the hyperparameters, and the trained inner names."""
        trained = tuple(name for name in problem.states.names if name in set(training.trained))
        states, start = [], 0
        for name in trained:
            entry = problem.states.entry(name)
            states.append(dataclasses.replace(entry, name=f'{training.model}/{name}', start=start))
            start += entry.size
        hyperparameters = [name for name, spec in training._get_specs().items() if isinstance(spec, ParameterSpec)]
        parameters = tuple(
            EntrySpec(f'{training.label}.{name}', index, (), training._get_specs()[name].domain)
            for index, name in enumerate(hyperparameters)
        )
        return LayoutSpec(tuple(states)), LayoutSpec(parameters), trained

    def _compile(self, initial_state: Mapping[str, object]) -> BaseProblemSpec:
        """Return the learning problem of the prepared system.

        Args:
            initial_state:
                Initial values of the trained parameters, by their paths from the root.

        Returns:
            problem:
                The learning problem.
        """
        problem, training = self.inner_, self.training_
        if problem is None or training is None:
            error_msg = 'The nested system did not compile. Prepare the compiler on a valid system.'
            raise ValidationError(error_msg)
        states, parameters, trained = self._layouts(problem, training)
        initial = np.concatenate(
            [np.asarray(problem.states.read(problem.initial_state, name), dtype=float).ravel() for name in trained],
        )
        hyperparameters = {entry.name.rpartition('.')[2]: entry.name for entry in parameters.entries}
        values = np.array([float(getattr(training, name)) for name in hyperparameters], dtype=float)
        step = self.step_
        learning = LearningProblemSpec(
            FormKind.LEARNING,
            states,
            parameters,
            initial,
            values,
            symbol('t'),
            system=self.system_,
            inner=problem,
            model=training.model,
            training=training.label,
            law=self.law_,
            trained=trained,
            hyperparameters=hyperparameters,
            step=step,
            measures=dict(self.measures_),
        )
        return self._with_initial_state(learning, initial_state)

    @staticmethod
    def _with_initial_state(problem: LearningProblemSpec, initial_state: Mapping[str, object]) -> LearningProblemSpec:
        """Return a learning problem whose trained parameters start from given values."""
        state = problem.initial_state.copy()
        for name, value in initial_state.items():
            if not problem.states.has(name):
                error_msg = f'{name} is not a trained parameter. Use one of {", ".join(problem.states.names)}.'
                raise ValidationError(error_msg)
            entry = problem.states.entry(name)
            state[entry.indices] = entry.encode(value)
        return dataclasses.replace(problem, initial_state=state)

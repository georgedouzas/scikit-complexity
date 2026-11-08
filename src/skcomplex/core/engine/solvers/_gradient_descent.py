"""Define the gradient descent solver.

The solver trains the parameters of a nested system epoch by epoch.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Callable, Mapping

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from ...components.interactions import OPTIMIZERS
from ...components.laws import GradientStepLaw
from ...components.spaces import DiscreteTime
from ...errors import SolverError, ValidationError
from ...specs import StructureStateSpec, SupportSpec
from ...types import Array, FormKind, OrderKind
from ..problems._learning_problem_spec import LearningProblemSpec
from ..types import TracedArray
from ._base import BaseSolver

PRECISIONS = ('float64', 'float32')

type Batch = tuple[tuple[TracedArray, ...], TracedArray]
type Unroll = Callable[[TracedArray, tuple[TracedArray, ...]], TracedArray]


class GradientDescentSolver(BaseSolver[LearningProblemSpec]):
    """Solver of the learning form.

    It trains the parameters of a nested system by gradient steps on JAX and Optax. Every unit of time is one epoch. The
    solver shuffles the sequences of the dataset with its random number generator, splits them into batches, and for
    every batch unrolls the nested system from its initial states over each sequence, writing the values of the dataset
    into its parameters before every inner step. A discrete map unrolls with `jax.lax.scan`, and ordinary differential
    equations with one `diffrax.diffeqsolve` per inner step, with `Tsit5`, a PID step size controller, and recursive
    checkpointing. The loss is the weighted mean of the loss observable over the steps and sequences, its gradient comes
    from `jax.value_and_grad`, and an Optax optimizer updates the trained parameters. Truncated backpropagation through
    time stops the gradient of the carried inner states every `truncation` steps. The optimizer reads its
    hyperparameters from the parameter vector before every frame, so a schedule takes effect at the next frame, and its
    state persists across frames until the solver is prepared again. The solver records the mean batch loss of every
    epoch and every metric over the whole dataset at the end of the epoch as reporters `<interaction path>/loss` and
    `<interaction path>/<metric>`.

    Args:
        precision:
            `'float64'` to compute in double precision, inside `jax.enable_x64(True)`, so the unrolled runs agree with
            NumPy within 1e-10, or `'float32'` for speed.

        random_state:
            The seed or generator of the shuffling of the sequences.

        until:
            The end time of the run the solver prepares for.

    Attributes:
        theta_:
            The trained parameters after the last epoch, as a JAX array.

        optimizer_state_:
            The state of the Optax optimizer, such as the moments of Adam.

    Raises:
        ValidationError:
            The precision is unknown.

    Examples:
        >>> from skcomplex.core.engine.solvers import GradientDescentSolver
        >>> GradientDescentSolver(precision='float32').precision
        'float32'
    """

    problem_: LearningProblemSpec
    theta_: TracedArray
    optimizer_state_: optax.OptState

    def __init__(
        self,
        precision: str = 'float64',
        *,
        random_state: int | np.random.Generator | None = None,
        until: float = math.inf,
    ) -> None:
        super().__init__(random_state=random_state, until=until)
        self.precision = precision

    @classmethod
    def _get_form(cls) -> FormKind:
        """Return the learning form."""
        return FormKind.LEARNING

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the discrete time of epochs and the gradient-step law."""
        return SupportSpec(
            time={DiscreteTime},
            space={None},
            laws={GradientStepLaw},
            noise=frozenset(),
            orders=frozenset({OrderKind.INTEGER}),
        )

    @classmethod
    def _is_stepwise(cls) -> bool:
        """Return that the solver advances in whole epochs."""
        return True

    def _law(self) -> GradientStepLaw:
        """Return the gradient-step law of the problem."""
        law = self.problem_.law
        if law is None:
            error_msg = 'The learning problem holds no gradient-step law. Compile a system that trains.'
            raise SolverError(error_msg)
        return law

    def _optimizer(self) -> tuple[optax.GradientTransformation, dict[str, str]]:
        """Return the Optax optimizer of the law and the names of its hyperparameters.

        The optimizer reads its hyperparameters from the state of the law. The names map every hyperparameter of the
        Optax optimizer to the parameter of the training that sets it.
        """
        name = self._law().optimizer
        names = {'learning_rate': 'learning_rate'}
        if name == OPTIMIZERS[1]:
            names['momentum'] = 'momentum'
        elif name == OPTIMIZERS[2]:
            names['decay'] = 'beta2'
        elif name == OPTIMIZERS[3]:
            names.update(b1='beta1', b2='beta2')
        factory = {OPTIMIZERS[2]: optax.rmsprop, OPTIMIZERS[3]: optax.adam}.get(name, optax.sgd)
        values = self._hyperparameters(self.problem_.initial_parameters)
        optimizer = optax.inject_hyperparams(factory)(**{key: values[item] for key, item in names.items()})
        return optimizer, names

    def _hyperparameters(self, p: Array) -> dict[str, float]:
        """Return the hyperparameters of the optimizer in a parameter vector, by name."""
        layout = self.problem_.parameters
        return {name: float(p[layout.entry(path).start]) for name, path in self.problem_.hyperparameters.items()}

    def _inner_indices(self) -> TracedArray:
        """Return the indices of the trained parameters in the states of the inner problem."""
        inner = self.problem_.inner
        if inner is None:
            error_msg = 'The learning problem holds no inner problem. Compile a system that trains.'
            raise SolverError(error_msg)
        indices = [
            np.arange(entry.start, entry.start + entry.size)
            for entry in (inner.states.entry(name) for name in self.problem_.trained)
        ]
        return jnp.asarray(np.concatenate(indices))

    def _data(self) -> tuple[tuple[slice, ...], tuple[Array, ...], dict[str, Array]]:
        """Return the parameter slices the dataset sets, its values, and the step weights.

        The values are flattened per step.
        """
        law, inner = self._law(), self.problem_.inner
        if inner is None:
            error_msg = 'The learning problem holds no inner problem. Compile a system that trains.'
            raise SolverError(error_msg)
        data = law.data
        slices = tuple(inner.parameters.locate(path)[1] for path in data.values)
        values = tuple(
            np.asarray(array, dtype=float).reshape(data.sequences, data.steps, -1) for array in data.values.values()
        )
        given = data.weights if isinstance(data.weights, Mapping) else {'loss': data.weights}
        labels = ['loss', *(item.label for item in law.metrics)]
        ones = np.ones((data.sequences, data.steps))
        weights = {
            label: np.asarray(given[label], dtype=float) if given.get(label) is not None else ones for label in labels
        }
        return slices, values, weights

    def _inputs_of(self, p: TracedArray, slices: tuple[slice, ...], step: tuple[TracedArray, ...]) -> TracedArray:
        """Return a parameter vector with the values of the dataset at one inner step.

        The values are written into the vector.
        """
        for where, value in zip(slices, step, strict=True):
            p = p.at[where].set(value)
        return p

    def _advanced(self, t: TracedArray, t1: TracedArray, x: TracedArray, p: TracedArray) -> TracedArray:
        """Return the inner state after one inner step.

        The step is an update or the solution of the ODE over the step.
        """
        function = self.problem_.step
        if function is None:
            error_msg = 'The learning problem holds no JAX function of its inner problem.'
            raise SolverError(error_msg)
        inner = self.problem_.inner
        if inner is not None and inner.form == FormKind.DISCRETE:
            return function(t, x, p)
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(function),
            diffrax.Tsit5(),
            t0=t,
            t1=t1,
            dt0=None,
            y0=x,
            args=p,
            saveat=diffrax.SaveAt(t1=True),
            stepsize_controller=diffrax.PIDController(rtol=1e-10, atol=1e-10),
            adjoint=diffrax.RecursiveCheckpointAdjoint(),
            max_steps=100000,
        )
        return solution.ys[-1]

    def _unroll(self) -> Unroll:
        """Return the function that gives every measure of one sequence.

        It takes the trained parameters and the values of the sequence. The function returns one row per inner step and
        one column per measure, the loss first.
        """
        law, problem = self._law(), self.problem_
        inner = problem.inner
        if inner is None:
            error_msg = 'The learning problem holds no inner problem. Compile a system that trains.'
            raise SolverError(error_msg)
        indices, slices = self._inner_indices(), self._data()[0]
        measures = [problem.measures['loss'], *(problem.measures[item.label] for item in law.metrics)]
        steps = law.data.steps
        ends = np.arange(1.0, steps + 1) if law.data.times is None else np.asarray(law.data.times, dtype=float)
        starts = np.concatenate([[0.0], ends[:-1]])
        truncation = law.truncation
        x0, p0 = jnp.asarray(inner.initial_state), jnp.asarray(inner.initial_parameters)

        def unroll(theta: TracedArray, sequence: tuple[TracedArray, ...]) -> TracedArray:
            def body(x: TracedArray, item: tuple[TracedArray, ...]) -> tuple[TracedArray, TracedArray]:
                k, t, t1, *step = item
                if truncation is not None:
                    x = jnp.where(k % truncation == 0, jax.lax.stop_gradient(x), x)
                x = x.at[indices].set(theta)
                p = self._inputs_of(p0, slices, tuple(step))
                following = self._advanced(t, t1, x, p)
                return following, jnp.stack([measure(t1, following, p) for measure in measures])

            scanned = (jnp.arange(steps), jnp.asarray(starts), jnp.asarray(ends), *sequence)
            return jax.lax.scan(body, x0, scanned)[1]

        return unroll

    def _prepare_functions(self) -> None:
        """Set the compiled functions of the loss, its gradient, and the measures.

        They run over many sequences.
        """
        unroll = jax.vmap(self._unroll(), in_axes=(None, 0))

        def loss(theta: TracedArray, batch: Batch) -> TracedArray:
            sequences, weights = batch
            values = unroll(theta, sequences)[..., 0]
            return jnp.sum(weights * values) / jnp.sum(weights)

        optimizer, self._names = self._optimizer()

        def update(
            theta: TracedArray,
            state: optax.OptState,
            batch: Batch,
        ) -> tuple[TracedArray, optax.OptState, TracedArray]:
            value, gradient = jax.value_and_grad(loss)(theta, batch)
            updates, state = optimizer.update(gradient, state, theta)
            return optax.apply_updates(theta, updates), state, value

        self._optimizer_ = optimizer
        self._update = jax.jit(update)
        self._measures = jax.jit(unroll)

    def _reporters(self, theta: TracedArray, loss: float | None) -> StructureStateSpec:
        """Return the loss and the metrics over the whole dataset.

        When given, the loss of the epoch is used.
        """
        law, path = self._law(), self.problem_.training
        _, values, weights = self._data()
        measured = np.asarray(self._measures(theta, tuple(jnp.asarray(item) for item in values)))
        labels = ['loss', *(item.label for item in law.metrics)]
        means = {
            label: float(np.sum(weights[label] * measured[..., column]) / np.sum(weights[label]))
            for column, label in enumerate(labels)
        }
        if loss is not None:
            means['loss'] = loss
        return StructureStateSpec(reporters={f'{path}/{label}': value for label, value in means.items()})

    def _prepare(self, context: LearningProblemSpec | None) -> None:
        """Prepare the solver for a learning problem.

        It first raises an error for an unknown precision.

        It gets a new optimizer state and the measures of frame 0.

        Args:
            context:
                The learning problem.

        Raises:
            ValidationError:
                The precision is unknown.

            ContextError:
                No problem is passed.

            SolverError:
                The problem is not a learning problem.
        """
        if self.precision not in PRECISIONS:
            error_msg = f'The precision is {self.precision!r}. Choose one of {", ".join(map(repr, PRECISIONS))}.'
            raise ValidationError(error_msg)
        super()._prepare(context)
        if not isinstance(self.problem_, LearningProblemSpec):
            error_msg = f'GradientDescentSolver solves learning problems, not a {type(self.problem_).__name__}.'
            raise SolverError(error_msg)
        with jax.enable_x64(self.precision == PRECISIONS[0]):
            self._prepare_functions()
            self.theta_ = jnp.asarray(self.problem_.initial_state)
            self.optimizer_state_ = self._optimizer_.init(self.theta_)
            self.snapshot_ = self._reporters(self.theta_, None)

    def _epoch(self, values: tuple[Array, ...], weights: Array) -> float:
        """Run one epoch of shuffled batches and update the trained parameters.

        The mean batch loss is returned.
        """
        law = self._law()
        order = self.rng_.permutation(law.data.sequences)
        losses = []
        for start in range(0, len(order), law.batch_size):
            rows = order[start : start + law.batch_size]
            batch = (tuple(jnp.asarray(item[rows]) for item in values), jnp.asarray(weights[rows]))
            self.theta_, self.optimizer_state_, value = self._update(self.theta_, self.optimizer_state_, batch)
            losses.append(float(value))
        return float(np.mean(losses))

    def _advance(self, t: float, x: Array, p: Array, until: float) -> Array:
        """Train for the whole epochs from time `t` to time `until`.

        Training uses the hyperparameters of the parameter vector.

        Args:
            t:
                The current epoch.

            x:
                The trained parameters.

            p:
                The hyperparameters.

            until:
                The epoch to advance to.

        Returns:
            state:
                The trained parameters after the epochs.
        """
        _, values, weights = self._data()
        loss = math.nan
        with jax.enable_x64(self.precision == PRECISIONS[0]):
            self.theta_ = jnp.asarray(x)
            values_of = self._hyperparameters(p)
            for name, item in self._names.items():
                self.optimizer_state_.hyperparams[name] = jnp.asarray(values_of[item])
            for _ in range(round(until - t)):
                loss = self._epoch(values, weights['loss'])
            self.snapshot_ = self._reporters(self.theta_, loss)
            return np.asarray(self.theta_, dtype=float)

    def _collect(self) -> None:
        """Keep the reporters of the last epoch and no events."""
        self.events_ = []

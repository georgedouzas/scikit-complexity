"""Run compiled problems in JAX.

They run as pure functions that JAX traces, compiles, and differentiates.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse
from jax.scipy.special import logsumexp

from ...components.laws import EquationLaw, GradientStepLaw, RateLaw, UpdateLaw
from ...components.spaces import ContinuousTime, DiscreteTime
from ...errors import BackendError, ValidationError
from ...specs import LoweredFunctionsSpec, ReadSpec, SupportSpec
from ...types import Array, FormKind, OrderKind
from ..problems._draw_kernel import DrawKernel
from ..problems._expression_printer import ExpressionPrinter
from ..problems._law_kernel import LawKernel
from ..problems._lowered_kernels_spec import LoweredKernelsSpec
from ..problems._reduction_kernel import ReductionKernel
from ..problems._reduction_schedule_spec import ReductionScheduleSpec
from ..problems._reference_spec import ReferenceSpec
from ..types import ReadKind, TracedArray, TracedFunction
from ._base import BaseBackend

FORMS = (FormKind.ODE, FormKind.DISCRETE)
MEMBER_READS = (ReadKind.MEMBER, ReadKind.REDUCED)
UNIFORM_FLOOR = 1e-12

type TracedRead = Callable[[TracedArray, TracedArray], TracedArray]
type TracedValues = Callable[[object, TracedArray, TracedArray], list[TracedArray]]


class JaxBackend(BaseBackend):
    """Backend that writes discrete maps and ODEs as pure JAX functions.

    The ODEs are ordinary differential equations. The functions return new arrays instead of writing into buffers, so
    JAX traces, compiles, and differentiates them. Reductions lower to sums, means, and matrix products, and a reduction
    weighted by a shaped parameter to the product with its block of the parameter vector reshaped into a matrix. The
    forward functions in `functions_` take and return NumPy arrays and compute in double precision. Rule, noise, and
    jump laws, reductions over neighbourhoods, and laws given as Python callables have no JAX form.

    Attributes:
        functions_:
            The forward functions of the form of the kernels of the last run, on NumPy arrays.

        traced_:
            The pure JAX function `f(t, x, p)` of the form of the kernels of the last run, the drift of ordinary
            differential equations or the update of a discrete map.

    Examples:
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.core.engine.backends import JaxBackend
        >>> from skcomplex.core.components.systems import System
        >>> from skcomplex.systems.biology.common.populations import PopulationElement
        >>> system = System(elements=[PopulationElement('a', x=2.0)])
        >>> problem = simulate(system, until=0, backend=JaxBackend()).problem
        >>> problem.backend
        'jax'
    """

    traced_: TracedFunction

    @staticmethod
    def advance(
        function: TracedFunction,
        t: TracedArray,
        t1: TracedArray,
        x: TracedArray,
        p: TracedArray,
        *,
        form: FormKind | str,
        throw: bool = True,
    ) -> TracedArray:
        """Return the state after one step of a pure function, traced by JAX.

        A discrete map applies its update once, and ordinary differential equations are solved from `t` to `t1` by
        `diffrax.diffeqsolve` with `Tsit5`, a PID step size controller, and recursive checkpointing, so gradients flow
        through the step.

        Args:
            function:
                The pure function `f(t, x, p)`, the update of a map or the drift of ordinary differential equations.

            t:
                The time of the state.

            t1:
                The time after the step.

            x:
                The state.

            p:
                The parameter vector.

            form:
                The form of the function, a discrete map or ordinary differential equations.

            throw:
                Whether a solve that fails, such as one that reaches its largest number of steps, raises an error, or
                else returns states that are not finite, as samplers that reject such proposals need.

        Returns:
            state:
                The state at `t1`.
        """
        if form == FormKind.DISCRETE:
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
            throw=throw,
        )
        if throw:
            return solution.ys[-1]
        return jnp.where(solution.result == diffrax.RESULTS.successful, solution.ys[-1], jnp.nan)

    @staticmethod
    def unroll(
        function: TracedFunction,
        form: FormKind | str,
        times: Array,
        *,
        throw: bool = True,
        draws: DrawKernel | None = None,
    ) -> Callable[..., TracedArray]:
        """Return the pure function from an initial state and a parameter vector to the states at given times.

        A discrete map is unrolled over every whole step up to the last time with `jax.lax.scan`, and ordinary
        differential equations over the intervals between the times, each solved by `advance`. A map that draws takes
        a JAX key, splits one key per step, and writes the uniform and normal values of the step after the parameters,
        where the draw kernel writes them in NumPy runs, so the run is a pure function of the key.

        Args:
            function:
                The pure function `f(t, x, p)`, the update of a map or the drift of ordinary differential equations.

            form:
                The form of the function, a discrete map or ordinary differential equations.

            times:
                The increasing times, at least 0, whole numbers for a map.

            throw:
                Whether a failed solve raises an error, or else gives states that are not finite.

            draws:
                The draw kernel of a map that draws, or `None`.

        Returns:
            run:
                The function `run(x0, p, key=None)` that returns one row of states per time, which needs a key when the
                map draws.

        Raises:
            BackendError:
                The draws hold draws from `scipy.stats` distributions or random orders, which have no JAX form, or
                ordinary differential equations draw.

        Examples:
            >>> import jax.numpy as jnp
            >>> import numpy as np
            >>> from skcomplex.core.engine.backends import JaxBackend
            >>> from skcomplex.core.types import FormKind
            >>> run = JaxBackend.unroll(lambda t, x, p: p * x, FormKind.DISCRETE, np.array([0.0, 2.0]))
            >>> np.asarray(run(jnp.array([1.0]), 3.0)).ravel().tolist()
            [1.0, 9.0]
        """
        times = np.asarray(times, dtype=float)
        if form == FormKind.DISCRETE:
            steps = np.arange(round(float(times.max()))) if times.size else np.arange(0)
            starts, ends, indices = steps.astype(float), steps + 1.0, np.rint(times).astype(int)
        else:
            ends = np.unique(times[times > 0])
            starts = np.concatenate([[0.0], ends[:-1]])
            indices = np.searchsorted(np.concatenate([[0.0], ends]), times)
        drawn = JaxBackend._drawn(draws, form)

        def run(x0: TracedArray, p: TracedArray, key: TracedArray | None = None) -> TracedArray:
            if drawn is not None and key is None:
                error_msg = 'The map draws at every step, so its run takes a JAX key, `run(x0, p, key)`.'
                raise ValidationError(error_msg)
            keys = jax.random.split(key, starts.size) if drawn is not None else jnp.zeros(starts.size)

            def body(x: TracedArray, item: tuple[TracedArray, ...]) -> tuple[TracedArray, TracedArray]:
                t, t1, step_key = item
                q = p if drawn is None else drawn(p, step_key)
                following = JaxBackend.advance(function, t, t1, x, q, form=form, throw=throw)
                return following, following

            scanned = jax.lax.scan(body, x0, (jnp.asarray(starts), jnp.asarray(ends), keys))[1]
            states = jnp.concatenate([x0[None, :], scanned])
            return states[jnp.asarray(indices)]

        return run

    @staticmethod
    def _drawn(
        draws: DrawKernel | None,
        form: FormKind | str,
    ) -> Callable[[TracedArray, TracedArray], TracedArray] | None:
        """Return the function that writes the values a key draws after the parameters.

        It is `None` without draws. Uniform values lie in `[1e-12, 1)`, so the logits of relaxed draws stay finite.

        Args:
            draws:
                The draw kernel of the map, or `None`.

            form:
                The form of the function.

        Returns:
            extend:
                The function `extend(p, key)` of the extended parameter vector, or `None`.

        Raises:
            BackendError:
                The draws hold draws from `scipy.stats` distributions or random orders, or ordinary differential
                equations draw.
        """
        if draws is None or not draws.size:
            return None
        if draws.sampled or draws.orders or form != FormKind.DISCRETE:
            error_msg = (
                'Differentiable runs draw uniform and normal values of maps, and the system draws from scipy.stats '
                'distributions, in random orders, or in ordinary differential equations, which have no JAX form.'
            )
            raise BackendError(error_msg)
        offset, normal = draws.offset, jnp.asarray(draws.normal)
        size = int(draws.normal.size)

        def extend(p: TracedArray, key: TracedArray) -> TracedArray:
            uniform_key, normal_key = jax.random.split(key)
            uniforms = jax.random.uniform(uniform_key, (size,), minval=UNIFORM_FLOOR)
            values = jnp.where(normal, jax.random.normal(normal_key, (size,)), uniforms)
            return jnp.concatenate([jnp.atleast_1d(p)[:offset], values])

        return extend

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, laws, and orders the backend writes functions of."""
        return SupportSpec(
            time={ContinuousTime, DiscreteTime},
            space={None},
            laws={EquationLaw, GradientStepLaw, RateLaw, UpdateLaw},
            noise=frozenset(),
            orders=frozenset({OrderKind.INTEGER}),
        )

    @classmethod
    def _get_name(cls) -> str:
        """Return the name of the backend that problems record."""
        return 'jax'

    @staticmethod
    def _reader(reference: ReferenceSpec, owner: str) -> TracedRead:
        """Return the function that reads one symbol of a law kernel.

        It reads it from the state and extended parameter vectors.
        """
        is_state, source = reference.is_state, reference.source
        if source is ReadKind.SINGLE or (source is ReadKind.REDUCED and reference.entry.members is None):
            index = reference.index
            return lambda x, q: (x if is_state else q)[index]
        if source in MEMBER_READS:
            members = reference.members
            return lambda x, q: (x if is_state else q)[members]
        error_msg = (
            f'A law of {owner} reads {reference.entry.name} as a {source} read, which the JAX backend does not write. '
            'Write the law with reads of its own members, single elements, and reductions over groups.'
        )
        raise BackendError(error_msg)

    @staticmethod
    def _values_of(kernel: LawKernel) -> TracedValues:
        """Return the pure function of the values of every output of a law kernel.

        It gives one value per evaluation index.
        """
        if kernel.pairs is not None or kernel.has_python_functions:
            error_msg = (
                f'A law of {kernel.owner} runs over pairs or calls a Python function, which the JAX backend does not '
                'write. Write it as a symbolic law of the members of a group.'
            )
            raise BackendError(error_msg)
        arguments = kernel.arguments
        readers = [JaxBackend._reader(kernel.references[item], kernel.owner) for item in arguments]
        printer = ExpressionPrinter('jax', time=kernel.time, compiled=True).prepare(arguments)
        function = ExpressionPrinter._function_of(printer.run([item.expression for item in kernel.outputs]).code_)
        rows = kernel.rows

        def values(t: object, x: TracedArray, q: TracedArray) -> list[TracedArray]:
            computed = function(t, *[read(x, q) for read in readers])
            return [jnp.broadcast_to(jnp.asarray(value, dtype=float), (rows,)) for value in computed]

        return values

    @staticmethod
    def _target_of(target: ReferenceSpec | None, owner: str) -> ReferenceSpec:
        """Return the target of an output.

        It checks first that the target is a member or single read.
        """
        if target is None or target.source not in {ReadKind.MEMBER, ReadKind.SINGLE}:
            error_msg = (
                f'A law of {owner} writes through a table or to no target, which the JAX backend does not write.'
            )
            raise BackendError(error_msg)
        return target

    @staticmethod
    def _adding(kernels: list[LawKernel], size: int) -> TracedFunction:
        """Return the pure function that sums the outputs of kernels into a zero vector."""
        parts = [
            (
                JaxBackend._values_of(kernel),
                [JaxBackend._target_of(item.target, kernel.owner) for item in kernel.outputs],
            )
            for kernel in kernels
        ]

        def evaluate(t: object, x: TracedArray, q: TracedArray) -> TracedArray:
            output = jnp.zeros(size)
            for values, targets in parts:
                for target, value in zip(targets, values(t, x, q), strict=True):
                    if target.source is ReadKind.SINGLE:
                        output = output.at[target.index].add(value.sum())
                    else:
                        output = output.at[target.members].add(value)
            return output

        return evaluate

    @staticmethod
    def _assigning(kernels: list[LawKernel]) -> TracedFunction:
        """Return the pure function that sets the outputs of kernels on the state.

        It adds the changes of combined outputs.
        """
        parts = []
        for kernel in kernels:
            if any(item.reduce is not None for item in kernel.outputs):
                error_msg = f'A law of {kernel.owner} reduces over rows, which the JAX backend does not write.'
                raise BackendError(error_msg)
            targets = [JaxBackend._target_of(item.target, kernel.owner) for item in kernel.outputs]
            parts.append((JaxBackend._values_of(kernel), targets, [item.combined for item in kernel.outputs]))

        def evaluate(t: object, x: TracedArray, q: TracedArray) -> TracedArray:
            output = x
            for values, targets, combined in parts:
                for target, computed, adds in zip(targets, values(t, x, q), combined, strict=True):
                    single = target.source is ReadKind.SINGLE
                    where, value = (target.index, computed[0]) if single else (target.members, computed)
                    output = output.at[where].add(value - x[where]) if adds else output.at[where].set(value)
            return output

        return evaluate

    @staticmethod
    def _reduced(kind: str, argument: TracedArray, condition: TracedArray) -> TracedArray:
        """Return the reduction of the selected members of a group.

        It is 0 for no member, except for the count.
        """
        count = condition.sum()
        if kind == 'count':
            return count.astype(float)
        if kind in {'max', 'min'}:
            fill = -jnp.inf if kind == 'max' else jnp.inf
            extreme = jnp.max if kind == 'max' else jnp.min
            return jnp.where(count > 0, extreme(jnp.where(condition, argument, fill)), 0.0)
        total = jnp.where(condition, argument, 0.0).sum()
        if kind == 'sum':
            return total
        mean = total / jnp.maximum(count, 1)
        if kind == 'mean':
            return mean
        if kind == 'var':
            return jnp.where(condition, (argument - mean) ** 2, 0.0).sum() / jnp.maximum(count, 1)
        if kind == 'logsumexp':
            return logsumexp(argument, b=condition)
        error_msg = f'The reduction {kind!r} has no JAX form. Use sum, mean, max, min, var, logsumexp, or count.'
        raise BackendError(error_msg)

    @staticmethod
    def _weight_matrix(weights: scipy.sparse.csr_array | ReadSpec) -> Callable[[TracedArray, TracedArray], object]:
        """Return the function that gives the weight matrix of a reduction.

        The matrix is a shaped parameter or a constant matrix.
        """
        if isinstance(weights, ReadSpec) and weights.entry is not None:
            is_state, index, shape = weights.is_state, weights.index, weights.entry.shape
            return lambda x, q: (x if is_state else q)[index].reshape(shape)
        matrix = np.asarray(scipy.sparse.csr_array(weights).toarray())
        return lambda x, q: matrix

    @staticmethod
    def _reduction_step(reduction: object) -> TracedFunction:
        """Return the pure function that writes one reduction into the parameters.

        It writes it into the extended parameter vector.
        """
        if not isinstance(reduction, ReductionKernel):
            error_msg = (
                'The laws reduce over neighbourhoods, which the JAX backend does not write. Reduce over groups, or '
                'write terms that factor into a part of the member and a part of the other member.'
            )
            raise BackendError(error_msg)
        symbols = reduction.symbols
        reads = [(reduction.reads[item].is_state, reduction.reads[item].index) for item in symbols]
        printer = ExpressionPrinter('jax', time=reduction.time, compiled=True).prepare(symbols)
        condition = reduction.condition if reduction.condition is not None else 1
        function = ExpressionPrinter._function_of(printer.run([reduction.argument, condition]).code_)
        kind, size, entry = reduction.kind, reduction.size, reduction.entry
        matrix = JaxBackend._weight_matrix(reduction.weights) if reduction.weights is not None else None

        def step(t: object, x: TracedArray, q: TracedArray) -> TracedArray:
            argument, selected = function(t, *[(x if is_state else q)[index] for is_state, index in reads])
            argument = jnp.broadcast_to(jnp.asarray(argument, dtype=float), (size,))
            selected = jnp.broadcast_to(jnp.asarray(selected, dtype=bool), (size,))
            if matrix is None:
                return q.at[entry.start].set(JaxBackend._reduced(kind, argument, selected))
            weighted = jnp.asarray(matrix(x, q)) @ jnp.where(selected, argument, 0.0)
            if kind == 'mean':
                weighted = weighted / jnp.maximum(selected.sum(), 1)
            return q.at[entry.indices].set(weighted)

        return step

    @staticmethod
    def _extending(function: TracedFunction, reductions: ReductionScheduleSpec | None) -> TracedFunction:
        """Return a pure function of the parameter vector that evaluates the reductions.

        The reductions are evaluated before the function.
        """
        if reductions is None or not reductions.steps:
            return function
        steps = [JaxBackend._reduction_step(item) for item in reductions.steps]
        extent, parameters = reductions.extent, reductions.parameters

        def extended(t: object, x: TracedArray, p: TracedArray) -> TracedArray:
            given = min(p.shape[0], parameters)
            q = jnp.zeros(extent).at[:given].set(p[:given])
            for step in steps:
                q = step(t, x, q)
            return function(t, x, q)

        return extended

    @staticmethod
    def _forward(function: TracedFunction) -> Callable[[float, Array, Array], Array]:
        """Return the function on NumPy arrays that runs a pure function compiled by JAX.

        It runs in double precision.
        """
        compiled = jax.jit(function)

        def forward(t: float, x: Array, p: Array) -> Array:
            with jax.enable_x64(True):
                return np.asarray(compiled(jnp.asarray(t), jnp.asarray(x), jnp.asarray(p)))

        return forward

    def _run(self, inputs: LoweredKernelsSpec | None) -> None:
        """Set the pure functions of the form of the kernels and their forward functions.

        The forward functions run on NumPy arrays.

        Args:
            inputs:
                The kernels of a problem.

        Raises:
            ValidationError:
                No kernels are passed.

            BackendError:
                The form is neither a discrete map nor ordinary differential equations, or a law has no JAX form.
        """
        if inputs is None:
            error_msg = 'A backend runs on the kernels of a problem. Pass a `LoweredKernelsSpec` to `run`.'
            raise ValidationError(error_msg)
        form, kernels = inputs.form, inputs.kernels
        if form not in FORMS:
            error_msg = f'The JAX backend writes discrete maps and ordinary differential equations, not {form}.'
            raise BackendError(error_msg)
        if form == FormKind.ODE:
            self.traced_ = self._extending(self._adding(kernels.get(RateLaw, []), inputs.size), inputs.reductions)
            self.functions_ = LoweredFunctionsSpec(drift=self._forward(self.traced_))
            return
        self.traced_ = self._extending(self._assigning(kernels.get(UpdateLaw, [])), inputs.reductions)
        self.functions_ = LoweredFunctionsSpec(update=self._forward(self.traced_))

"""Define the bases of what a system compiles to: values, problems, kernels, searches, printers, records, structures."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ...base import BaseConfig
from ...errors import ValidationError
from ...specs import RunStateSpec


class BaseKernel(BaseConfig[object, RunStateSpec]):
    """Base of the kernels: laws lowered to numbers, the parts of a problem that a solver runs at a state.

    A kernel stores in its constructor what the compiler lowered, such as compiled functions and the indices they read.
    `check()` validates them, `prepare(problem)` sets what a run needs, such as the first renewal times, and
    `run(state)` acts at a `RunStateSpec`, the time, the state vector, the parameters, and the random number generator,
    and keeps what the kernel computes, such as the next state or a matrix, in `values_`.

    A developer adds a law kind by subclassing it, together with the pass that builds the kernel and a solver that runs
    it. A kernel takes any context to prepare on, since the problem that holds it is built from its kernels.

    Attributes:
        values_:
            What the last run computed. Each kernel narrows its type.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.engine.problems import BaseKernel
        >>> from skcomplex.specs import RunStateSpec
        >>> class DecayKernel(BaseKernel):
        ...     def __init__(self, rate=1.0):
        ...         self.rate = rate
        ...
        ...     def _run(self, inputs):
        ...         self.values_ = -self.rate * inputs.x
        >>> DecayKernel(0.5).check().prepare().run(RunStateSpec(0.0, np.array([2.0]), np.zeros(0))).values_.tolist()
        [-1.0]
    """

    values_: object

    @staticmethod
    def _state_of(inputs: RunStateSpec | None) -> RunStateSpec:
        """Return the state a kernel runs at, raising an error when none is passed.

        Args:
            inputs:
                The state, or `None`.

        Returns:
            state:
                The state.

        Raises:
            ValidationError:
                No state is passed.
        """
        if inputs is None:
            error_msg = 'A kernel runs at a state. Pass a `RunStateSpec` to `run`.'
            raise ValidationError(error_msg)
        return inputs

    @staticmethod
    def _generator_of(state: RunStateSpec) -> np.random.Generator:
        """Return the random number generator of a state, raising an error when it has none.

        Args:
            state:
                The state.

        Returns:
            rng:
                The random number generator.

        Raises:
            ValidationError:
                The state has no random number generator.
        """
        if state.rng is None:
            error_msg = 'The kernel draws random numbers. Pass a state with a random number generator.'
            raise ValidationError(error_msg)
        return state.rng


class BaseSearch(BaseConfig[None, RunStateSpec]):
    """Base of the neighbour searches: the neighbours a kernel reads at a state.

    A search stores the entries and the neighbourhood it reads in its constructor, and `run(state)` keeps the
    neighbours at a `RunStateSpec` in `pairs_`: the row starts and the neighbours of every member, or the members of
    every pair.

    Attributes:
        pairs_:
            The neighbours of the last run. Each search narrows its type.
    """

    pairs_: object

    @staticmethod
    def _state_of(inputs: RunStateSpec | None) -> RunStateSpec:
        """Return the state a search runs at, raising an error when none is passed.

        Args:
            inputs:
                The state, or `None`.

        Returns:
            state:
                The state.

        Raises:
            ValidationError:
                No state is passed.
        """
        if inputs is None:
            error_msg = 'A search runs at a state. Pass a `RunStateSpec` to `run`.'
            raise ValidationError(error_msg)
        return inputs


class BasePrinter[ContextT, InputT](BaseConfig[ContextT, InputT]):
    """Base of the printers: they print Sage expressions as code of one target.

    A printer stores its target in its constructor, `prepare(symbols)` takes the symbols the code reads, in order, and
    `run(expressions)` keeps the code of the expressions, or a function that evaluates them, in `code_`. A printer
    names the types of the symbols it prepares on and of the expressions it prints, which differ between targets.

    Attributes:
        code_:
            The code of the last run. Each printer narrows its type.
    """

    code_: object

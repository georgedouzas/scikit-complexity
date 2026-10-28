"""Define the bases of what a system compiles to: values, problems, kernels, searches, printers, records, structures."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import TYPE_CHECKING

from ...base import BaseConfig

if TYPE_CHECKING:
    pass


class BaseKernel(BaseConfig):
    """Base of the kernels: laws lowered to numbers, the parts of a problem that a solver runs at a state.

    A kernel stores in its constructor what the compiler lowered, such as compiled functions and the indices they read.
    `check()` validates them, `prepare(problem)` sets what a run needs, such as the first renewal times, and
    `run(state)` acts at a `State`, the time, the state vector, the parameters, and the random number generator, and
    returns what the kernel computes, such as the next state or a matrix.

    A developer adds a law kind by subclassing it, together with the pass that builds the kernel and a solver that runs
    it.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.engine.problems import BaseKernel
        >>> from skcomplex.specs import RunStateSpec
        >>> class DecayKernel(BaseKernel):
        ...     def __init__(self, rate=1.0):
        ...         self.rate = rate
        ...
        ...     def _run(self, inputs):
        ...         return -self.rate * inputs.x
        >>> DecayKernel(0.5).check().prepare().run(RunStateSpec(0.0, np.array([2.0]), np.zeros(0))).tolist()
        [-1.0]
    """


class BaseSearch(BaseConfig):
    """Base of the neighbour searches: the neighbours a kernel reads at a state.

    A search stores the entries and the neighbourhood it reads in its constructor, and `run(state)` returns the
    neighbours at a `State`: the row starts and the neighbours of every member, or the members of every pair.
    """


class BasePrinter(BaseConfig):
    """Base of the printers: they print Sage expressions as code of one target.

    A printer stores its target in its constructor, `prepare(symbols)` takes the symbols the code reads, in order, and
    `run(expressions)` returns the code of the expressions, or a function that evaluates them.
    """







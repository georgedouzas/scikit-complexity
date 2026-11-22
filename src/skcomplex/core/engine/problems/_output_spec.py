"""Define the output spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass, field

from sage.all__sagemath_symbolics import Expression

from ...base import BaseSpec
from ...types import Indices
from ..types import NumericFunction
from ._noise_process_spec import _NoiseProcessSpec
from ._reference_spec import ReferenceSpec


@dataclass(frozen=True)
class OutputSpec(BaseSpec):
    """A spec of one value that a kernel of laws gives for one target.

    Every law of a kernel gives one value for its target, such as the rate of the size of a population, written as a
    symbolic expression or computed by a Python function. The spec also says how the value is combined when several laws
    set one state, whether it reads a named noise process, and whether its function draws random numbers. It holds the
    rate of a population, the noise of a price, and the update of a cell.

    Read more in the user guide, under Extend the library.

    Args:
        target:
            The target component, or `None` for the propensity of a jump.

        expression:
            The symbolic value, or `None` when a callable produces it.

        function:
            The callable or `Call` function that produces the values of every component of its target, or `None`.

        arguments:
            The argument expressions of a `Call` function, or `None` for a function of time and the named values.

        component:
            The component of the callable's result that this output holds.

        components:
            The number of components the callable returns per evaluation index.

        process:
            The named Wiener process of a noise output, or `None` for one independent process per evaluation index.

        combined:
            Whether an assigned output adds its change from the state to the value an earlier output set, so that
            several update laws of one state combine.

        reduce:
            `'sum'` or `'mean'` for an assigned output that sets every member it writes to the sum or the mean of the
            values of the rows that write it, or `None`.

        counts:
            The number of rows, over every kernel, that write the member of every row of an output that reduces, or
            `None`.

        relative:
            Whether a function reads the values keyed by the names of its component.

        random:
            Whether a function receives the random stream as the keyword argument `rng`.

        vectorized:
            Whether a `Call` function runs once per evaluation with arrays.

    See Also:
        LawKernel: The kernel that gives the values.

    Examples:
        >>> from skcomplex.systems.biology.ecology import LogisticGrowthSystem
        >>> kernel = LogisticGrowthSystem().prepare().simulate(until=0).problem.jacobian.kernels[0]
        >>> kernel.outputs[0].expression
        -r_growth*x_population*(x_population/K_growth - 1)
    """

    target: ReferenceSpec | None
    expression: Expression | None = None
    function: NumericFunction | Callable[..., object] | None = None
    arguments: tuple[Expression, ...] | None = None
    component: int = 0
    components: int = 1
    process: _NoiseProcessSpec | None = None
    combined: bool = False
    reduce: str | None = None
    counts: Indices | None = field(default=None, compare=False, repr=False)
    relative: bool = False
    random: bool = False
    vectorized: bool = False

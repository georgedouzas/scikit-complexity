"""Define the bilinear form of two densities that meet and change class."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression

from ....errors import ValidationError
from ._operators import _OperatorFunction
from ._reductions import _calls_in

BILINEAR = _OperatorFunction('bilinear')
RATE_DIMENSIONS = (0, 2)
OUTCOME_DIMENSIONS = (3,)


def bilinear(
    rate: object,
    outcomes: object,
    first: object,
    second: object = None,
    *,
    conserved: bool = False,
) -> Expression:
    """Return the rate at which two densities meet and move between classes.

    A density is a group with one element per class of a trait, holding how much of a population is in each class. When
    the first density `f` in class `h` meets the second density `g` in class `k` at the rate `rate[h, k]`, the meeting
    adds to class `i` with the probability `outcomes[h, k, i]`, so class `i` gains
    `Σ_hk rate[h, k] outcomes[h, k, i] f_h g_k`, such as `bilinear(-0.5, None, f, g)` for a loss of `f` at half the rate
    of meetings. With `conserved=True` the first density also loses what moves out of class `h`, so its total stays the
    same, and without a second density the form moves amounts between classes at a fixed rate. A `RateLaw` whose whole
    rate is the form runs as meetings, which an ODE solver integrates and `KineticSolver` samples. Traders exchange
    wealth when they meet, people shift their opinion after a talk, and particles stick together when they collide.

    Read more in the user guide, under Laws.

    Args:
        rate:
            The signed rate, a number or a parameter expression, or an array `r[h, k]` of them.

        outcomes:
            The array `B[h, k, i]` of the probabilities of the class in the target, numbers or parameter expressions
            that sum to one over `i`, or `None` for the class of the first density, which needs a target of its size. A
            conserved form needs the array, since the identity makes its gain and loss cancel.

        first:
            The symbol of the first density.

        second:
            The symbol of the second density, or `None` for a linear redistribution of a conserved form.

        conserved:
            Whether the form also takes the matching loss from the first density, which is then the target, so
            `df_i/dt = Σ_hk rate[h, k] outcomes[h, k, i] f_h g_k - f_i Σ_k rate[i, k] g_k`.

    Returns:
        form:
            The unevaluated form, which the compiler turns into sums over the classes of the densities.

    Raises:
        ValidationError:
            A conserved form has no outcome table, a form without a second density is not conserved, or a table has the
            wrong dimensions or a shape that disagrees with the other.

    See Also:
        RateLaw: Takes the form as its rate.
        contraction: Sums a coupling tensor times the values of the members of a group.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import bilinear
        >>> f, g = var('f g')
        >>> bilinear(-0.5, None, f, g).operator()
        bilinear
    """
    if conserved and outcomes is None:
        error_msg = (
            f'The conserved bilinear form of {first} has no outcome table, but without one its gain and loss cancel. '
            'Give the outcome probabilities `B[h, k, i]`.'
        )
        raise ValidationError(error_msg)
    if second is None and not conserved:
        error_msg = (
            f'The bilinear form of {first} has no second density, which only a conserved form takes. Give the second '
            'density, or set `conserved=True`.'
        )
        raise ValidationError(error_msg)
    rates = _get_table('rate', rate, RATE_DIMENSIONS)
    table = None if outcomes is None else _get_table('outcomes', outcomes, OUTCOME_DIMENSIONS)
    if table is not None and rates.ndim and rates.shape != table.shape[:2]:
        error_msg = (
            f'The rate of the bilinear form has the shape {rates.shape}, but its outcomes have the shape '
            f'{table.shape[:2]} over the classes of the first and the second density.'
        )
        raise ValidationError(error_msg)
    operands = [SR(first), SR(second is not None), SR(first if second is None else second), SR(conserved)]
    for item in (rates, table):
        operands += [SR(-1)] if item is None else [SR(item.ndim), *(SR(size) for size in item.shape)]
        operands += [] if item is None else [SR(entry) for entry in item.ravel()]
    return BILINEAR(*operands)


def bilinears_in(expression: object) -> list[Expression]:
    """Return the calls of `bilinear` in an expression, outermost first.

    The compiler reads them to turn every form into sums over the classes of its densities, each call listed once, such
    as one call in `2 * bilinear(1.0, None, f, g)`. A call inside another call is left for the outer one. It finds the
    meetings of traders, of people who talk, and of particles that collide.

    Read more in the user guide, under Extend the library.

    Args:
        expression:
            A symbolic expression, a number, or a vector of expressions.

    Returns:
        forms:
            The bilinear calls, each listed once.

    See Also:
        unpack_bilinear: Reads the densities and tables of a call.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import bilinear
        >>> from skcomplex.core.components.laws.utils._bilinear import bilinears_in
        >>> f, g = var('f g')
        >>> len(bilinears_in(2 * bilinear(1.0, None, f, g)))
        1
    """
    return _calls_in(expression, {'bilinear'})


def unpack_bilinear(
    form: Expression,
) -> tuple[Expression, Expression | None, object, npt.NDArray[np.object_] | None, bool]:
    """Return the densities, the tables, and the conservation of a bilinear form.

    It reads back the arguments that `bilinear` packed into one call, such as `(f, g, -0.5, None, False)` for
    `bilinear(-0.5, None, f, g)`, so the compiler can build the sums of the form. It opens the forms of traders,
    conversations, and colliding particles.

    Read more in the user guide, under Extend the library.

    Args:
        form:
            The bilinear call.

    Returns:
        arguments:
            The first density, the second density or `None`, the rate as a number, an expression, or an array, the
            outcome array or `None`, and whether the form is conserved.

    See Also:
        bilinear: Builds the calls it reads.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import bilinear
        >>> from skcomplex.core.components.laws.utils._bilinear import unpack_bilinear
        >>> f, g = var('f g')
        >>> unpack_bilinear(bilinear(-0.5, None, f, g))
        (f, g, -0.5, None, False)
    """
    operands = form.operands()
    first, has_second, second, conserved = operands[:4]
    rest = operands[4:]
    tables: list[object] = []
    for _ in range(2):
        rank = int(rest[0])
        if rank < 0:
            tables.append(None)
            rest = rest[1:]
            continue
        shape = tuple(int(size) for size in rest[1 : 1 + rank])
        count = int(np.prod(shape))
        entries = rest[1 + rank : 1 + rank + count]
        rest = rest[1 + rank + count :]
        tables.append(entries[0] if not rank else np.asarray(entries, dtype=object).reshape(shape))
    outcomes = tables[1]
    return (
        first,
        second if bool(has_second) else None,
        tables[0],
        outcomes if isinstance(outcomes, np.ndarray) else None,
        bool(conserved),
    )


def _get_table(name: str, value: object, dimensions: tuple[int, ...]) -> npt.NDArray[np.object_]:
    """Return a rate or outcome table as an object array after checking its dimensions."""
    table = np.asarray(value, dtype=object)
    if table.ndim not in dimensions:
        expected = ' or '.join(str(item) for item in dimensions)
        error_msg = (
            f'The {name} of a bilinear form must be an array of {expected} dimensions, but got shape {table.shape}.'
        )
        raise ValidationError(error_msg)
    return table

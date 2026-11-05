"""Define the algebraic law the compiler writes for the value of an algebraic state at a frame."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ...core.laws import BoundaryLaw


class AlgebraicLaw(BoundaryLaw):
    """Value of an algebraic state at a frame, the solution of the equations the state appears in.

    The solvers integrate no algebraic state, so the law applies only to the state a frame records.

    Args:
        target:
            The symbol of the algebraic state.

        value:
            A symbolic expression or a `CallSpec` that gives the value.

        label:
            The name of the law, or the empty string for a law without one.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.engine.compile._algebraic_law import AlgebraicLaw
        >>> v, r = var('v r')
        >>> AlgebraicLaw(v, 2 * r).prepare().run({'r': 1.5}).value_
        3.0
    """

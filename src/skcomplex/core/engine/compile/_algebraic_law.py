"""Define the law that gives an algebraic state its value."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ...components.laws._boundary import _BoundaryLaw


@dataclass(frozen=True)
class AlgebraicLaw(_BoundaryLaw):
    """A law that gives an algebraic state its value at every recorded frame.

    An algebraic state is a value that no rate moves and that equations fix at every moment, such as the current through
    a resistor that Ohm's law sets from the voltage. The compiler solves those equations and writes the solution as this
    law. No solver integrates the state, so the law only fills in its value at the frames a run records. It gives the
    currents of a circuit, the torque of a motor, and the flows of a network of pipes.

    Read more in the user guide, under Laws.

    Args:
        target:
            The symbol of the algebraic state.

        value:
            A symbolic expression or a `CallSpec` that gives the value.

        label:
            The name of the law, or the empty string for a law without one.

    See Also:
        EquationBlocksLowering: Solves the equations that give algebraic states.
        RelaxationLaw: Relaxes a lattice until it is stable.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.engine.compile._algebraic_law import AlgebraicLaw
        >>> v, r = var('v r')
        >>> AlgebraicLaw(v, 2 * r).value
        2*r
    """

    @classmethod
    def _get_name(cls) -> str:
        """Return the name of the law in messages, the class users write it with."""
        return 'ConstraintLaw'

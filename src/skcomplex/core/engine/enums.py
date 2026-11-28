"""Define the kinds that only the engine reads."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from enum import IntEnum, StrEnum


class LinearProgramBound(IntEnum):
    """A kind of bound of a linear program that a value computed during a run sets.

    A linear program has a lower and an upper bound on every variable, its columns, and on every constraint, its rows,
    and a bound that depends on the states or parameters is computed again before every solve, such as the uptake limit
    of a nutrient that falls as the medium runs out. The engine marks which of the four bounds such a value sets. It
    sets the uptake limits of a cell, the capacity of a pipe, and the budget of a firm.

    Read more in the user guide, under Solvers.

    See Also:
        LinearProgramKernel: Solves the linear program of a problem.

    Examples:
        >>> from skcomplex.core.engine.enums import LinearProgramBound
        >>> LinearProgramBound.ROW_LOWER.name, int(LinearProgramBound.ROW_LOWER)
        ('ROW_LOWER', 2)
    """

    COLUMN_LOWER = 0
    COLUMN_UPPER = 1
    ROW_LOWER = 2
    ROW_UPPER = 3


class ReadKind(StrEnum):
    """A kind of read: where a symbol of a law takes its value for each member.

    A law written for one member reads its own value, a shared value, the value of its partner in a pair or along an
    edge, a neighbour, a reduction such as a mean, or the node a topology request starts from, such as `'neighbor'` for
    the infected neighbours of a person. The engine records one kind per symbol and gathers the values accordingly. It
    reads the neighbours of a cell, the partner of a trade, and the mean of a flock.

    Read more in the user guide, under Extend the library.

    See Also:
        ReferenceSpec: Says which value a symbol stands for and how it is read.

    Examples:
        >>> from skcomplex.core.engine.enums import ReadKind
        >>> ReadKind('neighbor')
        <ReadKind.NEIGHBOR: 'neighbor'>
    """

    MEMBER = 'member'
    SINGLE = 'single'
    FIRST = 'first'
    SECOND = 'second'
    NEIGHBOR = 'neighbor'
    DERIVED = 'derived'
    REDUCED = 'reduced'
    OTHER = 'other'
    SOURCE = 'source'


class LinearOperatorKind(StrEnum):
    """A kind of linear operator that a term of the rate of a field applies to a state.

    A rate on a grid often holds a fixed linear operator of a state, the Laplacian of diffusion, a partial derivative of
    transport, or the inverse of a Helmholtz operator for a field solved from a source, such as `D * laplacian(u)` for
    heat. The engine keeps such terms apart so that a solver can treat them implicitly or in Fourier space. It spreads
    heat in a plate, carries a pollutant in a river, and smooths a potential around charges.

    Read more in the user guide, under Solvers.

    See Also:
        LinearTermSpec: A term of the rate of a field built from these operators.

    Examples:
        >>> from skcomplex.core.engine.enums import LinearOperatorKind
        >>> from skcomplex.systems.physics.heat import HeatDiffusionSystem
        >>> chain = HeatDiffusionSystem().prepare().simulate(until=0).problem.linear[0].chain
        >>> LinearOperatorKind(chain[0][0])
        <LinearOperatorKind.LAPLACIAN: 'laplacian'>
    """

    LAPLACIAN = 'laplacian'
    PARTIAL = 'partial'
    INVERSE_HELMHOLTZ = 'inverse_helmholtz'


class ComplexComponentKind(StrEnum):
    """A quantity that a visual draws from a complex value, such as its modulus.

    A complex value has no single number to colour a cell or draw a line by, so a visual picks its real or imaginary
    component, its modulus, or its phase, such as `'phase'` for the angle of a wave function. Observers name the drawn
    series `<value>/<quantity>`. It draws the amplitude of a quantum state, the phase of an oscillator, and the response
    of a filter.

    Read more in the user guide, under Observe.

    See Also:
        VisualSpec: Says how an element is drawn.

    Examples:
        >>> from skcomplex.core.engine.enums import ComplexComponentKind
        >>> f'state/{ComplexComponentKind.PHASE}'
        'state/phase'
    """

    REAL = 'real'
    IMAG = 'imag'
    MODULUS = 'modulus'
    PHASE = 'phase'

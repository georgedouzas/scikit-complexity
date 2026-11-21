"""Define the bases of what a system compiles to."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ...base import BaseConfig
from ...errors import ValidationError
from ._run_state_spec import RunStateSpec


class BaseKernel(BaseConfig[object, RunStateSpec]):
    """An abstract kernel, a compiled piece of a problem that a solver evaluates.

    A kernel is a law turned into numbers, such as the matrix of a coupling between two qubits that a solver multiplies
    by the state. It stores what the compiler built in its constructor, such as compiled functions and the indices they
    read. `prepare(problem)` checks them and sets what a run needs, and `run(state)` acts at a `RunStateSpec`, the time,
    the state, the parameters, and the random generator, and keeps what it computes, such as the next state, in
    `values_`. A new kind of law comes with a subclass, the pass that builds it, and a solver that runs it, as for the
    jumps of molecules, the moves of spins, and the trades of agents.

    Read more in the user guide, under Extend the library.

    Attributes:
        values_:
            What the last run computed. Each kernel narrows its type.

    See Also:
        RunStateSpec: The state of a run a kernel acts at.
        OperatorKernel: A kernel of a matrix on a Hilbert space.

    Examples:
        >>> from skcomplex.core.engine.problems import BaseKernel
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.physics.common.quantum import QubitElement
        >>> problem = System(elements=[QubitElement('a')]).prepare().simulate(until=0).problem
        >>> isinstance(problem.generator[0], BaseKernel)
        True
    """

    values_: object

    def _prepare(self, context: object | None) -> None:
        """Derive nothing from the context, since a kernel prepares nothing.

        A kernel that derives facts sets them.
        """

    @staticmethod
    def _get_state_of(inputs: RunStateSpec | None) -> RunStateSpec:
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
    def _get_generator_of(state: RunStateSpec) -> np.random.Generator:
        """Return the random number generator of a state.

        An error is raised when the state has none.

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
    """An abstract search of the neighbours of members at a state.

    A search finds the neighbours a kernel reads at a state, such as the atoms of a gas within a radius of each atom. It
    stores the entries and the neighbourhood it reads in its constructor, and `run(state)` keeps the neighbours at a
    `RunStateSpec` in `pairs_`, as the row starts and the neighbours of every member, or the members of every pair. It
    finds the atoms near an atom, the birds a bird sees, and the people a walker meets.

    Read more in the user guide, under Extend the library.

    Attributes:
        pairs_:
            The neighbours of the last run. Each search narrows its type.

    See Also:
        BaseKernel: Reads the neighbours a search finds.

    Examples:
        >>> from skcomplex.core.engine.problems import BaseSearch
        >>> from skcomplex.core.engine.problems._pair_search import PairSearch
        >>> issubclass(PairSearch, BaseSearch)
        True
    """

    pairs_: object

    def _prepare(self, context: None) -> None:
        """Derive nothing from the context, since a search prepares nothing.

        A search that derives facts sets them.
        """

    @staticmethod
    def _get_state_of(inputs: RunStateSpec | None) -> RunStateSpec:
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
    """An abstract printer, which writes Sage expressions as code of a language.

    A printer stores its target in its constructor, `prepare(symbols)` takes the symbols the code reads, in order, and
    `run(expressions)` keeps the code of the expressions, or a function that evaluates them, in `code_`, such as
    `'3*_cse0'` for `3 * (x + 1) ** 2`. A printer names the types of the symbols and expressions it takes, which differ
    between targets. Printers write the laws of a reaction for NumPy, of a gas for Numba, and of a pair energy for
    LAMMPS.

    Read more in the user guide, under Extend the library.

    Attributes:
        code_:
            The code of the last run. Each printer narrows its type.

    See Also:
        ExpressionPrinter: Prints for NumPy, Numba, and LAMMPS.

    Examples:
        >>> from skcomplex.core.engine.problems import BasePrinter, ExpressionPrinter
        >>> isinstance(ExpressionPrinter(), BasePrinter)
        True
    """

    code_: object

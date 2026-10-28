"""Base class of solvers that translate a compiled problem into a third-party library."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from abc import abstractmethod

from ...errors import BackendError
from ...specs import ProblemSpec
from ._solver import Solver


class ExternalSolver(Solver):
    """Base class of solvers that translate a compiled problem into a third-party library.

    Subclasses implement the hooks `_get_name`, `_get_form`, `_unsupported`, `_build`, and `_advance`, and list the
    other forms the library runs in `_get_forms`.
    Before the first frame, `prepare` checks every element, interaction, and law, and raises one error that lists
    everything the library cannot run.
    """

    @classmethod
    def _get_name(cls) -> str:
        """Return the name of the library the solver wraps.

        Raises:
            NotImplementedError:
                The subclass does not implement the hook.
        """
        raise NotImplementedError

    @abstractmethod
    def _unsupported(self, problem: ProblemSpec) -> list[str]:
        """Return a description of every element, interaction, or law of a problem the library cannot run.

        Args:
            problem:
                The compiled problem.

        Returns:
            unsupported:
                One description per unsupported item, empty when the problem is supported.

        Raises:
            NotImplementedError:
                The subclass does not implement the method.
        """
        raise NotImplementedError

    @abstractmethod
    def _build(self, problem: ProblemSpec) -> None:
        """Build the library model of a supported problem, drawing its seeds from the generator `rng_`.

        Args:
            problem:
                The compiled problem.

        Raises:
            NotImplementedError:
                The subclass does not implement the method.
        """
        raise NotImplementedError

    def _prepare(self, problem: ProblemSpec) -> None:
        """Check that the library supports a problem and build its model.

        Args:
            problem:
                The compiled problem.

        Raises:
            BackendError:
                The problem has a form, a complex state, an element, an interaction, or a law the library cannot run.
        """
        forms = {self._get_form(), *self._get_forms()}
        unsupported = [] if problem.form in forms else [f'the {problem.form.value} form']
        unsupported += [f'the complex state {entry.name}' for entry in problem.states.entries if entry.is_complex]
        unsupported += self._unsupported(problem)
        if unsupported:
            items = '; '.join(unsupported)
            error_msg = f'The {self._get_name()} backend cannot run this system. Unsupported: {items}.'
            raise BackendError(error_msg)
        super()._prepare(problem)
        self._build(problem)

"""Base class of solvers that translate a compiled problem.

They translate it into a third-party library.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from abc import abstractmethod

from ...errors import BackendError, ContextError
from ..problems import ProblemSpec
from ._solver import Solver


class ExternalSolver(Solver):
    """Base class of solvers that translate a compiled problem.

    They translate it into a third-party library. Subclasses implement the hooks `_get_name`, `_get_form`,
    `_unsupported`, `_build`, and `_advance`, and list the other forms the library runs in `_get_forms`. Before the
    first frame, `prepare` checks every element, interaction, and law, and raises one error that lists everything the
    library cannot run.
    """

    @classmethod
    def _get_name(cls) -> str:
        """Return the name of the library the solver wraps.

        Returns:
            name:
                The name of the library.

        Raises:
            NotImplementedError:
                The subclass does not implement the hook.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_unsupported(self, problem: ProblemSpec) -> list[str]:
        """Return a description of every part of a problem the library cannot run.

        The parts are elements, interactions, or laws.

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
        """Build the library system of a supported problem.

        Its seeds are drawn from the generator `rng_`.

        Args:
            problem:
                The compiled problem.

        Raises:
            NotImplementedError:
                The subclass does not implement the method.
        """
        raise NotImplementedError

    def _prepare(self, context: ProblemSpec | None) -> None:
        """Check that the library supports a problem and build its system.

        Args:
            context:
                The compiled problem.

        Raises:
            ContextError:
                No problem is passed.

            BackendError:
                The problem has a form, a complex state, an element, an interaction, or a law the library cannot run.
        """
        if context is None:
            error_msg = 'A solver prepares on a compiled problem. Pass the problem to `prepare`.'
            raise ContextError(error_msg)
        problem = context
        forms = {self._get_form(), *self._get_forms()}
        unsupported = [] if problem.form in forms else [f'the {problem.form} form']
        unsupported += [f'the complex state {entry.name}' for entry in problem.states.entries if entry.is_complex]
        unsupported += self._get_unsupported(problem)
        if unsupported:
            items = '; '.join(unsupported)
            error_msg = f'The {self._get_name()} backend cannot run this system. Unsupported: {items}.'
            raise BackendError(error_msg)
        super()._prepare(problem)
        self._build(problem)

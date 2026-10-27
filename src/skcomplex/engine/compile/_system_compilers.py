"""Define the compilers that compile a whole system they claim into a problem of its form."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping

from ...core.system import System
from ..problems import BaseProblem
from ..utils import BaseEstimator


class BaseSystemCompiler(BaseEstimator):
    """Compiler that compiles a whole system it claims, such as a quantum or a rewriting system, into a problem.

    `prepare(system)` sets `claims_`, whether the compiler compiles the system, and `run(system, backend=...,
    solver=..., initial_state=...)` compiles a claimed system into a problem. Subclasses implement the hooks `_prepare`
    and `_run`. Validation of a claimed system checks its structure and leaves the rest to the compiler, which raises an
    error for a system it cannot compile.

    Attributes:
        claims_:
            Whether the compiler compiles the prepared system.
    """

    claims_: bool

    def _prepare(self, system: System) -> None:
        """Set `claims_`, whether the compiler compiles a system.

        Args:
            system:
                The system.

        Raises:
            NotImplementedError:
                The subclass does not implement the hook.
        """
        raise NotImplementedError

    def _run(
        self,
        system: System,
        *,
        backend: object,
        solver: object,
        initial_state: Mapping[str, object],
    ) -> BaseProblem:
        """Compile a claimed system into a problem.

        Args:
            system:
                The system, with the observables of the run.

            backend:
                The backend requested for the problem.

            solver:
                The solver the problem runs on, or `None` for the default solver of its form.

            initial_state:
                Initial values that replace those of the components, by the dotted name of the state.

        Returns:
            problem:
                The compiled problem.

        Raises:
            NotImplementedError:
                The subclass does not implement the hook.
        """
        raise NotImplementedError

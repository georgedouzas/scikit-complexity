"""Register the passes that compile a whole system of a formalism into a problem of its form."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from abc import ABC, abstractmethod
from collections.abc import Mapping

from ...core.base import Registry
from ...core.system import System
from ._problem import Problem

SYSTEM_PASSES = Registry['SystemPass']('system pass')


def register_system_pass(system_pass: 'SystemPass') -> None:
    """Register a pass that compiles the systems it claims, keeping it once when registered again.

    Args:
        system_pass:
            The pass.
    """
    SYSTEM_PASSES.register(system_pass, system_pass)


def claiming_pass(system: System) -> 'SystemPass | None':
    """Return the first registered pass that claims a system, or `None` when the general compiler compiles it.

    Args:
        system:
            The system.

    Returns:
        system_pass:
            The pass that compiles the system, or `None`.

    Examples:
        >>> from skcomplex.engine.compile import claiming_pass
        >>> from skcomplex.models import create
        >>> claiming_pass(create('lotka_volterra')) is None
        True
    """
    for key in SYSTEM_PASSES:
        system_pass = SYSTEM_PASSES.resolve(key)
        if system_pass.claims(system):
            return system_pass
    return None


class SystemPass(ABC):
    """Pass of a formalism that compiles a whole system it claims into a problem of the formalism's form.

    Validation of a claimed system checks its structure and leaves the rest to the pass, which raises an error for a
    system it cannot compile.
    """

    @abstractmethod
    def claims(self, system: System) -> bool:
        """Check whether the pass compiles a system.

        Args:
            system:
                The system.

        Returns:
            claims:
                Whether the pass compiles the system.

        Raises:
            NotImplementedError:
                The subclass does not implement the method.
        """
        raise NotImplementedError

    @abstractmethod
    def compile(
        self,
        system: System,
        *,
        backend: object,
        solver: object,
        initial_state: Mapping[str, object],
    ) -> Problem:
        """Compile a claimed system into a problem.

        Args:
            system:
                The system, with the observables of the run.

            backend:
                The backend requested for the problem.

            solver:
                The solver the problem runs on, `'auto'` or a solver instance.

            initial_state:
                Initial values that replace those of the components, by the dotted name of the state.

        Returns:
            problem:
                The compiled problem.

        Raises:
            NotImplementedError:
                The subclass does not implement the method.
        """
        raise NotImplementedError

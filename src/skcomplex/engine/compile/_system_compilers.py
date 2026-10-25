"""Register the compilers that compile a whole system of a formalism into a problem of its form."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping

from ...core.base import BaseProblem, Registry
from ...core.system import System

SYSTEM_COMPILERS = Registry['BaseSystemCompiler']('system compiler')


def register_system_compiler(compiler: 'BaseSystemCompiler') -> None:
    """Register a compiler of the systems it claims, keeping it once when registered again.

    Args:
        compiler:
            The compiler.
    """
    SYSTEM_COMPILERS.register(compiler, compiler)


def claiming_compiler(system: System) -> 'BaseSystemCompiler | None':
    """Return the first registered compiler that claims a system, or `None` when the general compiler compiles it.

    Args:
        system:
            The system.

    Returns:
        compiler:
            The compiler of the system, or `None`.

    Examples:
        >>> from skcomplex.engine.compile import claiming_compiler
        >>> from skcomplex.models import create
        >>> claiming_compiler(create('lotka_volterra')) is None
        True
    """
    for key in SYSTEM_COMPILERS:
        compiler = SYSTEM_COMPILERS.resolve(key)
        if compiler.claims(system):
            return compiler
    return None


class BaseSystemCompiler:
    """Compiler of a formalism that compiles a whole system it claims into a problem of the formalism's form.

    Validation of a claimed system checks its structure and leaves the rest to the compiler, which raises an error for a
    system it cannot compile.
    """

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

    def compile(
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

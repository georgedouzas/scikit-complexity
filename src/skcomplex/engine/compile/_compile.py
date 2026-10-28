"""Compile a system into a problem of one mathematical form with the compiler that claims it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from typing import cast

from ...core.system import System
from ...specs import BaseProblemSpec, ObservableSpec
from ..backends import BaseBackend
from ._base import BaseCompiler
from ._pass_compiler import PassCompiler
from .passes._quantum_compiler import QuantumCompiler
from .passes._rewriting_compiler import RewritingCompiler


def compile_system(
    system: System,
    backend: BaseBackend | None = None,
    observables: Sequence[ObservableSpec] = (),
    *,
    solver: object = None,
    initial_state: Mapping[str, object] | None = None,
) -> BaseProblemSpec:
    """Compile a system into a problem of the form its law kinds determine.

    The compilers of whole systems, of quantum and of rewriting systems, are asked in turn whether they claim the
    system, and the general `PassCompiler` compiles every other system. The compiler that claims it validates the
    system in `prepare` and compiles it in `run`.

    Args:
        system:
            The system to compile.

        backend:
            The backend instance, or `None` for `NumbaBackend` on systems whose groups hold at least 1000 members in
            total and `NumpyBackend` on the others.

        observables:
            The observables to compute at every frame, labelled by their path from the root, such as
            `village/mean_wealth` for an observable of the nested system `village`.

        solver:
            The solver the problem runs on, or `None` for the default solver of its form, whose limits compiling
            checks, such as the largest dimension of a Hilbert space.

        initial_state:
            Initial values that replace those of the components, by the dotted name of the state, such as
            `{'rabbits.n': 20}`. The whole state of a quantum system is `<system label>.state`, which takes a vector, a
            density matrix, or `'ground'` for the lowest eigenstate of the Hamiltonian at the initial parameters.

    Returns:
        problem:
            The compiled problem.

    Raises:
        ValidationError:
            A child has the wrong type, a value is outside its domain or has the wrong shape, or a law is malformed.

        LabelError:
            A label is not valid, duplicated, or produces a symbol collision.

        RoleError:
            A role refers to an element that does not exist, or an interaction reads groups of different sizes member
            by member.

        SpaceError:
            A vector variable has no enclosing space, or a space is not valid.

        IncompatibleLawsError:
            The system mixes law kinds that no form supports.

        BackendError:
            An agent group declares a complex state.

        ValidationError:
            Validation finds problems of more than one kind, all listed in the message.

    Examples:
        >>> from skcomplex.engine.compile import compile_system
        >>> from skcomplex.core.system import System
        >>> from skcomplex.models.biology.ecology import ExponentialGrowthInteraction, PopulationElement
        >>> problem = compile_system(
        ...     System(
        ...         elements=[PopulationElement('rabbits', n=10)],
        ...         interactions=[ExponentialGrowthInteraction('growth', target='rabbits', r=0.5)],
        ...     ),
        ... )
        >>> problem.form.name
        'ODE'
        >>> problem.equations
        {'rabbits.n': n_rabbits*r_growth}
    """
    compilers: tuple[BaseCompiler, ...] = (
        QuantumCompiler(backend, solver, observables),
        RewritingCompiler(backend, solver, observables),
        PassCompiler(backend=backend, solver=solver, observables=observables),
    )
    compiler = next(item for item in compilers if item.check().prepare(system).claims_)
    return cast('BaseProblemSpec', compiler.run(initial_state))

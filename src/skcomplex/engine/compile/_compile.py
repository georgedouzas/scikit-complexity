"""Compile a system into a problem of one mathematical form with a compiler."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence

from sklearn.base import clone

from ...core.system import System
from ...specs import ObservableSpec
from ..backends import BaseBackend
from ..problems import BaseProblemSpec
from ._base import BaseCompiler
from ._pass_compiler import PassCompiler


def compile_system(
    system: System,
    backend: BaseBackend | None = None,
    observables: Sequence[ObservableSpec] = (),
    *,
    solver: object = None,
    initial_state: Mapping[str, object] | None = None,
    compiler: BaseCompiler | None = None,
) -> BaseProblemSpec:
    """Compile a system into a problem of the form its law kinds determine.

    The compiler validates the system in `prepare` and compiles it in `run`. `PassCompiler` compiles every system but
    operator ones, which `OperatorCompiler` compiles, and a model package can bring the compiler of a form of its own.

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
            `{'rabbits.n': 20}`. The state of an operator system is `<system label>.state`, which takes a vector, a
            density matrix, or `'ground'` for the lowest eigenstate of the generator at the initial parameters.

        compiler:
            The compiler, whose backend, solver, and observables this function sets, or `None` for `PassCompiler`.

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
        >>> from skcomplex.models.biology.common.populations import ExponentialGrowthInteraction, PopulationElement
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
    compiler = PassCompiler() if compiler is None else clone(compiler)
    compiler.set_params(backend=backend, solver=solver, observables=observables)
    return compiler.check().prepare(system).run(initial_state).problem_

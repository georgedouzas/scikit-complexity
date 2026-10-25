"""Compile a system into a problem of one mathematical form, continuing from the state its validation planned."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np

from ...core.base import BaseProblem, Form, ValidationError
from ...core.problems import BaseBackend, Partition, Problem, with_initial_state
from ...core.system import Observable, System
from ._compilation import Compilation
from ._declarations import is_interaction
from ._observables import _with_observables, check_held_reads
from ._pipeline import _registered_passes, run_passes
from ._planner import _Plan
from ._system_compilers import claiming_compiler
from ._tree import Node
from ._validation import validate
from .passes._core import CORE_PASSES


def _compiled(compilation: Compilation) -> Problem:
    """Return the problem of a compilation, running every pass it has not run yet."""
    return cast('Problem', run_passes(compilation, _registered_passes(CORE_PASSES)).problem)


def _partitioned(compilation: Compilation, plan: list[tuple[str, frozenset[str]]]) -> Problem:
    """Return the problem of a system split into partitions, each compiled with the laws of its owners."""
    partitions = []
    for name, owners in plan:
        problem = _compiled(compilation.fork(owners))
        owned = [entry.indices for entry in problem.states.entries if entry.name.rpartition('.')[0] in owners]
        indices = np.concatenate([np.arange(item.start, item.stop) for item in owned]) if owned else np.zeros(0)
        partitions.append(Partition(name, problem, indices.astype(np.intp), owners))
    first = partitions[0].problem
    equations = {
        key: value
        for item in partitions
        for key, value in item.problem.equations.items()
        if key.partition('.')[0] in item.owners
    }
    symbols = {
        key: item.problem.equation_symbols[key]
        for item in partitions
        for key in item.problem.equations
        if key in equations
    }
    return Problem(
        form=Form.COUPLED,
        states=first.states,
        parameters=first.parameters,
        initial_state=first.initial_state,
        initial_parameters=first.initial_parameters,
        time=first.time,
        symbols=first.symbols,
        equations=equations,
        equation_symbols=symbols,
        has_callables=any(item.problem.has_callables for item in partitions),
        has_group_sums=any(item.problem.has_group_sums for item in partitions),
        elements=first.elements,
        interactions=first.interactions,
        system=compilation.system,
        agents=tuple(agents for item in partitions for agents in item.problem.agents),
        agent_space=next((item.problem.agent_space for item in partitions if item.problem.agents), None),
        partitions=tuple(partitions),
    )


def _planned_problem(compilation: Compilation) -> Problem:
    """Return the problem of a validated compilation, one problem or one partition per planned subsystem.

    The owners of a partition are the elements and groups of its subsystem whose lowered laws it holds.
    """
    plan = cast('_Plan', compilation.plan)
    if len(plan.partitions) <= 1:
        return _compiled(compilation)
    kinds = cast('dict[str, set[type]]', run_passes(compilation, _registered_passes(CORE_PASSES), until='kinds').kinds)
    owners = [(item.name, frozenset(owner for owner in kinds if item.holds(owner))) for item in plan.partitions]
    return _partitioned(compilation, owners)


def compiled_problem(system: 'System | BaseProblem', purpose: str) -> Problem:
    """Return the compiler's problem of a system, or the problem itself, for work that reads the compiler's fields.

    Args:
        system:
            A system, or a problem compiled from one.

        purpose:
            What needs the compiler's problem, such as `'Linear stability'`, for the error message.

    Returns:
        problem:
            The compiler's problem.

    Raises:
        ValidationError:
            A formalism compiles the system to a problem of its own, which the work cannot read.

    Examples:
        >>> from skcomplex.engine.compile import compiled_problem
        >>> from skcomplex.models import create
        >>> compiled_problem(create('lotka_volterra'), 'Linear stability').form.name
        'ODE'
    """
    problem = system if isinstance(system, BaseProblem) else compile_system(system)
    if isinstance(problem, Problem):
        return problem
    error_msg = (
        f'{purpose} reads the problem of the general compiler, but the system compiles to a '
        f'{type(problem).__name__} of the {problem.form.value} form.'
    )
    raise ValidationError(error_msg)


def compile_system(
    system: System,
    backend: str | BaseBackend = 'auto',
    observables: Sequence[Observable] = (),
    *,
    solver: object = 'auto',
    initial_state: Mapping[str, object] | None = None,
) -> BaseProblem:
    """Compile a system into a problem of the form its law kinds determine.

    Elements and groups are compiled once each, whatever the size of a group. Laws written for one member are
    evaluated for every member, and pair interactions are evaluated for every pair within their cutoff. Compilation
    continues from the state that validation planned, so the system tree is flattened once.

    Args:
        system:
            The system to compile.

        backend:
            `'auto'`, a registered backend name such as `'numpy'` or `'numba'`, or a backend instance. `'auto'` picks
            Numba for systems whose groups hold at least 1000 members in total and NumPy otherwise.

        observables:
            The observables to compute at every frame, labelled by their path from the root, such as
            `village/mean_wealth` for an observable of the nested system `village`.

        solver:
            The solver the problem runs on, `'auto'` or a solver instance, whose limits compiling checks, such as the
            largest dimension of a Hilbert space.

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
            The backend is not registered, or an agent group declares a complex state.

        ValidationError:
            Validation finds problems of more than one kind, all listed in the message.

    Examples:
        >>> from skcomplex.engine.compile import compile_system
        >>> from skcomplex.core.system import System
        >>> from skcomplex.models.biology.ecology import ExponentialGrowthInteraction, PopulationComponent
        >>> problem = compile_system(
        ...     System(
        ...         components=[PopulationComponent('rabbits', n=10)],
        ...         interactions=[ExponentialGrowthInteraction('growth', target='rabbits', r=0.5)],
        ...     ),
        ... )
        >>> problem.form.name
        'ODE'
        >>> problem.equations
        {'rabbits.n': n_rabbits*r_growth}
    """
    report = validate(system, backend=backend, observables=observables)
    report.raise_errors()
    observed = _with_observables(system, observables)
    initial = dict(initial_state or {})
    compiler = claiming_compiler(observed)
    if compiler is not None:
        return cast('BaseProblem', compiler.run(observed, backend=backend, solver=solver, initial_state=initial))
    compilation = cast('Compilation', report.compilation)
    problem = _planned_problem(compilation)
    check_held_reads(problem, list(compilation.held))
    return with_initial_state(problem, initial)


def partition_system(system: System, backend: str | BaseBackend = 'auto') -> Problem:
    """Compile a system as one partition per subsystem, also when the whole system fits one form.

    The elements of the root system form one partition and every child subsystem forms another, so a co-simulation
    can advance them with separate solvers.

    Args:
        system:
            The system to compile.

        backend:
            The backend of every partition.

    Returns:
        problem:
            The partitioned problem.

    Raises:
        IncompatibleLawsError:
            A partition mixes law kinds that no form supports.

    Examples:
        >>> from skcomplex.engine.compile import partition_system
        >>> from skcomplex.core.system import System
        >>> from skcomplex.models.biology.ecology import ExponentialGrowthInteraction, PopulationComponent
        >>> system = System(
        ...     components=[
        ...         System(
        ...             'a',
        ...             components=[PopulationComponent('x')],
        ...             interactions=[ExponentialGrowthInteraction('g', target='x')],
        ...         ),
        ...         System(
        ...             'b',
        ...             components=[PopulationComponent('y')],
        ...             interactions=[ExponentialGrowthInteraction('g', target='y')],
        ...         ),
        ...     ],
        ... )
        >>> [partition.name for partition in partition_system(system).partitions]
        ['a', 'b']
    """
    compilation = run_passes(Compilation(system, backend), _registered_passes(CORE_PASSES), until='nodes')
    elements = {node.path for node in cast('list[Node]', compilation.nodes) if not is_interaction(node)}
    own = frozenset(path for path in elements if '/' not in path)
    plan = [(system.label, own)] if own else []
    for child in system.systems:
        plan.append((child.label, frozenset(path for path in elements if path.startswith(f'{child.label}/'))))
    return _partitioned(compilation, plan)

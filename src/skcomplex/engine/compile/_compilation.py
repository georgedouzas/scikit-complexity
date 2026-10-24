"""Hold the state of one compilation, the fields its passes produce in order from the system to the problem."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from ..base import Expression, Form
from ..laws import Event, Law
from ..system import System
from ._backends import Backend
from ._census import Census
from ._declarations import Block, Declarations
from ._jacobians import ExactJacobian
from ._kernels import Lowered, Output
from ._planner import _Plan
from ._problem import Problem, _MarkovMatrix
from ._reductions import ReductionPass
from ._renewals import RenewalLaw
from ._specs import KernelSpec
from ._stochastic import DrawLowering, MatrixLaw
from ._tree import Node

type Laws = list[tuple[KernelSpec, list[Law]]]
type BlockLaws = list[tuple[KernelSpec, list[Law], Block]]

INPUTS = ('system', 'backend', 'solver', 'owners', 'held')
PLANNING = frozenset({'nodes', 'census', 'plan'})


@dataclass
class Compilation:
    """State of one compilation of a system, whose fields its passes read and write in order.

    The fields from `nodes` to `problem` are produced in the order they are listed. A pass may also declare hidden
    states and parameters on the layout, which only grows, and record facts of its own under `facts`.

    Args:
        system:
            The system, with the observables of the run.

        backend:
            `'auto'`, a registered backend name, or a backend instance.

        solver:
            `'auto'`, a registered solver name, or a solver instance.

        owners:
            The paths of the elements and groups whose laws a partition keeps, or `None` to keep every law.

        held:
            The paths of the observables that roles bind to and laws read only at synchronisation points.

        nodes:
            The elements, groups, and interactions of the system tree, flattened once.

        census:
            The laws every component declares, with their owners.

        plan:
            The partitions of the system.

        layout:
            The states and parameters of every component, with the blocks of their symbols.

        specs:
            The kernel spec of every component with its block.

        collected:
            The laws of every kernel spec as declared, with its block.

        apart:
            The laws set apart from the kernels of their spec, which a pass of their own compiles.

        lowered:
            The laws of every kernel spec with complex states split and Markov laws taken out.

        markov:
            The Markov laws of every kernel spec.

        matrices:
            The Markov matrices, checked at the initial parameters.

        ordered:
            The laws with derivatives and lags read through companion states and equations solved.

        delayed:
            The laws with delayed states read through hidden parameters and events taken out.

        corrected:
            The laws with the drift correction of every noise law of a convention.

        shifted:
            The laws with member offset reads lowered to neighbour reads.

        drawn:
            The laws of the owners of the compilation, masked to the rows present, with draws read from parameters.

        drawn_apart:
            The laws set apart of the owners of the compilation.

        kinds:
            The law classes of every owner.

        drawing:
            The draws of the laws.

        form:
            The form of the problem.

        reduced:
            The laws with every reduction read through its placeholder.

        reduced_apart:
            The laws set apart, with every reduction read through its placeholder.

        reductions:
            The reductions the laws read.

        kernels:
            The kernels of every law kind.

        functions:
            The numeric functions, the equations, and the backends of the problem.

        jacobian:
            The exact Jacobian of the drift, or `None`.

        problem:
            The compiled problem.

        delays:
            The delayed states with their hidden parameters.

        placeholders:
            The placeholder of every delayed state read.

        events:
            The events of every kernel spec.

        renewals:
            The renewal laws with the hidden states that count them.

        outputs:
            The outputs of the laws that a pass of a formalism lowers itself, by the identity of the law, which the
            kernels take in place of the outputs of the law's value.

        facts:
            The facts that the passes of formalisms record, by name.

        done:
            The passes that ran.
    """

    system: System
    backend: str | Backend = 'auto'
    solver: object = 'auto'
    owners: frozenset[str] | None = None
    held: tuple[str, ...] = ()
    nodes: list[Node] | None = None
    census: Census | None = None
    plan: _Plan | None = None
    layout: Declarations | None = None
    specs: list[tuple[KernelSpec, Block]] | None = None
    collected: BlockLaws | None = None
    apart: Laws | None = None
    lowered: BlockLaws | None = None
    markov: list[tuple[KernelSpec, MatrixLaw]] | None = None
    matrices: tuple[_MarkovMatrix, ...] | None = None
    ordered: Laws | None = None
    delayed: Laws | None = None
    corrected: Laws | None = None
    shifted: Laws | None = None
    drawn: Laws | None = None
    drawn_apart: Laws | None = None
    kinds: dict[str, set[type]] | None = None
    drawing: DrawLowering | None = None
    form: Form | None = None
    reduced: Laws | None = None
    reduced_apart: Laws | None = None
    reductions: ReductionPass | None = None
    kernels: Lowered | None = None
    functions: dict[str, Any] | None = None
    jacobian: ExactJacobian | None = None
    problem: Problem | None = None
    delays: list[dict[str, Any]] = field(default_factory=list)
    placeholders: dict[tuple[int, str, int, str], Expression] = field(default_factory=dict)
    events: list[tuple[KernelSpec, Event]] = field(default_factory=list)
    renewals: list[RenewalLaw] = field(default_factory=list)
    outputs: dict[int, list[Output]] = field(default_factory=dict)
    facts: dict[str, Any] = field(default_factory=dict)
    done: set[object] = field(default_factory=set)

    def present(self) -> set[str]:
        """Return the fields the compilation holds: its inputs, the fields set so far, and the facts.

        Returns:
            fields:
                The names of the fields.
        """
        fields = {item.name for item in dataclasses.fields(self) if getattr(self, item.name) is not None}
        return fields | set(self.facts)

    def fork(self, owners: frozenset[str]) -> 'Compilation':
        """Return the compilation of a partition, which shares the nodes, the census, and the plan.

        Args:
            owners:
                The paths of the elements and groups whose laws the partition keeps.

        Returns:
            compilation:
                The compilation of the partition.
        """
        kept = {name: getattr(self, name) for name in (*INPUTS, *PLANNING)}
        done = {item for item in self.done if set(getattr(item, 'writes', ())) <= PLANNING}
        return Compilation(**{**kept, 'owners': owners}, done=done)

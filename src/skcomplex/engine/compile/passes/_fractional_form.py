"""Pass that gives an ODE problem with a state of a fractional order among its owners the fractional form."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from typing import cast

from ....core.errors import BackendError, SolverError
from ....core.spaces import ContinuousTime
from ...problems import (
    Form,
    OrderKind,
    VariableKind,
    declarations_of,
)
from .._census import Census
from .._census_record import CensusRecord
from .._compilation import Compilation
from .._declarations import Declarations
from .._forms import _FORMS
from .._issue import Issue
from .._pipeline import BasePass
from .._tree import Node, _declared_times
from ._fractional_state import FractionalState


class FractionalFormPass(BasePass):
    """Pass that gives an ODE problem with a state of a fractional order among its owners the fractional form."""

    reads = ('form', 'nodes', 'layout')
    writes = ('form',)

    def _run(self, compilation: Compilation) -> None:
        """Give the problem the fractional form when an owned state has a fractional order.

        Args:
            compilation:
                The compilation state.
        """
        if compilation.form == Form.ODE and self._owned_fractional(compilation):
            compilation.form = Form.FRACTIONAL

    @staticmethod
    def _fractional_states(nodes: list[Node]) -> list[FractionalState]:
        """Return every state of a fractional order of the elements and groups of a system.

        Args:
            nodes:
                The nodes of the system.

        Returns:
            states:
                The states of a fractional order, in tree order.
        """
        found = []
        for node in nodes:
            if node._is_interaction():
                continue
            for name, declaration in declarations_of(node.component_type, VariableKind.STATE).items():
                if declaration.is_fractional:
                    found.append(FractionalState(node.path, f'{node.path}.{name}', declaration.order))
        return found

    @staticmethod
    def _owned_fractional(compilation: Compilation) -> list[FractionalState]:
        """Return the fractional states of the owners of a compilation that its layout declares, leaving out agents."""
        declared = {entry.name for entry in cast('Declarations', compilation.layout).states}
        owners = compilation.owners
        return [
            item
            for item in FractionalFormPass._fractional_states(cast('list[Node]', compilation.nodes))
            if item.name in declared and (owners is None or item.path in owners)
        ]


def _support_issues(states: list[FractionalState], solver: object, backend: object) -> list[Issue]:
    """Return the fractional states that the solver or the backend passed does not integrate."""
    issues = []
    for item in states:
        if solver is not None and OrderKind.FRACTIONAL not in type(solver).supports.orders:
            message = (
                f'The solver {type(solver).__name__} does not integrate {item.name}, a state of order {item.order}.'
            )
            fix = 'Pass a solver that integrates fractional orders, such as `FractionalABMSolver`.'
            issues.append(Issue('order-unsupported', item.path, message, fix, error=SolverError))
        if backend is not None and OrderKind.FRACTIONAL not in type(backend).supports.orders:
            message = f'The backend {type(backend).__name__} does not run {item.name}, a state of order {item.order}.'
            fix = 'Pass a backend that runs fractional orders.'
            issues.append(Issue('order-unsupported', item.path, message, fix, error=BackendError))
    return issues


def _law_issues(states: list[FractionalState], records: list[CensusRecord]) -> list[Issue]:
    """Return the laws on fractional states of a class that takes the problem off the ODE form, the fractional one."""
    names = {item.name for item in states}
    ode = set().union(*(kinds for kinds, form in _FORMS.items() if form == Form.ODE))
    others = tuple(set().union(*(kinds for kinds, form in _FORMS.items() if form != Form.ODE)) - ode)
    issues = []
    for record in records:
        targets = sorted({entry.name for entry in record.targets} & names)
        if not targets or not isinstance(record.law, others):
            continue
        name = type(record.law).__name__
        message = (
            f'The {name} of {record.path} targets {", ".join(targets)}, a state of a fractional order, but no '
            f'fractional solver runs {name}.'
        )
        fix = 'Declare an integer order, or change the state with rate laws alone.'
        issues.append(Issue('solver-unsupported', record.path, message, fix, error=SolverError))
    return issues


def _order_issues(compilation: Compilation) -> list[Issue]:
    """Return the fractional states with an order outside `(0, 1)`, a time that is not continuous, or no support.

    Args:
        compilation:
            The compilation state after planning.

    Returns:
        issues:
            The issues.
    """
    states = FractionalFormPass._fractional_states(cast('list[Node]', compilation.nodes))
    if not states:
        return []
    times = _declared_times(compilation.system)
    issues = []
    for item in states:
        if not 0 < item.order < 1:
            message = (
                f'The state {item.name} declares the order {item.order}, which is neither an integer nor in (0, 1).'
            )
            fix = 'Declare a fractional order in the interval (0, 1), or an integer order of at least 1.'
            issues.append(Issue('fractional-order', item.path, message, fix))
        time = times.get(item.path.rpartition('/')[0])
        if time is not None and time is not ContinuousTime:
            message = (
                f'The state {item.name} has the fractional order {item.order}, but the system declares {time} time.'
            )
            fix = 'Declare continuous time, the only time in which fractional derivatives are defined.'
            issues.append(Issue('fractional-time', item.path, message, fix))
    issues += _law_issues(states, cast('Census', compilation.census).records)
    return issues + _support_issues(states, compilation.solver, compilation.backend)

"""Pass that gives an ODE problem with a state of a fractional order among its owners the fractional form."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from typing import cast

from ....core.spaces import ContinuousTime
from ....core.system import System
from ....errors import BackendError, SolverError
from ....specs import CensusRecordSpec, CensusSpec, DeclarationsSpec, FractionalStateSpec, IssueSpec, NodeSpec
from ....types import FormKind, OrderKind, VariableKind
from ..._utils import declarations_of
from .._base import BasePass
from .._compilation import Compilation
from .._forms import _FORMS


class FractionalFormPass(BasePass):
    """Pass that gives an ODE problem with a state of a fractional order among its owners the fractional form."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('form', 'nodes', 'layout')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('form',)

    def _run(self, inputs: object = None) -> None:
        """Give the problem the fractional form when an owned state has a fractional order."""
        compilation = self.compilation_
        if compilation.form == FormKind.ODE and self._owned_fractional(compilation):
            compilation.form = FormKind.FRACTIONAL

    @staticmethod
    def _declared_times(system: System, scope: str = '', inherited: type | None = None) -> dict[str, type | None]:
        """Return the class of the declared time of every system of a tree by path.

        A nested system without a time of its own inherits the time of its parent.

        Args:
            system:
                The root of the tree.

            scope:
                The path of the system.

            inherited:
                The time of the parent system.

        Returns:
            times:
                The time class of every system by path, such as `ContinuousTime`, or `None` for a system without one.
        """
        time = type(system.time) if system.time is not None else inherited
        times = {scope: time}
        for child in system.systems:
            path = f'{scope}/{child.label}' if scope else child.label
            times.update(FractionalFormPass._declared_times(child, path, time))
        return times

    @staticmethod
    def _fractional_states(nodes: list[NodeSpec]) -> list[FractionalStateSpec]:
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
            if node.interaction:
                continue
            for name, declaration in declarations_of(node.component_type, VariableKind.STATE).items():
                if declaration.is_fractional:
                    found.append(FractionalStateSpec(node.path, f'{node.path}.{name}', declaration.order))
        return found

    @staticmethod
    def _owned_fractional(compilation: Compilation) -> list[FractionalStateSpec]:
        """Return the fractional states of the owners of a compilation that its layout declares, leaving out agents."""
        declared = {entry.name for entry in cast('DeclarationsSpec', compilation.layout).states}
        owners = compilation.owners
        return [
            item
            for item in FractionalFormPass._fractional_states(cast('list[NodeSpec]', compilation.nodes))
            if item.name in declared and (owners is None or item.path in owners)
        ]

    @staticmethod
    def _support_issues(states: list[FractionalStateSpec], solver: object, backend: object) -> list[IssueSpec]:
        """Return the fractional states that the solver or the backend passed does not integrate."""
        issues = []
        for item in states:
            if solver is not None and OrderKind.FRACTIONAL not in type(solver)._get_support().orders:
                message = (
                    f'The solver {type(solver).__name__} does not integrate {item.name}, a state of order {item.order}.'
                )
                fix = 'Pass a solver that integrates fractional orders, such as `FractionalABMSolver`.'
                issues.append(IssueSpec('order-unsupported', item.path, message, fix, error=SolverError))
            if backend is not None and OrderKind.FRACTIONAL not in type(backend)._get_support().orders:
                message = (
                    f'The backend {type(backend).__name__} does not run {item.name}, a state of order {item.order}.'
                )
                fix = 'Pass a backend that runs fractional orders.'
                issues.append(IssueSpec('order-unsupported', item.path, message, fix, error=BackendError))
        return issues

    @staticmethod
    def _law_issues(states: list[FractionalStateSpec], records: list[CensusRecordSpec]) -> list[IssueSpec]:
        """Return the laws on fractional states of a class that takes the problem off the ODE and fractional forms."""
        names = {item.name for item in states}
        ode = set().union(*(kinds for kinds, form in _FORMS.items() if form == FormKind.ODE))
        others = tuple(set().union(*(kinds for kinds, form in _FORMS.items() if form != FormKind.ODE)) - ode)
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
            issues.append(IssueSpec('solver-unsupported', record.path, message, fix, error=SolverError))
        return issues

    @staticmethod
    def _order_issues(compilation: Compilation) -> list[IssueSpec]:
        """Return the fractional states with an order outside `(0, 1)`, a time that is not continuous, or no support.

        Args:
            compilation:
                The compilation state after planning.

        Returns:
            issues:
                The issues.
        """
        states = FractionalFormPass._fractional_states(cast('list[NodeSpec]', compilation.nodes))
        if not states:
            return []
        times = FractionalFormPass._declared_times(compilation.system)
        issues = []
        for item in states:
            if not 0 < item.order < 1:
                message = (
                    f'The state {item.name} declares the order {item.order}, which is neither an integer nor in (0, 1).'
                )
                fix = 'Declare a fractional order in the interval (0, 1), or an integer order of at least 1.'
                issues.append(IssueSpec('fractional-order', item.path, message, fix))
            time = times.get(item.path.rpartition('/')[0])
            if time is not None and time is not ContinuousTime:
                message = (
                    f'The state {item.name} has the fractional order {item.order}, but the system declares {time} time.'
                )
                fix = 'Declare continuous time, the only time in which fractional derivatives are defined.'
                issues.append(IssueSpec('fractional-time', item.path, message, fix))
        issues += FractionalFormPass._law_issues(states, cast('CensusSpec', compilation.census).records)
        return issues + FractionalFormPass._support_issues(states, compilation.solver, compilation.backend)

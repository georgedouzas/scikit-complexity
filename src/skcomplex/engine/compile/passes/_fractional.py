"""Compile the fractional form: states of fractional order, fractional noise, and the solvers that integrate them."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import cast

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression
from sage.modules.free_module_element import FreeModuleElement

from ....core.errors import BackendError, SolverError, ValidationError
from ....core.laws import BaseLaw, FractionalNoiseLaw, NoiseLaw
from ....core.spaces import ContinuousTime
from ....core.types import Vector
from ...problems import (
    Form,
    FractionalRows,
    Kernel,
    Layout,
    Lowered,
    OrderKind,
    Problem,
    VariableKind,
    components_of,
    declarations_of,
    symbols_of,
)
from .._census import Census, CensusRecord
from .._compilation import BlockLaws, Compilation
from .._declarations import Declarations
from .._forms import _FORMS
from .._issues import Issue
from .._pipeline import BasePass
from .._specs import KernelSpec
from .._tree import Node, _declared_times


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


@dataclass(frozen=True)
class FractionalState:
    """State of a fractional order with the path of its component.

    Args:
        path:
            The path of the element or group.

        name:
            The dotted name of the state.

        order:
            The order of the state.
    """

    path: str
    name: str
    order: float


class CheckFractionalNoisePass(BasePass):
    """Pass that checks the Hurst index and amplitude of fractional noise laws, with Hurst index 1/2 as noise laws."""

    reads = ('lowered', 'layout')
    writes = ('lowered',)

    def _run(self, compilation: Compilation) -> None:
        """Check the fractional noise laws.

        Args:
            compilation:
                The compilation state.
        """
        declarations = cast('Declarations', compilation.layout)
        compilation.lowered = [
            (spec, self._checked_fractional(spec, laws, declarations), block)
            for spec, laws, block in cast('BlockLaws', compilation.lowered)
        ]

    @staticmethod
    def _checked_hurst(spec: KernelSpec, law: FractionalNoiseLaw, declarations: Declarations, description: str) -> None:
        """Raise an error when the Hurst index of a fractional noise law is not a number or a parameter in `(0, 1)`."""
        hurst = SR(law.hurst)
        if hurst.is_numeric():
            values = np.asarray([float(hurst)])
        else:
            reference = spec.references.get(hurst)
            if reference is None or reference.is_state:
                error_msg = (
                    f'The {description} of {spec.owner} has the Hurst index {hurst}. Give a number or a parameter.'
                )
                raise ValidationError(error_msg)
            parameters = np.concatenate([np.zeros(0), *declarations.initial_parameters])
            values = np.asarray(reference.column_of(np.zeros(0), parameters), dtype=float)
        if not np.all((values > 0) & (values < 1)):
            error_msg = (
                f'The {description} of {spec.owner} has the Hurst index {hurst} = {values[0]}, outside the open '
                'interval (0, 1). Give a Hurst index between 0 and 1.'
            )
            raise ValidationError(error_msg)

    @staticmethod
    def _checked_fractional(spec: KernelSpec, laws: list[BaseLaw], declarations: Declarations) -> list[BaseLaw]:
        """Return laws with every fractional noise law of Hurst index 1/2 as a noise law, after checking the others."""
        result: list[BaseLaw] = []
        for law in laws:
            if not isinstance(law, FractionalNoiseLaw):
                result.append(law)
                continue
            description = CheckFractionalNoisePass._fractional_description(spec, law)
            CheckFractionalNoisePass._checked_hurst(spec, law, declarations, description)
            if SR(law.hurst).is_numeric() and float(SR(law.hurst)) == 0.5:  # noqa: PLR2004
                result.append(NoiseLaw(law._get_targets()[0], law.value, law.order))
                continue
            values = [item for item in components_of(law.value) if isinstance(item, Expression)]
            if not isinstance(law.value, Expression | FreeModuleElement) and not SR(law.value).is_numeric():
                error_msg = (
                    f'The {description} of {spec.owner} has an amplitude that is not symbolic. Give an expression.'
                )
                raise ValidationError(error_msg)
            read = [
                item
                for item in sorted(symbols_of(values), key=str)
                if (reference := spec.references.get(item)) is not None and reference.is_state
            ]
            if read:
                error_msg = (
                    f'The {description} of {spec.owner} has an amplitude that depends on the state '
                    f'{KernelSpec._state_name(spec, read[0])}. Only state-independent amplitudes are supported for a '
                    'Hurst index other than 1/2.'
                )
                raise ValidationError(error_msg)
            result.append(law)
        return result

    @staticmethod
    def _fractional_description(spec: KernelSpec, law: FractionalNoiseLaw) -> str:
        """Return the description of a fractional noise law by the states it drives."""
        states = dict.fromkeys(KernelSpec._state_name(spec, item) for item in components_of(law._get_targets()[0]))
        return f'fractional noise law of {", ".join(states)}'


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


class FractionalProblemPass(BasePass):
    """Pass that adds the noise rows of fractional noise laws and the orders of the states to the problem."""

    reads = ('problem', 'kernels', 'nodes', 'layout')
    writes = ('problem',)

    def _run(self, compilation: Compilation) -> None:
        """Add the fractional noise rows and the state orders to the problem.

        Args:
            compilation:
                The compilation state.
        """
        problem = cast('Problem', compilation.problem)
        lowered = cast('Lowered', compilation.kernels)
        compilation.problem = replace(
            problem,
            fractional=self._fractional_rows(lowered) if problem.form == Form.SDE else (),
            orders=(
                self._state_orders(FractionalFormPass._owned_fractional(compilation), problem.states)
                if problem.form == Form.FRACTIONAL
                else np.zeros(0)
            ),
        )

    @staticmethod
    def _hurst_function(kernel: Kernel, hurst: object) -> Callable[[Vector], Vector]:
        """Return the function of the parameter vector that gives the Hurst index of every row of a noise kernel."""
        rows = kernel.rows
        if SR(hurst).is_numeric():
            value = float(SR(hurst))
            return lambda p: np.full(rows, value)
        reference = kernel.references[SR(hurst)]
        empty = np.zeros(0)
        return lambda p: np.broadcast_to(np.asarray(reference.column_of(empty, p), dtype=float), (rows,)).copy()

    @staticmethod
    def _fractional_rows(lowered: Lowered) -> tuple[FractionalRows, ...]:
        """Return the noise rows that fractional noise laws drive, in the order of the noise rows."""
        rows, position = [], 0
        for kernel in lowered.kernels[NoiseLaw]:
            for item in kernel.outputs:
                spec, law = lowered.laws[id(item)]
                if isinstance(law, FractionalNoiseLaw):
                    positions = position + np.arange(kernel.rows, dtype=np.intp)
                    description = CheckFractionalNoisePass._fractional_description(spec, law)
                    rows.append(
                        FractionalRows(
                            description,
                            positions,
                            FractionalProblemPass._hurst_function(kernel, law.hurst),
                        ),
                    )
                position += kernel.rows
        return tuple(rows)

    @staticmethod
    def _state_orders(states: list[FractionalState], layout: Layout) -> Vector:
        """Return the order of every state component, 1 for the states outside the fractional ones."""
        orders = np.ones(layout.size)
        for item in states:
            orders[layout.entry(item.name).indices] = item.order
        return orders

"""Pass that adds the noise rows of fractional noise laws and the orders of the states to the problem."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import cast

import numpy as np
from sage.all__sagemath_symbolics import SR

from ....core.laws import FractionalNoiseLaw, NoiseLaw
from ....types import Array
from ...problems import Form, Layout, Problem
from ...problems._fractional_rows import FractionalRows
from ...problems._law_kernel import LawKernel
from ...problems._lowered_kernels import LoweredKernels
from .._base import BasePass
from ._check_fractional_noise import CheckFractionalNoisePass
from ._fractional_form import FractionalFormPass
from ._fractional_state import FractionalState


class FractionalProblemPass(BasePass):
    """Pass that adds the noise rows of fractional noise laws and the orders of the states to the problem."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('problem', 'kernels', 'nodes', 'layout')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('problem',)

    def _run(self, inputs: object = None) -> None:
        """Add the fractional noise rows and the state orders to the problem."""
        compilation = self.compilation_
        problem = cast('Problem', compilation.problem)
        lowered = cast('LoweredKernels', compilation.kernels)
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
    def _hurst_function(kernel: LawKernel, hurst: object) -> Callable[[Array], Array]:
        """Return the function of the parameter vector that gives the Hurst index of every row of a noise kernel."""
        rows = kernel.rows
        if SR(hurst).is_numeric():
            value = float(SR(hurst))
            return lambda p: np.full(rows, value)
        reference = kernel.references[SR(hurst)]
        empty = np.zeros(0)
        return lambda p: np.broadcast_to(np.asarray(reference.column_of(empty, p), dtype=float), (rows,)).copy()

    @staticmethod
    def _fractional_rows(lowered: LoweredKernels) -> tuple[FractionalRows, ...]:
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
    def _state_orders(states: list[FractionalState], layout: Layout) -> Array:
        """Return the order of every state component, 1 for the states outside the fractional ones."""
        orders = np.ones(layout.size)
        for item in states:
            orders[layout.entry(item.name).indices] = item.order
        return orders

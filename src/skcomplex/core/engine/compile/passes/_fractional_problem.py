"""Pass that adds the fractional noise rows and the state orders to the problem.

The noise rows are those of noise laws with a Hurst index.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
from sage.all__sagemath_symbolics import SR

from ....components.laws import NoiseLaw
from ....specs import FractionalRowsSpec, FractionalStateSpec, LayoutSpec
from ....types import Array, FormKind
from ...problems._kernel_spec import KernelSpec
from ...problems._law_kernel import LawKernel
from ...problems._lowered_kernels_spec import LoweredKernelsSpec
from .._base import BasePass
from .._compilation import Compilation
from ._check_fractional_noise import CheckFractionalNoisePass
from ._fractional_form import FractionalFormPass


class FractionalProblemPass(BasePass):
    """Pass that adds the fractional noise rows and the state orders to the problem.

    The noise rows are those of noise laws with a Hurst index.
    """

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('problem', 'kernels', 'nodes', 'layout')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('problem',)

    def _run(self, inputs: object = None) -> None:
        """Add the fractional noise rows and the state orders to the problem."""
        compilation = self.compilation_
        problem = Compilation._get_required(compilation.problem, 'problem')
        lowered = Compilation._get_required(compilation.kernels, 'kernels')
        compilation.problem = replace(
            problem,
            fractional=self._get_fractional_rows(lowered) if problem.form == FormKind.SDE else (),
            orders=(
                self._get_state_orders(FractionalFormPass._get_owned_fractional(compilation), problem.states)
                if problem.form == FormKind.FRACTIONAL
                else np.zeros(0)
            ),
        )

    @staticmethod
    def _get_hurst_function(kernel: LawKernel, hurst: object) -> Callable[[Array], Array]:
        """Return the function of the parameter vector that gives the Hurst index.

        It gives the index of every row of a noise kernel.
        """
        rows = kernel.rows
        if SR(hurst).is_numeric():
            value = float(SR(hurst))
            return lambda p: np.full(rows, value)
        reference = kernel.references[SR(hurst)]
        empty = np.zeros(0)
        return lambda p: np.broadcast_to(np.asarray(reference.get_column_of(empty, p), dtype=float), (rows,)).copy()

    @staticmethod
    def _get_fractional_rows(lowered: LoweredKernelsSpec) -> tuple[FractionalRowsSpec, ...]:
        """Return the noise rows that noise laws with a Hurst index drive.

        They are in the order of the noise rows.
        """
        rows, position = [], 0
        for kernel in lowered.kernels[NoiseLaw]:
            for item in kernel.outputs:
                spec, law = lowered.laws[id(item)]
                if isinstance(law, NoiseLaw) and law.hurst is not None:
                    if not isinstance(spec, KernelSpec):
                        error_msg = f'The fractional noise law of output {id(item)} has no kernel spec.'
                        raise TypeError(error_msg)
                    positions = position + np.arange(kernel.rows, dtype=np.intp)
                    description = CheckFractionalNoisePass._get_fractional_description(spec, law)
                    rows.append(
                        FractionalRowsSpec(
                            description,
                            positions,
                            FractionalProblemPass._get_hurst_function(kernel, law.hurst),
                        ),
                    )
                position += kernel.rows
        return tuple(rows)

    @staticmethod
    def _get_state_orders(states: list[FractionalStateSpec], layout: LayoutSpec) -> Array:
        """Return the order of every state component.

        It is 1 for the states outside the fractional ones.
        """
        orders = np.ones(layout.size)
        for item in states:
            orders[layout.entry(item.name).indices] = item.order
        return orders

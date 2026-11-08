"""Define the problem functions spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression

from ...specs._accelerated_spec import AcceleratedSpec
from ...specs._base_spec import BaseSpec
from ...specs._jump_channels_spec import _JumpChannelsSpec
from ...specs._wiener_block_spec import WienerBlockSpec
from ...types import Array, VectorFunction
from ._draw_kernel import DrawKernel


@dataclass(frozen=True)
class _ProblemFunctionsSpec(BaseSpec):
    """Numeric functions, equations, and backends of a problem.

    The passes lower them and the problem takes them.

    Args:
        equations:
            The equation of every state, by state name.

        equation_symbols:
            The symbol of every state the equations read, by state name.

        has_callables:
            Whether a law of the form holds a Python callable.

        has_group_sums:
            Whether a law of the form sums over the members of a group.

        drift:
            The drift function, or `None`.

        sources:
            The part of the drift without the flux differences of finite-volume fields, or `None` without fluxes.

        diffusion:
            The function that returns every Wiener process amplitude, or `None`.

        noise_rows:
            The state index each noise amplitude drives.

        noise_columns:
            The Wiener process each noise amplitude multiplies.

        n_processes:
            The number of Wiener processes.

        noise_blocks:
            The blocks of correlated Wiener processes.

        propensities:
            The propensity function of the jumps, or `None`.

        jumps:
            The channels of the jumps.

        update:
            The update of a map, which takes a generator when the map draws random numbers, or `None`.

        boundary:
            The boundary laws, or `None`.

        algebraic:
            The function that sets the algebraic states of a state vector, or `None`.

        draws:
            The draws of the update laws, or `None`.

        backend:
            The name of the backend that lowered the functions.

        backends:
            The backend that runs the form and every law kind.

        accelerated:
            The compiled loops of the form, or `None`.
    """

    equations: Mapping[str, Expression]
    equation_symbols: Mapping[str, Expression]
    has_callables: bool
    has_group_sums: bool
    drift: VectorFunction | None = None
    sources: VectorFunction | None = None
    diffusion: VectorFunction | None = None
    noise_rows: npt.NDArray[np.intp] = field(default_factory=lambda: np.zeros(0, dtype=np.intp))
    noise_columns: npt.NDArray[np.intp] = field(default_factory=lambda: np.zeros(0, dtype=np.intp))
    n_processes: int = 0
    noise_blocks: tuple[WienerBlockSpec, ...] = ()
    propensities: VectorFunction | None = None
    jumps: tuple[_JumpChannelsSpec, ...] = ()
    update: Callable[..., Array] | None = None
    boundary: VectorFunction | None = None
    algebraic: VectorFunction | None = None
    draws: DrawKernel | None = None
    backend: str = 'numpy'
    backends: Mapping[str, str] = field(default_factory=dict)
    accelerated: AcceleratedSpec | None = None

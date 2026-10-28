"""Numeric functions of a form produced by a backend."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from ...core.laws import BoundaryLaw, JumpLaw, NoiseLaw, RateLaw, UpdateLaw
from ...specs import BaseSpec
from ...types import Indices, VectorFunction
from ..problems._accelerated import Accelerated
from ..problems._jump_channels import _JumpChannels
from ..problems._wiener_block import WienerBlock

_FUNCTION_KINDS: dict[str, type] = {
    'drift': RateLaw,
    'diffusion': NoiseLaw,
    'propensities': JumpLaw,
    'update': UpdateLaw,
    'boundary': BoundaryLaw,
}


@dataclass(frozen=True)
class LoweredFunctions(BaseSpec):
    """Numeric functions of a form produced by a backend.

    Args:
        drift:
            The drift function, or `None`.

        diffusion:
            The function that returns every Wiener process amplitude, or `None`.

        noise_rows:
            The state index each noise amplitude drives.

        noise_columns:
            The Wiener process each noise amplitude draws.

        n_processes:
            The number of Wiener processes.

        noise_blocks:
            The blocks of correlated processes with their covariance factors.

        propensities:
            The function that returns every jump channel rate, or `None`.

        jumps:
            The channels of every jump law.

        update:
            The next-state function of a discrete map, or `None`.

        boundary:
            The function that applies the boundary laws, or `None`.

        accelerated:
            The compiled stepper loops, or `None` when the backend has none.

        backends:
            The backend that runs the kernels of every law kind, by the kind's name, when it differs by kind.
    """

    drift: VectorFunction | None = None
    diffusion: VectorFunction | None = None
    noise_rows: Indices = field(default_factory=lambda: np.zeros(0, dtype=np.intp))
    noise_columns: Indices = field(default_factory=lambda: np.zeros(0, dtype=np.intp))
    n_processes: int = 0
    noise_blocks: tuple[WienerBlock, ...] = ()
    propensities: VectorFunction | None = None
    jumps: tuple[_JumpChannels, ...] = ()
    update: VectorFunction | None = None
    boundary: VectorFunction | None = None
    accelerated: Accelerated | None = None
    backends: Mapping[str, str] = field(default_factory=dict)

"""Define the spec of the numeric functions a backend writes from a problem."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from ...base import BaseSpec
from ...types import Indices, VectorFunction
from ..problems._accelerated_spec import _AcceleratedSpec
from ..problems._jump_channels_spec import _JumpChannelsSpec
from ..problems._wiener_block_spec import _WienerBlockSpec


@dataclass(frozen=True)
class LoweredFunctionsSpec(BaseSpec):
    """A spec of the numeric functions that a backend writes from a problem.

    It holds what a solver calls, each a function of the time, the states, and the parameters: the rates `f` of
    `x' = f(t, x, p)`, the size of the noise, the rates of random jumps, or the next state of a map. A field that the
    problem does not need stays `None`, so `LoweredFunctionsSpec()` holds no function at all. The rates of a falling
    ball, the noise of a stock price, and the jumps of a chemical reaction all reach their solver through it.

    Read more in the user guide, under Backends.

    Args:
        drift:
            The function `f(t, x, p)` of the rates of the states, or `None`.

        diffusion:
            The function that returns the size of every source of noise, or `None`.

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

    See Also:
        BaseBackend: Writes these functions in `functions_`.

    Examples:
        >>> from skcomplex.core.engine.backends import LoweredFunctionsSpec
        >>> LoweredFunctionsSpec().drift is None
        True
    """

    drift: VectorFunction | None = None
    diffusion: VectorFunction | None = None
    noise_rows: Indices = field(default_factory=lambda: np.zeros(0, dtype=np.intp))
    noise_columns: Indices = field(default_factory=lambda: np.zeros(0, dtype=np.intp))
    n_processes: int = 0
    noise_blocks: tuple[_WienerBlockSpec, ...] = ()
    propensities: VectorFunction | None = None
    jumps: tuple[_JumpChannelsSpec, ...] = ()
    update: VectorFunction | None = None
    boundary: VectorFunction | None = None
    accelerated: _AcceleratedSpec | None = None
    backends: Mapping[str, str] = field(default_factory=dict)

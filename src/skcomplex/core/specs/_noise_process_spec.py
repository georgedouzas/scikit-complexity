"""Define the noise process spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ._base_spec import BaseSpec


@dataclass(frozen=True)
class NoiseProcessSpec(BaseSpec):
    """A spec of a named Wiener process that a noise output draws.

    It is shared by every output with the same key.

    Args:
        key:
            The identity of the process: a local label of a component, or the path of a `WienerProcessSpec`.

        index:
            The process within a `WienerProcessSpec`.

        dimension:
            The number of processes of a `WienerProcessSpec`, or 0 for a local label, which takes one process per
            member.

        factor:
            The factor `L` with `L L^T = C` of the covariance of a `WienerProcessSpec`, or `None` for the identity.

        links:
            For the link processes of a conserved noise, the process of every evaluation index among the `dimension`
            processes of the key, or -1 for a link that carries no process, otherwise `None`.
    """

    key: object
    index: int = 0
    dimension: int = 0
    factor: npt.NDArray[np.float64] | None = field(default=None, compare=False)
    links: npt.NDArray[np.intp] | None = field(default=None, compare=False)

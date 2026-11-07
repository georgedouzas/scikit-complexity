"""Adaptation of the conductances of a transport network to the pressure drops of its Kirchhoff flows."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.specs import CallSpec, ParameterSpec, RoleSpec


class FlowAdaptationInteraction(Interaction):
    """Adaptation of the conductances of a transport network to the pressure drops of its Kirchhoff flows.

    The nodes accumulate no fluid, so the pressures `p` solve the mass balance `h_i = sum_j C_ij (p_i - p_j)` with
    the external flows `h`, which sum to zero, at every instant. Every conductance follows `dC_ij/dt = C_ij (alpha1
    |p_i - p_j|^gamma - alpha2)`: it grows with the power dissipated along its edge and decays at a constant rate. The
    balance is solved numerically with the pressure of the first node fixed at zero.

    Args:
        label:
            The label of the interaction.

        vessels:
            The path of the group of vessel segments on the edges.

        edges:
            The ends of every edge, in the order of the group.

        h:
            The external flow of every node, positive for sources and negative for sinks.

        alpha1:
            The growth of the conductance with the pressure drop.

        alpha2:
            The decay rate of the conductance.

        gamma:
            The exponent of the pressure drop.

    Examples:
        >>> from skcomplex.systems.biology.physiology import FlowAdaptationInteraction
        >>> FlowAdaptationInteraction('adaptation', vessels='vessels', edges=((0, 1),), h=(1.0, -1.0)).gamma
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        vessels: Annotated[str, RoleSpec()],
        edges: Sequence[tuple[int, int]] = (),
        h: Sequence[float] = (),
        alpha1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        alpha2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
    ) -> None:
        super().__init__(label)
        self.vessels = vessels
        self.edges = edges
        self.h = h
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.gamma = gamma

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the adaptation of every conductance to the pressure drop of the current flows.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        rate = CallSpec(self._adaptation, (s.vessels.C, s.alpha1, s.alpha2, s.gamma), vectorized=True)
        return [RateLaw(s.vessels.C, rate)]

    def _adaptation(
        self,
        t: float,
        C: npt.NDArray[np.float64],
        alpha1: npt.NDArray[np.float64],
        alpha2: npt.NDArray[np.float64],
        gamma: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return the rates of the conductances at the pressures of the mass balance."""
        drops = self._pressure_drops(np.asarray(C, dtype=float))
        return C * (alpha1 * np.abs(drops) ** gamma - alpha2)

    def _pressure_drops(self, C: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the pressure drop `p_i - p_j` along every edge, solving `h = K p` with the first pressure at zero."""
        ends = np.asarray(self.edges, dtype=int).reshape(-1, 2)
        size = len(self.h)
        laplacian = np.zeros((size, size))
        np.add.at(laplacian, (ends[:, 0], ends[:, 0]), C)
        np.add.at(laplacian, (ends[:, 1], ends[:, 1]), C)
        np.add.at(laplacian, (ends[:, 0], ends[:, 1]), -C)
        np.add.at(laplacian, (ends[:, 1], ends[:, 0]), -C)
        pressures = np.zeros(size)
        pressures[1:] = np.linalg.lstsq(laplacian[1:, 1:], np.asarray(self.h, dtype=float)[1:], rcond=None)[0]
        return pressures[ends[:, 0]] - pressures[ends[:, 1]]

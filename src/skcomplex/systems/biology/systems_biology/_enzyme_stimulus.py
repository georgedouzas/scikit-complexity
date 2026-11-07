"""Activation or deactivation of an enzyme by an outside stimulus or a basal enzyme of fixed concentration."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import step
from ....core.specs import ParameterSpec, RoleSpec


class EnzymeStimulusInteraction(Interaction):
    """Activation or deactivation of an enzyme by an outside stimulus or a basal enzyme of fixed concentration.

    The regulator has the concentration `level`, which rises by `change` at the time `at`, as the input `I` of a
    circuit, and stays fixed for a basal enzyme `F` with no change. It activates the target at the rate `k I (1 - X) /
    (1 - X + K)` or deactivates it at the rate `k I X / (X + K)`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the regulated enzyme.

        sign:
            1 for activation or -1 for deactivation.

        k:
            The catalytic rate.

        K:
            The Michaelis constant.

        level:
            The concentration of the regulator before the change.

        change:
            The step of the concentration at the time `at`.

        at:
            The time of the step.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import EnzymeStimulusInteraction
        >>> EnzymeStimulusInteraction('input', target='A', level=0.5, change=0.1, at=300.0).change
        0.1
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        sign: int = 1,
        k: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        K: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        level: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        change: float = 0.0,
        at: float = 0.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.sign = sign
        self.k = k
        self.K = K
        self.level = level
        self.change = change
        self.at = at

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the conversion of the target by the stimulus.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        regulator = s.level + step(s.t, self.at, self.change)
        return [RateLaw(s.target.X, self._conversion(regulator, s.target.X, s.k, s.K))]

    def _conversion(self, regulator: Expression, X: Expression, k: Expression, K: Expression) -> Expression:
        """Return the Michaelis-Menten conversion of the target by the stimulus of the sign of the interaction."""
        if self.sign > 0:
            return k * regulator * (1 - X) / (1 - X + K)
        return -k * regulator * X / (X + K)

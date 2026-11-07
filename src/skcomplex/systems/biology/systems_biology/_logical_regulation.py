"""Regulation of a logical variable, which steps one level at a time towards the focal level its regulators set."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from itertools import product
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import SR, Expression

from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, JumpLaw
from ....core.components.laws.utils import where
from ....core.specs import RoleSpec


class LogicalRegulationInteraction(Interaction):
    """Regulation of a logical variable, which steps one level at a time towards the focal level its regulators set.

    In the formalism of Thomas, every regulator acts when its level reaches its threshold. The logical parameters map
    every combination of acting regulators, a tuple of truth values in the order of the regulators, to the focal level
    of the target. The target rises or falls by one level towards its focal level at the unit rate, so the transitions
    of all variables are asynchronous random events.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the regulated variable.

        regulators:
            The paths of the regulating variables.

        thresholds:
            The level at which every regulator acts, in the order of the regulators.

        parameters:
            The focal level of the target for every combination of acting regulators.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import LogicalRegulationInteraction
        >>> regulation = LogicalRegulationInteraction(
        ...     'x_regulation', target='x', regulators=['y'], thresholds=(1,), parameters={(False,): 2, (True,): 0}
        ... )
        >>> regulation.parameters[(True,)]
        0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        regulators: Annotated[Sequence[str], RoleSpec(variadic=True)] = (),
        thresholds: Sequence[int] = (),
        parameters: Mapping[tuple[bool, ...], int] | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.regulators = regulators
        self.thresholds = thresholds
        self.parameters = parameters

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the steps of the target up and down towards its focal level.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        level = s.target.level
        focal = self._focal(s.regulators)
        return [
            JumpLaw(where(focal > level, 1, 0), {level: 1}),
            JumpLaw(where(focal < level, 1, 0), {level: -1}),
        ]

    def _focal(self, regulators: Sequence[SimpleNamespace]) -> Expression:
        """Return the focal level of the target, the logical parameter of the regulators that act."""
        parameters = self.parameters or {}
        focal: Expression = SR(0)
        for acting in product((False, True), repeat=len(regulators)):
            indicator: Expression = SR(1)
            for regulator, threshold, active in zip(regulators, self.thresholds, acting, strict=True):
                indicator *= (
                    where(regulator.level >= threshold, 1, 0) if active else where(regulator.level < threshold, 1, 0)
                )
            focal += parameters.get(acting, 0) * indicator
        return focal

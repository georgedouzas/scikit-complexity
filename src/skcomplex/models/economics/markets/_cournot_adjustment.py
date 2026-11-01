"""Simultaneous best responses of two firms to their beliefs about the rival's last output."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.interactions import MapInteraction
from ....engine.solvers import ModelView
from ....specs import ParameterSpec, RoleSpec
from ._duopolist_view import DuopolistView


class CournotAdjustmentInteraction(MapInteraction):
    """Simultaneous best responses of two firms to their beliefs about the rival's last output.

    Under the inverse demand `P = a - b Q`, each firm sets `q_i = (a - c_i) / (2 b) - belief_j / 2`, where the
    belief is the rival's last output scaled by a factor drawn uniformly from `[1 - error, 1 + error]`. It is a
    synchronous map of the agent form of the outputs, so both firms respond to the outputs of the step before.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of the two firms.

        a:
            The intercept of the inverse demand.

        b:
            The slope of the inverse demand.

        error:
            The largest relative error of a firm's belief about the rival's output.

    Examples:
        >>> from skcomplex.models.economics.markets import CournotAdjustmentInteraction
        >>> CournotAdjustmentInteraction('market', target='firms', error=0.1).check().error
        0.1
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        a: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1500.0,
        b: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        error: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, target=target, state='output', function=self._get_response, form='agent')
        self.a = a
        self.b = b
        self.error = error

    def _get_response(self, agent: DuopolistView, model: ModelView) -> float:
        """Return the best response of a firm to its belief about the last output of its rival."""
        a, b, error = (model.parameter(name) for name in ('a', 'b', 'error'))
        rival = next(firm for firm in model.agents('target', DuopolistView) if firm.id != agent.id)
        belief = rival.output * (1 + error * float(model.rng.uniform(-1.0, 1.0)))
        return max((a - agent.cost) / (2 * b) - belief / 2, 0.0)

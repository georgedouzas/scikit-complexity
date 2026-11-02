"""The best responses of a Cournot duopoly with cross holdings."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from ....core.base import BasePart
from ....core.elements import Element
from ....core.laws import EquationLaw
from ....specs import ParameterSpec, StateSpec


class CournotBestResponseElement(Element):
    """Quantities of two firms at the best responses of a Cournot game, where each firm holds a share of the other.

    The conditions are `alpha - 2 q_i - q_j = c_i / (1 + lambda_i)` and its mirror (Forrest 2018 Eq. 21.5), whose
    solution is `q_i = (alpha - 2 c_i / (1 + lambda_i) + c_j / (1 + lambda_j)) / 3` (Eq. 21.6).

    Args:
        label:
            The label of the element.

        q_i:
            The initial guess of the quantity of the first firm, which the equations replace.

        q_j:
            The initial guess of the quantity of the second firm, which the equations replace.

        demand:
            The intercept `alpha` of the inverse demand.

        c_i:
            The unit cost of the first firm.

        c_j:
            The unit cost of the second firm.

        lambda_i:
            The cross holding of the first firm.

        lambda_j:
            The cross holding of the second firm.

    Examples:
        >>> from skcomplex.models.economics.markets import CournotBestResponseElement
        >>> CournotBestResponseElement('duopoly', demand=8.0).demand
        8.0
    """

    def __init__(
        self,
        label: str,
        *,
        q_i: Annotated[float, StateSpec()] = 0.0,
        q_j: Annotated[float, StateSpec()] = 0.0,
        demand: Annotated[float, ParameterSpec()] = 10.0,
        c_i: Annotated[float, ParameterSpec()] = 1.0,
        c_j: Annotated[float, ParameterSpec()] = 2.0,
        lambda_i: Annotated[float, ParameterSpec()] = 0.0,
        lambda_j: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label)
        self.q_i = q_i
        self.q_j = q_j
        self.demand = demand
        self.c_i = c_i
        self.c_j = c_j
        self.lambda_i = lambda_i
        self.lambda_j = lambda_j

    def _get_laws(self, s: SimpleNamespace) -> Sequence[BasePart]:
        """Return the two conditions of the best responses.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The equations of the two quantities.
        """
        return [
            EquationLaw(s.demand - 2 * s.q_i - s.q_j == s.c_i / (1 + s.lambda_i), target=s.q_i),
            EquationLaw(s.demand - 2 * s.q_j - s.q_i == s.c_j / (1 + s.lambda_j), target=s.q_j),
        ]

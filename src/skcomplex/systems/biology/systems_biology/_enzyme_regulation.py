"""Michaelis-Menten activation or deactivation of an enzyme by another enzyme of a circuit."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.errors import ValidationError
from ....core.specs import ParameterSpec, RoleSpec


class EnzymeRegulationInteraction(Interaction):
    """Michaelis-Menten activation or deactivation of an enzyme by another enzyme of a circuit.

    An activating source converts the inactive fraction of the target at the rate `k X_s (1 - X) / (1 - X + K)`, and a
    deactivating source the active fraction at the rate `k X_s X / (X + K)`, the kinetics of Ma et al. (2009).

    Args:
        label:
            The label of the interaction.

        target:
            The path of the regulated enzyme.

        source:
            The path of the regulating enzyme.

        sign:
            1 for activation or -1 for deactivation.

        k:
            The catalytic rate.

        K:
            The Michaelis constant.

    Raises:
        ValidationError:
            The sign is neither 1 nor -1.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import EnzymeRegulationInteraction
        >>> EnzymeRegulationInteraction('ab', target='B', source='A', sign=-1).sign
        -1
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        source: Annotated[str, RoleSpec()],
        sign: int = 1,
        k: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        K: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.source = source
        self.sign = sign
        self.k = k
        self.K = K

    def _check(self) -> None:
        """Check the sign.

        Raises:
            ValidationError:
                The sign is neither 1 nor -1.
        """
        super()._check()
        if self.sign not in {1, -1}:
            error_msg = f'The sign of a regulation must be 1 or -1, not {self.sign}.'
            raise ValidationError(error_msg)

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the conversion of the target by the source.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.target.X, self._conversion(s.source.X, s.target.X, s.k, s.K))]

    def _conversion(self, regulator: Expression, X: Expression, k: Expression, K: Expression) -> Expression:
        """Return the Michaelis-Menten conversion of the target by a regulator of the sign of the interaction."""
        if self.sign > 0:
            return k * regulator * (1 - X) / (1 - X + K)
        return -k * regulator * X / (X + K)

"""Rates of a biochemical system in the S-system form of biochemical systems theory."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.specs import ParameterSpec, RoleSpec
from ._generalized_mass_action import GeneralizedMassActionInteraction


class SSystemInteraction(GeneralizedMassActionInteraction):
    """Rates of a biochemical system in the S-system form of biochemical systems theory.

    Every species has one aggregated production and one aggregated degradation, both power laws:
    `dx_i/dt = alpha_i prod_j x_j^g_ij - beta_i prod_j x_j^h_ij`. It is the generalized mass action form with two
    fluxes per species.

    Args:
        label:
            The label of the interaction.

        species:
            The paths of the species.

        alpha:
            The rate constant of the production of every species.

        beta:
            The rate constant of the degradation of every species.

        g:
            The kinetic orders of the productions, one row per species.

        h:
            The kinetic orders of the degradations, one row per species.

    Raises:
        ValidationError:
            The kinetic orders do not have one row and one column per species.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import SSystemInteraction
        >>> SSystemInteraction('rates', species=['X'], alpha=[1.0], beta=[1.0], g=((0.0,),), h=((1.0,),)).h
        ((1.0,),)
    """

    def __init__(
        self,
        label: str,
        *,
        species: Annotated[list[str], RoleSpec(variadic=True)],
        alpha: Annotated[Sequence[float], ParameterSpec(per='species', domain=RealSet.closed_open(0, oo))] = (),
        beta: Annotated[Sequence[float], ParameterSpec(per='species', domain=RealSet.closed_open(0, oo))] = (),
        g: Sequence[Sequence[float]] = ((0.0,),),
        h: Sequence[Sequence[float]] = ((0.0,),),
    ) -> None:
        super().__init__(label, species=species)
        self.alpha = alpha
        self.beta = beta
        self.g = g
        self.h = h

    def _check_matrices(self) -> None:
        """Raise an error for kinetic orders without one row and one column per species."""
        self._check_shape('g', len(self.species), len(self.species))
        self._check_shape('h', len(self.species), len(self.species))

    def _get_fluxes(self, s: SimpleNamespace, values: Sequence[Expression]) -> list[Expression]:
        """Return the production and the degradation flux of every species, in turn."""
        return [
            flux
            for alpha, beta, g, h in zip(s.alpha, s.beta, self.g, self.h, strict=True)
            for flux in (alpha * self._get_power_product(values, g), beta * self._get_power_product(values, h))
        ]

    def _get_stoichiometry(self) -> Sequence[Sequence[float]]:
        """Return the coefficient 1 of every species in its production and -1 in its degradation."""
        size = len(self.species)
        return [
            [1.0 if flux == 2 * row else -1.0 if flux == 2 * row + 1 else 0.0 for flux in range(2 * size)]
            for row in range(size)
        ]

"""Rates of a biochemical system in the generalized mass action form of biochemical systems theory."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from math import prod
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import SR, Expression

from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.errors import ValidationError
from ....core.specs import RoleSpec


class GeneralizedMassActionInteraction(Interaction):
    """Rates of a biochemical system in the generalized mass action form of biochemical systems theory.

    Flux `j` is the power law `v_j = gamma_j prod_k x_k^f_jk`, and species `i` changes as `dx_i/dt = sum_j s_ij v_j`.
    The kinetic orders `f_jk` are real numbers, negative for inhibition.

    Args:
        label:
            The label of the interaction.

        species:
            The paths of the species, in the order of the rows of the stoichiometry.

        stoichiometry:
            The stoichiometric coefficient `s_ij` of every species in every flux, one row per species.

        gamma:
            The rate constant of every flux.

        orders:
            The kinetic order `f_jk` of every species in every flux, one row per flux.

    Raises:
        ValidationError:
            The matrices do not have one row per flux or species and one entry per species or flux.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import GeneralizedMassActionInteraction
        >>> v = GeneralizedMassActionInteraction(
        ...     'v', species=['X'], stoichiometry=((1, -1),), gamma=(1, 2), orders=((0,), (1,))
        ... )
        >>> v.gamma
        (1, 2)
    """

    def __init__(
        self,
        label: str,
        *,
        species: Annotated[list[str], RoleSpec(variadic=True)],
        stoichiometry: Sequence[Sequence[float]] = ((0.0,),),
        gamma: Sequence[float] = (0.0,),
        orders: Sequence[Sequence[float]] = ((0.0,),),
    ) -> None:
        super().__init__(label)
        self.species = species
        self.stoichiometry = stoichiometry
        self.gamma = gamma
        self.orders = orders

    def _get_power_product(self, values: Sequence[Expression], orders: Sequence[float]) -> Expression:
        """Return the product `prod_k x_k^f_k` of the values raised to their kinetic orders, leaving out zero orders."""
        return prod((value**order for value, order in zip(values, orders, strict=True) if order), start=SR(1))

    def _check_shape(self, name: str, rows: int, columns: int) -> None:
        """Raise an error unless a matrix argument has the given numbers of rows and columns."""
        matrix = getattr(self, name)
        if len(matrix) == rows and all(len(row) == columns for row in matrix):
            return
        error_msg = (
            f'The matrix {name} of {self.label!r} must have {rows} rows of {columns} entries, one row per flux or '
            'species.'
        )
        raise ValidationError(error_msg)

    def _check_matrices(self) -> None:
        """Raise an error for a stoichiometry or kinetic orders that do not match the species and the fluxes."""
        self._check_shape('stoichiometry', len(self.species), len(self.gamma))
        self._check_shape('orders', len(self.gamma), len(self.species))

    def _get_fluxes(self, s: SimpleNamespace, values: Sequence[Expression]) -> list[Expression]:
        """Return the power-law fluxes `gamma_j prod_k x_k^f_jk`."""
        return [rate * self._get_power_product(values, row) for rate, row in zip(self.gamma, self.orders, strict=True)]

    def _get_stoichiometry(self) -> Sequence[Sequence[float]]:
        """Return the stoichiometric coefficient of every species in every flux, one row per species."""
        return self.stoichiometry

    def _check(self) -> None:
        """Raise an error for matrices that do not match the species and the fluxes.

        Raises:
            ValidationError:
                The matrices do not have one row per flux or species and one entry per species or flux.
        """
        super()._check()
        self._check_matrices()

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the sum of the power-law fluxes of every species.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        values = [item.x for item in s.species]
        fluxes = self._get_fluxes(s, values)
        return [
            RateLaw(value, sum((c * flux for c, flux in zip(row, fluxes, strict=True) if c), start=SR(0)))
            for value, row in zip(values, self._get_stoichiometry(), strict=True)
        ]

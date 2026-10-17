"""Newtonian mechanics systemss.

This module implements classes which can be used to define and solve the
dynamical equations of Netwonian systems.
"""

# Author: Georgios Douzas <gdouzas@icloud.com> License: MIT

from collections.abc import Iterable
from typing import Any, Self, cast

import numpy as np
import numpy.typing as npt
from sage.all import Expression, assume, desolve, diff, function, symbolic_expression, var, vector, zero_vector
from sklearn.utils import check_scalar

from ...base.component import BaseElement, BaseElementsInteraction, BaseEnvironmentInteraction
from ...base.system import BaseSystem, BaseSystemState
from ...component import EuclideanSpace


def init_force(F: list[Expression | float | int]) -> list[Expression]:
    """Check the provided force vector."""
    try:
        error_msg = f'Parameter `F` should be a list of numeric elements or symbolic expressions. Got `{F}` instead.'
        for F_component in F:
            assert isinstance(F_component, Expression | float | int), error_msg
    except TypeError as error:
        raise TypeError(error_msg) from error
    F_norm = [
        symbolic_expression(
            float(F_func_component) if isinstance(F_func_component, float | int) else F_func_component,
        )
        for F_func_component in F
    ]
    return F_norm


def init_potential(V: Expression | float | int) -> symbolic_expression:
    """Check the provided potential."""
    error_msg = f'Parameter `V` should be a numeric element or symbolic expression. Got `{V}` instead.'
    if not isinstance(V, Expression | float | int):
        raise TypeError(error_msg)
    return symbolic_expression(V)


class PointParticle(BaseElement):

    def __init__(self: Self, label: str, m: int | float | None = None) -> None:
        self.m = m
        super().__init__(label)

    def _init_param(self: Self, param_name: str) -> Self:
        """Check the particle's mass."""
        if param_name == 'm':
            self.m_ = float(check_scalar(self.m, param_name, (int, float), min_val=0.0, include_boundaries='neither'))
        return self


class InternalForce(BaseElementsInteraction):

    def __init__(
        self: Self,
        label: str,
        element_1_label: str,
        element_2_label: str,
        F: list[Expression] | None = None,
    ) -> None:
        self.F = F
        super().__init__(label, element_1_label, element_2_label)

    def _init_param(self: Self, param_name: str) -> Self:
        """Check the particle's mass."""
        if param_name == 'F':
            self.F_ = init_force(cast(list[Expression | float | int], self.F))
        return self


class InternalPotential(BaseElementsInteraction):

    def __init__(
        self: Self,
        label: str,
        element_1_label: str,
        element_2_label: str,
        V: Expression | None = None,
    ) -> None:
        self.V = V
        super().__init__(label, element_1_label, element_2_label)

    def _init_param(self: Self, param_name: str) -> Self:
        """Check the particle's mass."""
        if param_name == 'V':
            self.V_ = init_potential(self.V)
        return self


class ExternalForce(BaseEnvironmentInteraction):

    def __init__(self: Self, label: str, element_label: str, F: list[Expression] | None = None) -> None:
        self.F = F
        super().__init__(label, element_label)

    def _init_param(self: Self, param_name: str) -> Self:
        """Check the particle's mass."""
        if param_name == 'F':
            self.F_ = init_force(cast(list[Expression | float | int], self.F))
        return self


class ExternalPotential(BaseEnvironmentInteraction):

    def __init__(self: Self, label: str, element_label: str, V: Expression | None = None) -> None:
        self.V = V
        super().__init__(label, element_label)

    def _init_param(self: Self, param_name: str) -> Self:
        """Check the particle's mass."""
        if param_name == 'V':
            self.V_ = init_potential(self.V)
        return self


class PhaseSpacePoint(BaseSystemState):
    """Point of the phase space."""


class ClassicalMechanicsSystem(BaseSystem):
    """Classical mechanics system of particles.

    Simulate a classicail mechanics system of particles.

    Args:
        particles:
            The particles of the system.

        internal_interactions:
            The internal interactions between the particles.

        external_interactions:
            The external interactions between the particles and the environment.

        n_dim:
            Number of spatial dimensions.
    """

    def __init__(
        self: Self,
        particles: Iterable[PointParticle],
        space: EuclideanSpace,
        internal_interactions: Iterable[InternalForce | InternalPotential] | None = None,
        external_interactions: Iterable[ExternalForce | ExternalPotential] | None = None,
    ) -> None:
        super().__init__(
            elements=particles,
            elements_interactions=internal_interactions,
            environment_interactions=external_interactions,
            space=space,
        )

    def _set_dynamic_equations(self: Self) -> Self:
        t = var('t')
        for element in self.elements_:
            label = element.label
            F = zero_vector(self.space_.n_dim_)
            for interaction in self._get_element_interactions(element):
                F += vector(interaction.F_)
            coords = [f'{coord}__{label}' for coord in self.space_.coordinates_[:]]
            velocities = [f'v_{coord}__{label}' for coord in self.space_.coordinates_[:]]
            self.dynamic_equations_['formulas'][label] = []
            for ind, F_component in enumerate(F):
                assume(element.m_ > 0)
                self.dynamic_equations_['formulas'][label].append(
                    element.m_ * diff(function(coords[ind])(t), t, 2)
                    == symbolic_expression(F_component)(
                        **{symbol: function(symbol)(t) for symbol in (coords + velocities)},
                    ),
                )
        return self

    def _solve_dynamic_equations(self: Self, I: dict[str, Any] | None = None) -> Self:
        t = var('t')
        for label, des in self.dynamic_equations_['formulas'].items():
            coords = [f'{coord}__{label}' for coord in self.space_.coordinates_[:]]
            self.dynamic_equations_['solutions'][label] = []
            for ind, de in enumerate(des):
                self.dynamic_equations_['solutions'][label].append(
                    desolve(de, dvar=function(coords[ind])(t), ivar=t, ics=I[label][ind] if I is not None else None),
                )
        return self

    def simulate(self: Self, method: str = 'dynamic_equations', I: dict[str, Any] | None = None) -> Self:
        """Simulate the system."""
        super().simulate()

        # Dynamic equations
        if method == 'dynamic_equations':
            self.dynamic_equations_dict[str, dict[str, Any]] = {'formulas': {}, 'solutions': {}}
            self._set_dynamic_equations()._solve_dynamic_equations(I)

        return self

    def predict(self: Self, time_points: npt.NDArray[np.float32]) -> list[PhaseSpacePoint]:
        """Predict the system states.

        Args:
            time_points:
                The time points to predict the state of the system.

        Returns:
            system_states:
                The predicted states of the system.
        """
        return []

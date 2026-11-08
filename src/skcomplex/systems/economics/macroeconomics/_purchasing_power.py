"""Macroeconomics systems: the categorized purchasing power of money.

It runs under sector policies, decoupled by feedback.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from functools import partial
from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import Expression

from ....core.components.elements import DeclaredElement
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import step
from ....core.components.system import BaseSystem, RunSettings, System
from ....core.specs import DeclaredVariableSpec, ObservableSpec, ParameterSpec, StateSpec, VisualSpec

SECTORS = 3


A_MATRIX = ((-1.0, 0.3, 0.2), (0.2, -0.8, 0.3), (0.1, 0.4, -0.6))


B_MATRIX = ((1.0, 0.2, 0.1), (0.1, 1.0, 0.2), (0.2, 0.1, 1.0))


C_MATRIX = ((1.0, 0.3, 0.0), (0.0, 1.0, 0.3), (0.2, 0.0, 1.0))


D_MATRIX = ((1.0, 0.2, 0.0), (0.0, 1.0, 0.2), (0.1, 0.0, 1.0))


class PurchasingPowerElement(DeclaredElement):
    """National economy of three spending sectors, with sector policies.

    It follows `z' = A z + B u` and `y = C z + D u`. The state `z_i` is the gap between the demand and the supply of
    money of sector `i`, the basic, the desired, and the luxurious living conditions, the output `y_i` is the index of
    the sector, and the policy `x_i` of a sector is a step of size `policy_i` at the time `policy_time_i`. Without
    feedback the economy takes the policies as its inputs, `u = x`. With feedback, the market reactions of Forrest's
    feedback component set `u` so that every sector responds to its own policy only, as the decoupled systems
    `w_i' = A w_i + B_i x_i` and `y_i = C_i w_i + D_i x_i`, whose states `w_ik` the element holds. The feedback
    component holds a copy of `z` that equals `z` from the common zero start, so `u = D^-1 (y - C z)` with the decoupled
    outputs `y`.

    Args:
        label:
            The label of the economy.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        variables:
            Declared states and parameters that replace the derived ones, by name.

        A:
            The matrix of the state.

        B:
            The matrix of the inputs in the state.

        C:
            The matrix of the state in the outputs.

        D:
            The nonsingular matrix of the inputs in the outputs.

        feedback:
            Whether the feedback component decouples the sectors.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import PurchasingPowerElement
        >>> PurchasingPowerElement('economy', feedback=True).get_value('z1')
        0.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        variables: Mapping[str, DeclaredVariableSpec] | None = None,
        A: Sequence[Sequence[float]] = A_MATRIX,
        B: Sequence[Sequence[float]] = B_MATRIX,
        C: Sequence[Sequence[float]] = C_MATRIX,
        D: Sequence[Sequence[float]] = D_MATRIX,
        feedback: bool = False,
    ) -> None:
        super().__init__(label, visual=visual, variables=variables)
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.feedback = feedback

    def _declare_variables(self) -> Mapping[str, DeclaredVariableSpec]:
        """Return the states `z_i`, with feedback `w_ik`, starting at zero.

        It also returns the sizes and times of the policies.
        """
        states = [f'z{i}' for i in range(1, SECTORS + 1)]
        if self.feedback:
            states += [f'w{i}{k}' for i in range(1, SECTORS + 1) for k in range(1, SECTORS + 1)]
        declared = {name: DeclaredVariableSpec(0.0, StateSpec()) for name in states}
        for i in range(1, SECTORS + 1):
            declared[f'policy{i}'] = DeclaredVariableSpec(0.0, ParameterSpec())
            declared[f'policy_time{i}'] = DeclaredVariableSpec(1.0, ParameterSpec())
        return {**declared, **(self.variables or {})}

    @staticmethod
    def _policies(s: SimpleNamespace) -> list[Expression]:
        """Return the policies of the sectors, steps of their sizes at their times."""
        return [step(s.t, getattr(s, f'policy_time{i}'), getattr(s, f'policy{i}')) for i in range(1, SECTORS + 1)]

    def get_outputs(self, s: SimpleNamespace) -> list[Expression]:
        """Return the indices `y` of the sectors.

        Args:
            s:
                The namespace of the economy.

        Returns:
            outputs:
                One index per sector.
        """
        x, C, D = self._policies(s), np.asarray(self.C), np.asarray(self.D)
        if not self.feedback:
            z = [getattr(s, f'z{i}') for i in range(1, SECTORS + 1)]
            return [sum(C[i, k] * z[k] + D[i, k] * x[k] for k in range(SECTORS)) for i in range(SECTORS)]
        return [
            sum(C[i, k] * getattr(s, f'w{i + 1}{k + 1}') for k in range(SECTORS)) + D[i, i] * x[i]
            for i in range(SECTORS)
        ]

    def get_inputs(self, s: SimpleNamespace) -> list[Expression]:
        """Return the inputs `u` of the economy.

        They are the policies, or `D^-1 (y - C z)` with feedback.

        Args:
            s:
                The namespace of the economy.

        Returns:
            inputs:
                One input per sector.
        """
        if not self.feedback:
            return self._policies(s)
        C, inverse = np.asarray(self.C), np.linalg.inv(np.asarray(self.D, dtype=float))
        z, y = [getattr(s, f'z{i}') for i in range(1, SECTORS + 1)], self.get_outputs(s)
        gaps = [y[i] - sum(C[i, k] * z[k] for k in range(SECTORS)) for i in range(SECTORS)]
        return [sum(inverse[i, k] * gaps[k] for k in range(SECTORS)) for i in range(SECTORS)]

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the economy and, with feedback, of the decoupled systems.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        A, B = np.asarray(self.A), np.asarray(self.B)
        z, u = [getattr(s, f'z{i}') for i in range(1, SECTORS + 1)], self.get_inputs(s)
        laws: list[BaseLaw] = [
            RateLaw(z[i], sum(A[i, k] * z[k] + B[i, k] * u[k] for k in range(SECTORS))) for i in range(SECTORS)
        ]
        if self.feedback:
            x = self._policies(s)
            for i in range(SECTORS):
                w = [getattr(s, f'w{i + 1}{k + 1}') for k in range(SECTORS)]
                laws += [
                    RateLaw(w[j], sum(A[j, k] * w[k] for k in range(SECTORS)) + B[j, i] * x[i]) for j in range(SECTORS)
                ]
        return laws


def _sector_value(s: SimpleNamespace, economy: PurchasingPowerElement, method: str, index: int) -> Expression:
    """Return the index or the input of one sector of the economy."""
    values: list[Expression] = getattr(economy, method)(s)
    return values[index]


def _observables(system: System) -> list[ObservableSpec]:
    """Return the index and the input of every sector."""
    economy = system.elements[0]
    if not isinstance(economy, PurchasingPowerElement):
        return []
    return [
        ObservableSpec(
            f'{name}{i + 1}',
            partial(_sector_value, economy=economy, method=method, index=i),
            element='economy',
        )
        for name, method in (('y', 'get_outputs'), ('u', 'get_inputs'))
        for i in range(SECTORS)
    ]


class CategorizedPurchasingPowerSystem(BaseSystem):
    """Forrest's categorized purchasing power of money.

    Its three sectors can be steered one by one by feedback. The economy is split into the sectors of the basic, the
    desired, and the luxurious living conditions, with the gaps `z_i = D_i - S_i` between the demand and the supply of
    money of each, which the categorized monetary and fiscal policies `x_i` move through `z' = A z + B x`, and the
    indices of the sectors `y = C z + D x`, from the zero state (Forrest 2018, Eqs. 22.6 and 22.7). The policies of one
    sector act on all of them. By Theorem 11.3 and Example 11.2 the system can be decoupled by feedback into the systems
    `w' = A w + B_i x_i` and `y_i = C_i w + D_i x_i` of the sectors (Eqs. 22.9 to 22.12), whose market reactions keep
    each index on its own policy, so money poured into the luxury sector leaves the basic sector alone. The source gives
    no numbers, so the matrices are illustrative, with a stable `A` and a stable `A - B D^-1 C`, and the policies are
    steps.

    The system has the gaps `economy.z1` to `economy.z3` and, with feedback, the decoupled states `economy.w11` to
    `economy.w33`. The observables of the system are the indices `economy.y1` to `economy.y3` and the inputs of the
    economy `economy.u1` to `economy.u3`.

    Args:
        policy:
            The size of the step policy of every sector.

        policy_time:
            The time of the step policy of every sector.

        feedback:
            Whether the feedback component decouples the sectors.

        A:
            The matrix of the state.

        B:
            The matrix of the policies in the state.

        C:
            The matrix of the state in the indices.

        D:
            The nonsingular matrix of the policies in the indices.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import CategorizedPurchasingPowerSystem
        >>> CategorizedPurchasingPowerSystem(feedback=False).prepare().system_.elements[0].feedback
        False
    """

    field = 'economics/macroeconomics'
    topics = ('purchasing power', 'monetary policy', 'fiscal policy', 'decoupling', 'feedback', 'linear system', 'ode')
    references = ('Forrest 2018',)
    until = 15.0
    dt = 0.05

    def __init__(
        self,
        policy: Sequence[float] = (0.0, 0.0, 1.0),
        policy_time: Sequence[float] = (1.0, 1.0, 1.0),
        *,
        feedback: bool = True,
        A: Sequence[Sequence[float]] = A_MATRIX,
        B: Sequence[Sequence[float]] = B_MATRIX,
        C: Sequence[Sequence[float]] = C_MATRIX,
        D: Sequence[Sequence[float]] = D_MATRIX,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.policy = policy
        self.policy_time = policy_time
        self.feedback = feedback
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        variables = {
            name: DeclaredVariableSpec(float(value), ParameterSpec())
            for i in range(SECTORS)
            for name, value in ((f'policy{i + 1}', self.policy[i]), (f'policy_time{i + 1}', self.policy_time[i]))
        }
        economy = PurchasingPowerElement(
            'economy',
            variables=variables,
            A=self.A,
            B=self.B,
            C=self.C,
            D=self.D,
            feedback=self.feedback,
        )
        return System(self.label, elements=[economy])

    def get_run_settings(self) -> RunSettings:
        """Return the tuned run settings.

        The observables of the parts are built from the model parameters.
        """
        settings = super().get_run_settings()
        settings['observables'] = _observables(self._pass_parameters())
        return settings

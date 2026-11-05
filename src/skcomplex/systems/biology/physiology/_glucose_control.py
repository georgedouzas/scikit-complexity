"""Blood glucose held near its norm by insulin, which lowers it, and glucagon, which raises it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class GlucoseControlElement(Element):
    """Blood glucose held near its norm by insulin, which lowers it, and glucagon, which raises it.

    The glucose follows `dG/dt = R - (k0 + k_i I) G + k_n N + u(t)`, where `u` is a meal of height `meal` from
    `meal_time` and an exercise of height `-exercise` from `exercise_time`, each lasting `duration`. High glucose
    secretes insulin, `dI/dt = s_i h(G) - d_i I`, and low glucose secretes glucagon, `dN/dt = s_n (1 - h(G)) - d_n N`,
    with the Hill switch `h(G) = G^n / (G^n + K^n)`. The production `R` is set so that the norm `G = K` is a steady
    state.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `glucose_dual_hormone_control` read its parameters.

    Args:
        label:
            The label of the organism.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        G:
            The initial glucose, relative to the norm.

        I:
            The initial insulin.

        N:
            The initial glucagon.

        R:
            The basal production of glucose.

        k0:
            The uptake of glucose without insulin.

        k_i:
            The uptake of glucose per unit of insulin.

        k_n:
            The release of glucose per unit of glucagon, 0 without the glucagon loop.

        s_i:
            The maximal secretion of insulin.

        s_n:
            The maximal secretion of glucagon.

        d_i:
            The clearance rate of insulin.

        d_n:
            The clearance rate of glucagon.

        K:
            The norm of the glucose, where both hormones are half secreted.

        n:
            The steepness of the secretion switch.

        meal:
            The rate of glucose intake during the meal.

        exercise:
            The rate of extra glucose uptake during the exercise.

        meal_time:
            The start of the meal.

        exercise_time:
            The start of the exercise.

        duration:
            The duration of the meal and of the exercise.

    Examples:
        >>> from skcomplex.systems.biology.physiology import GlucoseControlElement
        >>> GlucoseControlElement('organism').meal_time
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        G: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        I: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        N: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        R: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.75,
        k0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        k_i: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        k_n: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        s_i: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        s_n: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        d_i: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        d_n: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        K: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        n: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 4.0,
        meal: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        exercise: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.8,
        meal_time: float = 2.0,
        exercise_time: float = 12.0,
        duration: float = 2.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.G = G
        self.I = I
        self.N = N
        self.R = R
        self.k0 = k0
        self.k_i = k_i
        self.k_n = k_n
        self.s_i = s_i
        self.s_n = s_n
        self.d_i = d_i
        self.d_n = d_n
        self.K = K
        self.n = n
        self.meal = meal
        self.exercise = exercise
        self.meal_time = meal_time
        self.exercise_time = exercise_time
        self.duration = duration

"""Regulation on the organism level: antagonistic hormone loops and an extracellular cascade with positive feedback."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import Domain, parameter, state
from ....core.components import Element
from ....core.laws import Law, Rate, Symbols, pulse
from ....core.runtime import register_model
from ....core.system import System


class GlucoseControl(Element):
    """Blood glucose held near its norm by insulin, which lowers it, and glucagon, which raises it.

    The glucose follows `dG/dt = R - (k0 + k_i I) G + k_n N + u(t)`, where `u` is a meal of height `meal` from
    `meal_time` and an exercise of height `-exercise` from `exercise_time`, each lasting `duration`. High glucose
    secretes insulin, `dI/dt = s_i h(G) - d_i I`, and low glucose secretes glucagon, `dN/dt = s_n (1 - h(G)) - d_n N`,
    with the Hill switch `h(G) = G^n / (G^n + K^n)`. The production `R` is set so that the norm `G = K` is a steady
    state.

    Args:
        label:
            The label of the organism.

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
        >>> from skcomplex.models.biology.physiology import GlucoseControl
        >>> GlucoseControl('organism').meal_time
        2.0
    """

    G: float = state(default=1.0, domain=Domain.NONNEGATIVE)
    I: float = state(default=1.0, domain=Domain.NONNEGATIVE)
    N: float = state(default=0.5, domain=Domain.NONNEGATIVE)
    R: float = parameter(default=0.75, domain=Domain.NONNEGATIVE)
    k0: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)
    k_i: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)
    k_n: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    s_i: float = parameter(default=2.0, domain=Domain.NONNEGATIVE)
    s_n: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    d_i: float = parameter(default=1.0, domain=Domain.POSITIVE)
    d_n: float = parameter(default=1.0, domain=Domain.POSITIVE)
    K: float = parameter(default=1.0, domain=Domain.POSITIVE)
    n: float = parameter(default=4.0, domain=Domain.POSITIVE)
    meal: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    exercise: float = parameter(default=0.8, domain=Domain.NONNEGATIVE)
    meal_time: float = 2.0
    exercise_time: float = 12.0
    duration: float = 2.0

    def laws(self, s: Symbols) -> list[Law]:
        """Return the glucose balance and the secretion and clearance of insulin and glucagon.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        intake = s.meal * pulse(s.t, self.meal_time, self.duration) - s.exercise * pulse(
            s.t,
            self.exercise_time,
            self.duration,
        )
        high = s.G**s.n / (s.G**s.n + s.K**s.n)
        return [
            Rate(s.G, s.R - (s.k0 + s.k_i * s.I) * s.G + s.k_n * s.N + intake),
            Rate(s.I, s.s_i * high - s.d_i * s.I),
            Rate(s.N, s.s_n * (1 - high) - s.d_n * s.N),
        ]


class Coagulation(Element):
    """Coagulation cascade: thrombin amplified by positive feedback, switched off by protein C, and fibrinolysis.

    An injury of height `injury` from `injury_time` for `duration` activates thrombin, which activates more of itself
    through the feedback of the cascade, `dT/dt = j(t) + ka T^2 / (k_half^2 + T^2) - (d_t + k_c C) T`. Thrombin
    activates the inhibitor protein C, `dC/dt = s_c T - d_c C`, and converts fibrinogen to fibrin,
    `dF/dt = s_f T - k_p P F`. Fibrin activates plasmin, `dP/dt = s_p F - d_p P`, which dissolves the fibrin.

    Args:
        label:
            The label of the blood.

        T:
            The initial thrombin.

        C:
            The initial activated protein C.

        F:
            The initial fibrin.

        P:
            The initial plasmin.

        ka:
            The maximal rate of the positive feedback.

        k_half:
            The thrombin at which the feedback is half its maximum.

        d_t:
            The inactivation rate of thrombin by antithrombin.

        k_c:
            The inactivation of thrombin per unit of protein C.

        s_c:
            The activation of protein C per unit of thrombin.

        d_c:
            The inactivation rate of protein C.

        s_f:
            The production of fibrin per unit of thrombin.

        k_p:
            The lysis of fibrin per unit of plasmin.

        s_p:
            The activation of plasmin per unit of fibrin.

        d_p:
            The inactivation rate of plasmin.

        injury:
            The rate of thrombin activation by the injury.

        injury_time:
            The start of the injury.

        duration:
            The duration of the injury.

    Examples:
        >>> from skcomplex.models.biology.physiology import Coagulation
        >>> Coagulation('blood').injury_time
        1.0
    """

    T: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    C: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    F: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    P: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    ka: float = parameter(default=3.0, domain=Domain.NONNEGATIVE)
    k_half: float = parameter(default=1.0, domain=Domain.POSITIVE)
    d_t: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    k_c: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    s_c: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)
    d_c: float = parameter(default=0.05, domain=Domain.NONNEGATIVE)
    s_f: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    k_p: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    s_p: float = parameter(default=0.02, domain=Domain.NONNEGATIVE)
    d_p: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)
    injury: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    injury_time: float = 1.0
    duration: float = 1.0

    def laws(self, s: Symbols) -> list[Law]:
        """Return the activation and inactivation of thrombin, protein C, fibrin, and plasmin.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        trigger = s.injury * pulse(s.t, self.injury_time, self.duration)
        feedback = s.ka * s.T**2 / (s.k_half**2 + s.T**2)
        return [
            Rate(s.T, trigger + feedback - (s.d_t + s.k_c * s.C) * s.T),
            Rate(s.C, s.s_c * s.T - s.d_c * s.C),
            Rate(s.F, s.s_f * s.T - s.k_p * s.P * s.F),
            Rate(s.P, s.s_p * s.F - s.d_p * s.P),
        ]


@register_model(
    field='biology/physiology',
    tags=('homeostasis', 'hormones', 'insulin', 'glucagon', 'negative feedback', 'ode'),
    references=('Konieczny et al. 2023',),
    defaults={'until': 24.0, 'dt': 0.05},
)
def glucose_dual_hormone_control(
    *,
    glucagon: bool = True,
    meal: float = 1.0,
    exercise: float = 0.8,
    meal_time: float = 2.0,
    exercise_time: float = 12.0,
    duration: float = 2.0,
) -> System:
    """Return blood glucose stabilised by antagonistic hormone loops, insulin against highs and glucagon against lows.

    A meal raises the glucose and secretes insulin, which raises the uptake of glucose, and an exercise lowers it and
    secretes glucagon, which releases glucose from the liver. Each receptor acts in one direction only, so one loop
    alone corrects excursions to one side of the norm. Two opposing loops on the same quantity attenuate the
    excursions to both sides, after Konieczny et al. The time unit is about one hour, and the glucose is relative to
    its norm.

    Args:
        glucagon:
            Whether the glucagon loop acts, `False` for insulin alone.

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

    Returns:
        system:
            The model, with the glucose `organism.G`, the insulin `organism.I`, and the glucagon `organism.N`.

    Examples:
        >>> from skcomplex.models.biology.physiology import glucose_dual_hormone_control
        >>> glucose_dual_hormone_control(glucagon=False).elements[0].k_n
        0.0
    """
    k_n = 1.0 if glucagon else 0.0
    insulin, glucagon_level = 2.0 * 0.5, 1.0 * 0.5
    organism = GlucoseControl(
        'organism',
        I=insulin,
        N=glucagon_level,
        R=(0.5 + 0.5 * insulin) - k_n * glucagon_level,
        k_n=k_n,
        meal=meal,
        exercise=exercise,
        meal_time=meal_time,
        exercise_time=exercise_time,
        duration=duration,
    )
    return System(components=[organism])


@register_model(
    field='biology/physiology',
    tags=('coagulation', 'positive feedback', 'threshold', 'excitable', 'fibrinolysis', 'ode'),
    references=('Konieczny et al. 2023',),
    defaults={'until': 120.0, 'dt': 0.1},
)
def coagulation_positive_feedback(
    *,
    injury: float = 1.0,
    injury_time: float = 1.0,
    duration: float = 1.0,
    ka: float = 3.0,
) -> System:
    """Return the coagulation cascade, an all-or-nothing burst of thrombin bounded by protein C and fibrinolysis.

    Thrombin activates its own production through the factors of the cascade, a positive feedback that amplifies an
    injury above a threshold into a burst of thrombin and a clot of fibrin, while a weaker injury fades. With the
    default rates, the burst needs a thrombin of `(ka - sqrt(ka^2 - 4)) / 2`, the unstable state of the feedback.
    Thrombin also activates protein C, which switches the burst off, and the fibrin activates plasmin, which dissolves
    the clot, so the clot stays at the site and in time. The positive feedback, the off switch, and the fibrinolysis
    follow Konieczny et al.

    Args:
        injury:
            The rate of thrombin activation by the injury.

        injury_time:
            The start of the injury.

        duration:
            The duration of the injury.

        ka:
            The maximal rate of the positive feedback, which is bistable above 2.

    Returns:
        system:
            The model, with the thrombin `blood.T`, the protein C `blood.C`, the fibrin `blood.F`, and the plasmin
            `blood.P`.

    Examples:
        >>> from skcomplex.models.biology.physiology import coagulation_positive_feedback
        >>> coagulation_positive_feedback(injury=0.2).elements[0].injury
        0.2
    """
    blood = Coagulation('blood', injury=injury, injury_time=injury_time, duration=duration, ka=ka)
    return System(components=[blood])

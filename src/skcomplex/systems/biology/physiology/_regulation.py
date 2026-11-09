"""Regulation on the organism level.

It covers antagonistic hormone loops and an extracellular cascade with positive feedback.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import RateInteraction
from ....core.components.laws.utils import delayed, hill, pulse
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import DeclaredVariableSpec, ParameterSpec, StateSpec, VisualSpec
from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations
from ...general.reactions import ReactionInteraction


class CoagulationElement(Element):
    """Coagulation cascade, thrombin amplified by positive feedback.

    Protein C switches it off, and fibrinolysis clears the clot.

    An injury of height `injury` from `injury_time` for `duration` activates thrombin, which activates more of itself
    through the feedback of the cascade, `dT/dt = j(t) + ka T^2 / (k_half^2 + T^2) - (d_t + k_c C) T`. Thrombin
    activates the inhibitor protein C, `dC/dt = s_c T - d_c C`, and converts fibrinogen to fibrin,
    `dF/dt = s_f T - k_p P F`. Fibrin activates plasmin, `dP/dt = s_p F - d_p P`, which dissolves the fibrin.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `CoagulationPositiveFeedbackSystem` read its parameters.

    Args:
        label:
            The label of the blood.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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
        >>> from skcomplex.systems.biology.physiology import CoagulationElement
        >>> CoagulationElement('blood').injury_time
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        T: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        C: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        F: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        P: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        ka: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3.0,
        k_half: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        d_t: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        k_c: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        s_c: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        d_c: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        s_f: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        k_p: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        s_p: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.02,
        d_p: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        injury: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        injury_time: float = 1.0,
        duration: float = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.T = T
        self.C = C
        self.F = F
        self.P = P
        self.ka = ka
        self.k_half = k_half
        self.d_t = d_t
        self.k_c = k_c
        self.s_c = s_c
        self.d_c = d_c
        self.s_f = s_f
        self.k_p = k_p
        self.s_p = s_p
        self.d_p = d_p
        self.injury = injury
        self.injury_time = injury_time
        self.duration = duration


class GlucoseControlElement(Element):
    """Blood glucose held near its norm by insulin and glucagon.

    Insulin lowers it, and glucagon raises it.

    The glucose follows `dG/dt = R - (k0 + k_i I) G + k_n N + u(t)`, where `u` is a meal of height `meal` from
    `meal_time` and an exercise of height `-exercise` from `exercise_time`, each lasting `duration`. High glucose
    secretes insulin, `dI/dt = s_i h(G) - d_i I`, and low glucose secretes glucagon, `dN/dt = s_n (1 - h(G)) - d_n N`,
    with the Hill switch `h(G) = G^n / (G^n + K^n)`. The production `R` is set so that the norm `G = K` is a steady
    state.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `GlucoseDualHormoneControlSystem` read its parameters.

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


class GlucoseDualHormoneControlSystem(BaseSystem):
    """Blood glucose stabilised by antagonistic hormone loops.

    Insulin acts against highs and glucagon against lows.

    A meal raises the glucose and secretes insulin, which raises the uptake of glucose, and an exercise lowers it and
    secretes glucagon, which releases glucose from the liver. Each receptor acts in one direction only, so one loop
    alone corrects excursions to one side of the norm. Two opposing loops on the same quantity attenuate the excursions
    to both sides, after Konieczny et al. The time unit is about one hour, and the glucose is relative to its norm. The
    system has the glucose `organism.G`, the insulin `organism.I`, and the glucagon `organism.N`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.physiology import GlucoseDualHormoneControlSystem
        >>> GlucoseDualHormoneControlSystem(glucagon=False).prepare().system_.elements[0].k_n
        0.0
    """

    field = 'biology/physiology'
    topics = ('homeostasis', 'hormones', 'insulin', 'glucagon', 'negative feedback', 'ode')
    references = ('Konieczny et al. 2023',)
    until = 24.0
    dt = 0.05

    def __init__(
        self,
        *,
        glucagon: bool = True,
        meal: float = 1.0,
        exercise: float = 0.8,
        meal_time: float = 2.0,
        exercise_time: float = 12.0,
        duration: float = 2.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.glucagon = glucagon
        self.meal = meal
        self.exercise = exercise
        self.meal_time = meal_time
        self.exercise_time = exercise_time
        self.duration = duration

    def _build_system(self) -> CompositeSystem:
        """Return the parts of blood glucose stabilised by antagonistic hormone loops."""
        k_n = 1.0 if self.glucagon else 0.0
        insulin, glucagon_level = 2.0 * 0.5, 1.0 * 0.5
        organism = GlucoseControlElement(
            'organism',
            I=insulin,
            N=glucagon_level,
            R=(0.5 + 0.5 * insulin) - k_n * glucagon_level,
            k_n=k_n,
            meal=self.meal,
            exercise=self.exercise,
            meal_time=self.meal_time,
            exercise_time=self.exercise_time,
            duration=self.duration,
        )
        interactions = [
            ReactionInteraction('release', target='organism', products={'G': 1}, propensity=lambda s: s.target.R),
            ReactionInteraction(
                'basal_uptake',
                target='organism',
                reactants={'G': 1},
                propensity=lambda s: s.target.k0 * s.target.G,
            ),
            ReactionInteraction(
                'insulin_uptake',
                target='organism',
                reactants={'G': 1},
                modifiers={'I': 1},
                propensity=lambda s: s.target.k_i * s.target.I * s.target.G,
            ),
            ReactionInteraction(
                'glucagon_release',
                target='organism',
                products={'G': 1},
                propensity=lambda s: s.target.k_n * s.target.N,
            ),
            RateInteraction(
                'meal',
                target='organism',
                state='G',
                rate=lambda s: s.target.meal * pulse(s.t, organism.meal_time, organism.duration),
            ),
            RateInteraction(
                'exercise',
                target='organism',
                state='G',
                rate=lambda s: -s.target.exercise * pulse(s.t, organism.exercise_time, organism.duration),
            ),
            ReactionInteraction(
                'insulin_secretion',
                target='organism',
                products={'I': 1},
                propensity=lambda s: s.target.s_i * hill(s.target.G, s.target.K, s.target.n),
            ),
            ReactionInteraction(
                'insulin_clearance',
                target='organism',
                reactants={'I': 1},
                propensity=lambda s: s.target.d_i * s.target.I,
            ),
            ReactionInteraction(
                'glucagon_secretion',
                target='organism',
                products={'N': 1},
                propensity=lambda s: s.target.s_n * (1 - hill(s.target.G, s.target.K, s.target.n)),
            ),
            ReactionInteraction(
                'glucagon_clearance',
                target='organism',
                reactants={'N': 1},
                propensity=lambda s: s.target.d_n * s.target.N,
            ),
        ]
        return CompositeSystem(self.label, elements=[organism], interactions=interactions)


class CoagulationPositiveFeedbackSystem(BaseSystem):
    """The coagulation cascade, an all-or-nothing burst of thrombin.

    Protein C and fibrinolysis bound the burst.

    Thrombin activates its own production through the factors of the cascade, a positive feedback that amplifies an
    injury above a threshold into a burst of thrombin and a clot of fibrin, while a weaker injury fades. With the
    default rates, the burst needs a thrombin of `(ka - sqrt(ka^2 - 4)) / 2`, the unstable state of the feedback.
    Thrombin also activates protein C, which switches the burst off, and the fibrin activates plasmin, which dissolves
    the clot, so the clot stays at the site and in time. The positive feedback, the off switch, and the fibrinolysis
    follow Konieczny et al. The system has the thrombin `blood.T`, the protein C `blood.C`, the fibrin `blood.F`, and
    the plasmin `blood.P`.

    Args:
        injury:
            The rate of thrombin activation by the injury.

        injury_time:
            The start of the injury.

        duration:
            The duration of the injury.

        ka:
            The maximal rate of the positive feedback, which is bistable above 2.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.physiology import CoagulationPositiveFeedbackSystem
        >>> CoagulationPositiveFeedbackSystem(injury=0.2).prepare().system_.elements[0].injury
        0.2
    """

    field = 'biology/physiology'
    topics = ('coagulation', 'positive feedback', 'threshold', 'excitable', 'fibrinolysis', 'ode')
    references = ('Konieczny et al. 2023',)
    until = 120.0
    dt = 0.1

    def __init__(
        self,
        *,
        injury: float = 1.0,
        injury_time: float = 1.0,
        duration: float = 1.0,
        ka: float = 3.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.injury = injury
        self.injury_time = injury_time
        self.duration = duration
        self.ka = ka

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the coagulation cascade."""
        blood = CoagulationElement(
            'blood',
            injury=self.injury,
            injury_time=self.injury_time,
            duration=self.duration,
            ka=self.ka,
        )
        interactions = [
            RateInteraction(
                'trigger',
                target='blood',
                state='T',
                rate=lambda s: s.target.injury * pulse(s.t, blood.injury_time, blood.duration),
            ),
            ReactionInteraction(
                'thrombin_feedback',
                target='blood',
                products={'T': 1},
                propensity=lambda s: s.target.ka * hill(s.target.T, s.target.k_half, 2),
            ),
            ReactionInteraction(
                'thrombin_decay',
                target='blood',
                reactants={'T': 1},
                propensity=lambda s: s.target.d_t * s.target.T,
            ),
            ReactionInteraction(
                'thrombin_inhibition',
                target='blood',
                reactants={'T': 1},
                modifiers={'C': 1},
                propensity=lambda s: s.target.k_c * s.target.C * s.target.T,
            ),
            ReactionInteraction(
                'inhibitor_production',
                target='blood',
                products={'C': 1},
                propensity=lambda s: s.target.s_c * s.target.T,
            ),
            ReactionInteraction(
                'inhibitor_decay',
                target='blood',
                reactants={'C': 1},
                propensity=lambda s: s.target.d_c * s.target.C,
            ),
            ReactionInteraction(
                'fibrin_production',
                target='blood',
                products={'F': 1},
                propensity=lambda s: s.target.s_f * s.target.T,
            ),
            ReactionInteraction(
                'fibrinolysis',
                target='blood',
                reactants={'F': 1},
                modifiers={'P': 1},
                propensity=lambda s: s.target.k_p * s.target.P * s.target.F,
            ),
            ReactionInteraction(
                'plasmin_production',
                target='blood',
                products={'P': 1},
                propensity=lambda s: s.target.s_p * s.target.F,
            ),
            ReactionInteraction(
                'plasmin_decay',
                target='blood',
                reactants={'P': 1},
                propensity=lambda s: s.target.d_p * s.target.P,
            ),
        ]
        return CompositeSystem(self.label, elements=[blood], interactions=interactions)


class SteerableFeedbackLoopSystem(DynamicalSystem):
    """A receptor-effector loop that holds a product at its set point.

    It holds it until a steering signal shifts it.

    The loop of the negative feedback system of Konieczny et al. (2023) settles at the set point where synthesis `v / (1
    + (x / K)^n)` balances consumption `d x`. A steering signal, such as a bacterial toxin acting on the temperature
    receptors in fever, raises the sensitivity of the receptor by the factor `1 + steering`, so the loop climbs to a
    higher set point while it acts and returns once it ends. A slow effector or a long transmission delay turns the
    regulation into sustained oscillations around the set point. The system has the product `loop.x` and the effector
    activity `loop.e`.

    Args:
        v:
            The largest rate of synthesis.

        d:
            The consumption rate of the product.

        K:
            The sensitivity of the receptor.

        n:
            The steepness of the receptor.

        T:
            The response time of the effector.

        tau:
            The transmission delay between the receptor and the effector.

        steering:
            The relative shift of the sensitivity by the steering signal.

        start:
            The time the steering signal starts.

        duration:
            The duration of the steering signal.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.physiology import SteerableFeedbackLoopSystem
        >>> SteerableFeedbackLoopSystem(tau=3.0).prepare().system_.elements[0].get_value('tau')
        3.0
    """

    field = 'biology/physiology'
    topics = ('negative feedback', 'homeostasis', 'set point', 'fever', 'delay')
    references = ('Wach et al. 2015', 'Konieczny et al. 2023')
    until = 60.0
    dt = 0.1

    def __init__(
        self,
        *,
        v: float = 2.0,
        d: float = 1.0,
        K: float = 1.0,
        n: float = 4.0,
        T: float = 1.0,
        tau: float = 0.5,
        steering: float = 0.5,
        start: float = 20.0,
        duration: float = 20.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.v = v
        self.d = d
        self.K = K
        self.n = n
        self.T = T
        self.tau = tau
        self.steering = steering
        self.start = start
        self.duration = duration

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'loop'

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the states and the parameters with their specs."""
        return {
            'x': DeclaredVariableSpec(0.0, StateSpec(domain=RealSet.closed_open(0, oo))),
            'e': DeclaredVariableSpec(0.0, StateSpec(domain=RealSet.closed(0, 1))),
            'v': DeclaredVariableSpec(self.v, ParameterSpec(domain=RealSet.open(0, oo))),
            'd': DeclaredVariableSpec(self.d, ParameterSpec(domain=RealSet.open(0, oo))),
            'K': DeclaredVariableSpec(self.K, ParameterSpec(domain=RealSet.open(0, oo))),
            'n': DeclaredVariableSpec(self.n, ParameterSpec(domain=RealSet.open(0, oo))),
            'T': DeclaredVariableSpec(self.T, ParameterSpec(domain=RealSet.open(0, oo))),
            'tau': DeclaredVariableSpec(self.tau, ParameterSpec(domain=RealSet.open(0, oo))),
            'steering': DeclaredVariableSpec(self.steering, ParameterSpec()),
        }

    def _get_equations(self) -> Equations:
        """Return the synthesis and consumption of the product and the effector response."""
        return {
            'x': lambda s: s.target.v * s.target.e - s.target.d * s.target.x,
            'e': lambda s: (self._receptor(s) - s.target.e) / s.target.T,
        }

    def _sensitivity(self, s: SimpleNamespace) -> Expression:
        """Return `sensitivity` from the namespace of an equation."""
        return s.target.K * (1 + s.target.steering * pulse(s.t, self.start, self.duration))

    def _receptor(self, s: SimpleNamespace) -> Expression:
        """Return `receptor` from the namespace of an equation."""
        return 1 / (1 + (delayed(s.target.x, s.target.tau) / self._sensitivity(s)) ** s.target.n)

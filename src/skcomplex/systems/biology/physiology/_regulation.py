"""Regulation on the organism level: antagonistic hormone loops and an extracellular cascade with positive feedback."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ....core.components.interactions import RateInteraction, ReactionInteraction
from ....core.components.laws.utils import hill, pulse
from ....core.components.system import System
from ..._catalog import register_system
from ._coagulation import CoagulationElement
from ._feedback_loop import FeedbackLoopElement
from ._glucose_control import GlucoseControlElement


@register_system(
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
            The system, with the glucose `organism.G`, the insulin `organism.I`, and the glucagon `organism.N`.

    Examples:
        >>> from skcomplex.systems.biology.physiology import glucose_dual_hormone_control
        >>> glucose_dual_hormone_control(glucagon=False).elements[0].k_n
        0.0
    """
    k_n = 1.0 if glucagon else 0.0
    insulin, glucagon_level = 2.0 * 0.5, 1.0 * 0.5
    organism = GlucoseControlElement(
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
    return System(elements=[organism], interactions=interactions)


@register_system(
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
            The system, with the thrombin `blood.T`, the protein C `blood.C`, the fibrin `blood.F`, and the plasmin
            `blood.P`.

    Examples:
        >>> from skcomplex.systems.biology.physiology import coagulation_positive_feedback
        >>> coagulation_positive_feedback(injury=0.2).elements[0].injury
        0.2
    """
    blood = CoagulationElement('blood', injury=injury, injury_time=injury_time, duration=duration, ka=ka)
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
    return System(elements=[blood], interactions=interactions)


@register_system(
    field='biology/physiology',
    tags=('negative feedback', 'homeostasis', 'set point', 'fever', 'delay'),
    references=('Wach et al. 2015', 'Konieczny et al. 2023'),
    defaults={'until': 60.0, 'dt': 0.1},
)
def steerable_feedback_loop(
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
) -> System:
    """Return a receptor-effector loop that holds a product at its set point until a steering signal shifts it.

    The loop of the negative feedback system of Konieczny et al. (2023) settles at the set point where synthesis
    `v / (1 + (x / K)^n)` balances consumption `d x`. A steering signal, such as a bacterial toxin acting on the
    temperature receptors in fever, raises the sensitivity of the receptor by the factor `1 + steering`, so the loop
    climbs to a higher set point while it acts and returns once it ends. A slow effector or a long transmission delay
    turns the regulation into sustained oscillations around the set point.

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

    Returns:
        system:
            The system, with the product `loop.x` and the effector activity `loop.e`.

    Examples:
        >>> from skcomplex.systems.biology.physiology import steerable_feedback_loop
        >>> steerable_feedback_loop(tau=3.0).elements[0].tau
        3.0
    """
    loop = FeedbackLoopElement(
        'loop',
        v=v,
        d=d,
        K=K,
        n=n,
        T=T,
        tau=tau,
        steering=steering,
        start=start,
        duration=duration,
    )
    return System(elements=[loop])

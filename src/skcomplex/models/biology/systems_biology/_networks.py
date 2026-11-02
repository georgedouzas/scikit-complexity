"""Systems biology models: cellular oscillators and power-law rate equations."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Sequence

from ....core.interactions import BaseInteraction, ReactionInteraction
from ....core.system import System
from ....errors import ValidationError
from ..._catalog import register_model
from ..common.kinetics import MichaelisMentenInteraction
from ._generalized_mass_action import GeneralizedMassActionInteraction
from ._phosphorylation_loop import PhosphorylationLoopElement
from ._repression import RepressionInteraction
from ._s_system import SSystemInteraction
from ._species import SpeciesElement

RAMAN = 'Raman 2021'


PATHWAY_ALPHA = (2.0, 1.0, 1.0)


PATHWAY_BETA = (1.0, 1.0, 1.0)


PATHWAY_G = ((0.0, 0.0, -0.5), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0))


PATHWAY_H = ((0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5))


PATHWAY_STOICHIOMETRY = ((1.0, -1.0, 0.0, 0.0), (0.0, 1.0, -1.0, 0.0), (0.0, 0.0, 1.0, -1.0))


PATHWAY_GAMMA = (2.0, 1.0, 1.0, 1.0)


PATHWAY_ORDERS = ((0.0, 0.0, -0.5), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5))


MIN_GOODWIN_STAGES = 2


@register_model(
    field='biology/systems_biology',
    tags=('oscillator', 'phosphorylation', 'Michaelis-Menten', 'negative feedback', 'ode'),
    references=(RAMAN, 'Mogilner, Wollman, and Marshall 2006'),
    defaults={'until': 40.0, 'dt': 0.05},
)
def phosphorylation_oscillator(
    A: float = 1.0,
    B: float = 0.0,
    C: float = 0.0,
    *,
    kp1: float = 1.0,
    kd1: float = 2.0,
    kp2: float = 2.0,
    kd2: float = 1.0,
    kp3: float = 2.0,
    kd3: float = 1.0,
    K: float = 0.01,
) -> System:
    """Return three proteins whose phosphorylation cycles regulate each other in a loop that oscillates.

    C converts A, A converts B, and B converts C, each reaction with Michaelis-Menten kinetics and each total
    conserved, so the loop is a negative feedback of three steps. The book prints the equations and no parameter
    values. The defaults are a choice: totals of 1, Michaelis constants `K` far below the totals, which make every
    cycle an ultrasensitive switch, and rates that put every switch at half the total of its enzyme. The proteins then
    switch on and off in turn on a limit cycle of period about 4.2. With large `K` the switches become graded and the
    loop settles on a fixed point. The amounts are fractions of the totals.

    Args:
        A:
            The initial amount of unphosphorylated A.

        B:
            The initial amount of unphosphorylated B.

        C:
            The initial amount of unphosphorylated C.

        kp1:
            The maximal rate that restores A.

        kd1:
            The catalytic rate of the conversion of A by C.

        kp2:
            The catalytic rate of the conversion of B by A.

        kd2:
            The maximal rate that removes B.

        kp3:
            The catalytic rate of the conversion of C by B.

        kd3:
            The maximal rate that removes C.

        K:
            The Michaelis constant of every reaction.

    Returns:
        system:
            The model, with the amounts `loop.A`, `loop.B`, and `loop.C`.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import phosphorylation_oscillator
        >>> phosphorylation_oscillator(K=0.1).elements[0].K2
        0.1
    """
    loop = PhosphorylationLoopElement(
        'loop',
        A=A,
        B=B,
        C=C,
        kp1=kp1,
        kd1=kd1,
        kp2=kp2,
        kd2=kd2,
        kp3=kp3,
        kd3=kd3,
        K1=K,
        K2=K,
        K3=K,
    )
    reactions = [
        MichaelisMentenInteraction(
            'a_restoration',
            target='loop',
            products={'A': 1},
            substrate=lambda s: s.target.A_T - s.target.A,
            vmax=lambda s: s.target.kp1,
            km=lambda s: s.target.K1,
        ),
        MichaelisMentenInteraction(
            'a_conversion',
            target='loop',
            reactants={'A': 1},
            substrate='A',
            vmax=lambda s: s.target.kd1,
            km=lambda s: s.target.K1,
            modifiers={'C': 1},
        ),
        MichaelisMentenInteraction(
            'b_conversion',
            target='loop',
            products={'B': 1},
            substrate=lambda s: s.target.B_T - s.target.B,
            vmax=lambda s: s.target.kp2,
            km=lambda s: s.target.K2,
            modifiers={'A': 1},
        ),
        MichaelisMentenInteraction(
            'b_removal',
            target='loop',
            reactants={'B': 1},
            substrate='B',
            vmax=lambda s: s.target.kd2,
            km=lambda s: s.target.K2,
        ),
        MichaelisMentenInteraction(
            'c_conversion',
            target='loop',
            products={'C': 1},
            substrate=lambda s: s.target.C_T - s.target.C,
            vmax=lambda s: s.target.kp3,
            km=lambda s: s.target.K3,
            modifiers={'B': 1},
        ),
        MichaelisMentenInteraction(
            'c_removal',
            target='loop',
            reactants={'C': 1},
            substrate='C',
            vmax=lambda s: s.target.kd3,
            km=lambda s: s.target.K3,
        ),
    ]
    return System(elements=[loop], interactions=reactions)


@register_model(
    field='biology/systems_biology',
    tags=('biochemical systems theory', 'power law', 'S-system', 'generalized mass action', 'ode'),
    references=(RAMAN, 'Savageau 1976', 'Voit 2000'),
    defaults={'until': 20.0, 'dt': 0.05},
)
def biochemical_systems_theory(
    x: Sequence[float] = (0.5, 1.0, 4.0),
    *,
    form: str = 's-system',
    alpha: Sequence[float] = PATHWAY_ALPHA,
    beta: Sequence[float] = PATHWAY_BETA,
    g: Sequence[Sequence[float]] = PATHWAY_G,
    h: Sequence[Sequence[float]] = PATHWAY_H,
    stoichiometry: Sequence[Sequence[float]] = PATHWAY_STOICHIOMETRY,
    gamma: Sequence[float] = PATHWAY_GAMMA,
    orders: Sequence[Sequence[float]] = PATHWAY_ORDERS,
) -> System:
    """Return a biochemical system of power-law rates, in the S-system or the generalized mass action form.

    The S-system form gives every species one production and one degradation,
    `dx_i/dt = alpha_i prod_j x_j^g_ij - beta_i prod_j x_j^h_ij`. The generalized mass action form sums power-law
    fluxes, `dx_i/dt = sum_j s_ij gamma_j prod_k x_k^f_jk`. The book states only the two forms. The default is a
    choice, an unbranched pathway `X1 -> X2 -> X3` whose end product inhibits the supply of `X1`, written in both forms
    with the same rates, `dX1/dt = 2 X3^-0.5 - X1^0.5`, `dX2/dt = X1^0.5 - X2^0.5`, and `dX3/dt = X2^0.5 - X3^0.5`.
    The steady state of an S-system solves the linear system `(g - h) log x = log(beta / alpha)`, here `x = (2, 2, 2)`.

    Args:
        x:
            The initial concentrations of the species.

        form:
            `'s-system'` or `'gma'`.

        alpha:
            The rate constants of the productions of the S-system.

        beta:
            The rate constants of the degradations of the S-system.

        g:
            The kinetic orders of the productions of the S-system, one row per species.

        h:
            The kinetic orders of the degradations of the S-system, one row per species.

        stoichiometry:
            The stoichiometric coefficients of the generalized mass action form, one row per species.

        gamma:
            The rate constants of the fluxes of the generalized mass action form.

        orders:
            The kinetic orders of the fluxes of the generalized mass action form, one row per flux.

    Returns:
        system:
            The model, with the concentrations `X1.x`, `X2.x`, and so on.

    Raises:
        ValidationError:
            The form is unknown.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import biochemical_systems_theory
        >>> biochemical_systems_theory(form='gma').interactions[0].gamma
        (2.0, 1.0, 1.0, 1.0)
    """
    species = [SpeciesElement(f'X{index}', x=float(value)) for index, value in enumerate(x, start=1)]
    paths = [item.label for item in species]
    if form == 's-system':
        rates: BaseInteraction = SSystemInteraction(
            'rates',
            species=paths,
            alpha=[float(value) for value in alpha],
            beta=[float(value) for value in beta],
            g=tuple(tuple(row) for row in g),
            h=tuple(tuple(row) for row in h),
        )
    elif form == 'gma':
        rates = GeneralizedMassActionInteraction(
            'rates',
            species=paths,
            stoichiometry=tuple(tuple(row) for row in stoichiometry),
            gamma=tuple(gamma),
            orders=tuple(tuple(row) for row in orders),
        )
    else:
        error_msg = f'Unknown form {form!r}. Use s-system or gma.'
        raise ValidationError(error_msg)
    return System(elements=species, interactions=[rates])


@register_model(
    field='biology/systems_biology',
    tags=('oscillator', 'negative feedback', 'Hill function', 'gene regulation', 'ode'),
    references=(RAMAN, 'Goodwin 1965', 'Griffith 1968'),
    defaults={'until': 60.0, 'dt': 0.1},
)
def goodwin_oscillator(
    variables: int = 3,
    x: Sequence[float] | None = None,
    *,
    a: float = 100.0,
    b: float = 1.0,
    K: float = 1.0,
    n: float = 12.0,
    k: float = 1.0,
) -> System:
    """Return the Goodwin oscillator, a chain of stages whose last one represses the synthesis of the first.

    The first stage, an mRNA, is synthesized at `a / (1 + (x_N / K)^n)` and every later stage at `k` times the stage
    before it, and every stage decays at rate `b`: `dx_1/dt = a / (1 + (x_N / K)^n) - b x_1` and
    `dx_i/dt = k x_(i-1) - b x_i`. The book points to BioModels and prints no equations, so the model takes the standard
    form of Goodwin with the degradation of Griffith. With equal degradation rates the fixed point loses stability when
    the loop gain passes `sec(pi / N)^N`, 8 for three stages. When `a` is large the repressor saturates and the gain is
    close to `n`, so three stages oscillate for Hill coefficients above about 8 and settle for lower ones. The
    concentrations and the time are dimensionless.

    Args:
        variables:
            The number of stages, at least 2.

        x:
            The initial concentrations of the stages, or `None` for 0.1, 0.2, and so on.

        a:
            The maximal synthesis rate of the first stage.

        b:
            The degradation rate of every stage.

        K:
            The concentration of the last stage at half repression.

        n:
            The Hill coefficient of the repression.

        k:
            The synthesis rate of every later stage per unit of the stage before it.

    Returns:
        system:
            The model, with the concentrations `x1.x` to `x<variables>.x`.

    Raises:
        ValidationError:
            There are fewer than two stages, or the initial concentrations do not match them.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import goodwin_oscillator
        >>> [stage.label for stage in goodwin_oscillator(4).elements]
        ['x1', 'x2', 'x3', 'x4']
    """
    initial = [0.1 * index for index in range(1, variables + 1)] if x is None else list(x)
    if variables < MIN_GOODWIN_STAGES or len(initial) != variables:
        error_msg = f'The Goodwin loop needs at least 2 stages and one initial value per stage, not {variables} stages.'
        raise ValidationError(error_msg)
    labels = [f'x{index}' for index in range(1, variables + 1)]
    stages = [SpeciesElement(label, x=float(value)) for label, value in zip(labels, initial, strict=True)]
    repression = RepressionInteraction(
        'repression',
        products={(labels[0], 'x'): 1},
        modifiers={(labels[-1], 'x'): 1},
        alpha=a,
        K=K,
        n=n,
    )
    syntheses = [
        ReactionInteraction(f'synthesis_{target}', products={(target, 'x'): 1}, modifiers={(source, 'x'): 1}, k=k)
        for source, target in itertools.pairwise(labels)
    ]
    degradations = [ReactionInteraction(f'degradation_{label}', reactants={(label, 'x'): 1}, k=b) for label in labels]
    return System(elements=stages, interactions=[repression, *syntheses, *degradations])

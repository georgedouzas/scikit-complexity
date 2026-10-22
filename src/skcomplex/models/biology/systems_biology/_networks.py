"""Systems biology models: cellular oscillators and power-law rate equations."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Sequence
from math import prod

from sage.all__sagemath_symbolics import SR

from ....core.base import Domain, Expression, ValidationError, parameter, role, state
from ....core.components import Element
from ....core.interactions import Interaction
from ....core.laws import Law, Rate, Symbols
from ....core.runtime import register_model
from ....core.system import System

RAMAN = 'Raman 2021'
PATHWAY_ALPHA = (2.0, 1.0, 1.0)
PATHWAY_BETA = (1.0, 1.0, 1.0)
PATHWAY_G = ((0.0, 0.0, -0.5), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0))
PATHWAY_H = ((0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5))
PATHWAY_STOICHIOMETRY = ((1.0, -1.0, 0.0, 0.0), (0.0, 1.0, -1.0, 0.0), (0.0, 0.0, 1.0, -1.0))
PATHWAY_GAMMA = (2.0, 1.0, 1.0, 1.0)
PATHWAY_ORDERS = ((0.0, 0.0, -0.5), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5))


def _power_product(values: Sequence[Expression], orders: Sequence[float]) -> Expression:
    """Return the product `prod_k x_k^f_k` of the values raised to their kinetic orders, leaving out zero orders."""
    return prod((value**order for value, order in zip(values, orders, strict=True) if order), start=SR(1))


def _check_shape(name: str, matrix: Sequence[Sequence[float]], rows: int, columns: int) -> None:
    """Raise an error unless a matrix has the given numbers of rows and columns."""
    if len(matrix) == rows and all(len(row) == columns for row in matrix):
        return
    error_msg = f'The matrix {name} must have {rows} rows of {columns} entries, one row per flux or species.'
    raise ValidationError(error_msg)


class PhosphorylationLoop(Element):
    """Three proteins, each phosphorylated and dephosphorylated by Michaelis-Menten kinetics, that regulate each other.

    The states are the unphosphorylated amounts, and each total is conserved. With `u_T = A_T - A`, and so on:

    - `dA/dt = kp1 u_T / (K1 + u_T) - kd1 C A / (K1 + A)`,
    - `dB/dt = kp2 A (B_T - B) / (K2 + B_T - B) - kd2 B / (K2 + B)`,
    - `dC/dt = kp3 B (C_T - C) / (K3 + C_T - C) - kd3 C / (K3 + C)`.

    Args:
        label:
            The label of the loop.

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

        K1:
            The Michaelis constant of both reactions of A.

        K2:
            The Michaelis constant of both reactions of B.

        K3:
            The Michaelis constant of both reactions of C.

        A_T:
            The total amount of A.

        B_T:
            The total amount of B.

        C_T:
            The total amount of C.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import PhosphorylationLoop
        >>> PhosphorylationLoop('loop', A=0.5).A
        0.5
    """

    A: float = state(default=1.0, domain=Domain.NONNEGATIVE)
    B: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    C: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    kp1: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    kd1: float = parameter(default=2.0, domain=Domain.NONNEGATIVE)
    kp2: float = parameter(default=2.0, domain=Domain.NONNEGATIVE)
    kd2: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    kp3: float = parameter(default=2.0, domain=Domain.NONNEGATIVE)
    kd3: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    K1: float = parameter(default=0.01, domain=Domain.POSITIVE)
    K2: float = parameter(default=0.01, domain=Domain.POSITIVE)
    K3: float = parameter(default=0.01, domain=Domain.POSITIVE)
    A_T: float = parameter(default=1.0, domain=Domain.POSITIVE)
    B_T: float = parameter(default=1.0, domain=Domain.POSITIVE)
    C_T: float = parameter(default=1.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the Michaelis-Menten rates of the three proteins.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        A, B, C = s.A, s.B, s.C
        return [
            Rate(A, s.kp1 * (s.A_T - A) / (s.K1 + s.A_T - A) - s.kd1 * C * A / (s.K1 + A)),
            Rate(B, s.kp2 * A * (s.B_T - B) / (s.K2 + s.B_T - B) - s.kd2 * B / (s.K2 + B)),
            Rate(C, s.kp3 * B * (s.C_T - C) / (s.K3 + s.C_T - C) - s.kd3 * C / (s.K3 + C)),
        ]


class Metabolite(Element):
    """Metabolite or other species of a biochemical system, with a positive concentration.

    Args:
        label:
            The label of the metabolite.

        x:
            The initial concentration.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import Metabolite
        >>> Metabolite('X1', x=2.0).x
        2.0
    """

    x: float = state(default=1.0, domain=Domain.POSITIVE)


class GeneralizedMassAction(Interaction):
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

    Examples:
        >>> from skcomplex.models.biology.systems_biology import GeneralizedMassAction
        >>> v = GeneralizedMassAction('v', species=['X'], stoichiometry=((1, -1),), gamma=(1, 2), orders=((0,), (1,)))
        >>> v.gamma
        (1, 2)
    """

    species: list[str] = role(variadic=True)
    stoichiometry: Sequence[Sequence[float]] = ((0.0,),)
    gamma: Sequence[float] = (0.0,)
    orders: Sequence[Sequence[float]] = ((0.0,),)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the sum of the power-law fluxes of every species.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.

        Raises:
            ValidationError:
                The stoichiometry or the kinetic orders do not match the species and the fluxes.
        """
        values = [item.x for item in s.species]
        _check_shape('stoichiometry', self.stoichiometry, len(values), len(self.gamma))
        _check_shape('orders', self.orders, len(self.gamma), len(values))
        fluxes = [rate * _power_product(values, row) for rate, row in zip(self.gamma, self.orders, strict=True)]
        return [
            Rate(value, sum((c * flux for c, flux in zip(row, fluxes, strict=True) if c), start=SR(0)))
            for value, row in zip(values, self.stoichiometry, strict=True)
        ]


class SSystem(Interaction):
    """Rates of a biochemical system in the S-system form of biochemical systems theory.

    Every species has one aggregated production and one aggregated degradation, both power laws:
    `dx_i/dt = alpha_i prod_j x_j^g_ij - beta_i prod_j x_j^h_ij`.

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

    Examples:
        >>> from skcomplex.models.biology.systems_biology import SSystem
        >>> SSystem('rates', species=['X'], alpha=[1.0], beta=[1.0], g=((0.0,),), h=((1.0,),)).h
        ((1.0,),)
    """

    species: list[str] = role(variadic=True)
    alpha: list[float] = parameter(default=(), per='species', domain=Domain.NONNEGATIVE)
    beta: list[float] = parameter(default=(), per='species', domain=Domain.NONNEGATIVE)
    g: Sequence[Sequence[float]] = ((0.0,),)
    h: Sequence[Sequence[float]] = ((0.0,),)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the production minus the degradation of every species.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.

        Raises:
            ValidationError:
                The kinetic orders do not have one row and one column per species.
        """
        values = [item.x for item in s.species]
        _check_shape('g', self.g, len(values), len(values))
        _check_shape('h', self.h, len(values), len(values))
        return [
            Rate(value, alpha * _power_product(values, g) - beta * _power_product(values, h))
            for value, alpha, beta, g, h in zip(values, s.alpha, s.beta, self.g, self.h, strict=True)
        ]


class GoodwinStage(Element):
    """Stage of the Goodwin loop, an mRNA, an enzyme, or a product, which decays at a constant rate.

    Args:
        label:
            The label of the stage.

        x:
            The initial concentration.

        b:
            The degradation rate.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import GoodwinStage
        >>> GoodwinStage('mRNA', x=0.1).x
        0.1
    """

    x: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    b: float = parameter(default=1.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the decay `-b x`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rate(s.x, -s.b * s.x)]


class HillRepression(Interaction):
    """Synthesis of a target repressed by a Hill function of a repressor, `a / (1 + (y / K)^n)`.

    Args:
        label:
            The label of the interaction.

        repressor:
            The path of the repressing stage.

        target:
            The path of the synthesized stage.

        a:
            The synthesis rate without repressor.

        K:
            The repressor concentration at half repression.

        n:
            The Hill coefficient.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import HillRepression
        >>> HillRepression('repression', repressor='product', target='mRNA', n=12.0).n
        12.0
    """

    repressor: str = role()
    target: str = role()
    a: float = parameter(default=100.0, domain=Domain.NONNEGATIVE)
    K: float = parameter(default=1.0, domain=Domain.POSITIVE)
    n: float = parameter(default=12.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the repressed synthesis of the target.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rate(s.target.x, s.a / (1 + (s.repressor.x / s.K) ** s.n))]


class Synthesis(Interaction):
    """Synthesis of a target in proportion to a source, `k x`, such as translation or an enzymatic product.

    Args:
        label:
            The label of the interaction.

        source:
            The path of the stage that drives the synthesis.

        target:
            The path of the synthesized stage.

        k:
            The synthesis rate per unit of source.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import Synthesis
        >>> Synthesis('translation', source='mRNA', target='enzyme', k=2.0).k
        2.0
    """

    source: str = role()
    target: str = role()
    k: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the synthesis `k x` of the target.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rate(s.target.x, s.k * s.source.x)]


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
    loop = PhosphorylationLoop(
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
    return System(components=[loop])


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
    species = [Metabolite(f'X{index}', x=float(value)) for index, value in enumerate(x, start=1)]
    paths = [item.label for item in species]
    if form == 's-system':
        rates: Interaction = SSystem(
            'rates',
            species=paths,
            alpha=[float(value) for value in alpha],
            beta=[float(value) for value in beta],
            g=tuple(tuple(row) for row in g),
            h=tuple(tuple(row) for row in h),
        )
    elif form == 'gma':
        rates = GeneralizedMassAction(
            'rates',
            species=paths,
            stoichiometry=tuple(tuple(row) for row in stoichiometry),
            gamma=tuple(gamma),
            orders=tuple(tuple(row) for row in orders),
        )
    else:
        error_msg = f'Unknown form {form!r}. Use s-system or gma.'
        raise ValidationError(error_msg)
    return System(components=species, interactions=[rates])


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
    if variables < 2 or len(initial) != variables:  # noqa: PLR2004
        error_msg = f'The Goodwin loop needs at least 2 stages and one initial value per stage, not {variables} stages.'
        raise ValidationError(error_msg)
    labels = [f'x{index}' for index in range(1, variables + 1)]
    stages = [GoodwinStage(label, x=float(value), b=b) for label, value in zip(labels, initial, strict=True)]
    repression = HillRepression('repression', repressor=labels[-1], target=labels[0], a=a, K=K, n=n)
    syntheses = [
        Synthesis(f'synthesis_{target}', source=source, target=target, k=k)
        for source, target in itertools.pairwise(labels)
    ]
    return System(components=stages, interactions=[repression, *syntheses])

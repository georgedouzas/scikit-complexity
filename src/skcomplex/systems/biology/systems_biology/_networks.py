"""Cellular oscillators, glycolytic oscillations, and power-law rate equations."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Mapping, Sequence
from math import prod
from types import MappingProxyType, SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import SR, Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import DeclaredElement, Element
from ....core.components.interactions import BaseInteraction, Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.spaces import BaseSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.errors import ValidationError
from ....core.specs import DeclaredVariableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ...general.reactions import ReactionInteraction
from ..common.kinetics import MichaelisMentenInteraction, RepressionInteraction, SpeciesElement

CONCENTRATIONS = MappingProxyType(
    {
        'ATP': 4.49064,
        'ADP': 0.108367,
        'AMP': 0.00261149,
        'GLC': 0.0112817,
        'F6P': 0.65939,
        'FBP': 0.00770135,
        'GAP': 0.00190919,
        'NAD': 3.62057,
        'NADH': 0.616118,
        'DPG': 0.299109,
        'PEP': 0.0021125,
        'PYR': 0.00422702,
        'ACA': 0.0738334,
        'EtOH': 0.33981,
        'P': 0.0,
    },
)
CONSTANTS = MappingProxyType(
    {
        'V1': 0.5,
        'K1GLC': 0.1,
        'K1ATP': 0.063,
        'V2': 1.5,
        'K2': 0.0016,
        'k2': 0.017,
        'K2ATP': 0.01,
        'k3f': 1.0,
        'k3b': 50.0,
        'V4': 10.0,
        'K4GAP': 1.0,
        'K4NAD': 1.0,
        'k5f': 1.0,
        'k5b': 0.5,
        'V6': 10.0,
        'K6PEP': 0.2,
        'K6ADP': 0.3,
        'V7': 2.0,
        'K7PYR': 0.3,
        'k8f': 1.0,
        'k8b': 0.000143,
        'k9f': 10.0,
        'k9b': 10.0,
        'k10': 0.05,
        'flow': 0.011,
    },
)
FEEDS = MappingProxyType({'ATP': 3.5, 'ADP': 1.1, 'NADH': 0.24, 'NAD': 4.0, 'GLC': 50.0})
RAMAN = 'Raman 2021'
PATHWAY_ALPHA = (2.0, 1.0, 1.0)
PATHWAY_BETA = (1.0, 1.0, 1.0)
PATHWAY_G = ((0.0, 0.0, -0.5), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0))
PATHWAY_H = ((0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5))
PATHWAY_STOICHIOMETRY = ((1.0, -1.0, 0.0, 0.0), (0.0, 1.0, -1.0, 0.0), (0.0, 0.0, 1.0, -1.0))
PATHWAY_GAMMA = (2.0, 1.0, 1.0, 1.0)
PATHWAY_ORDERS = ((0.0, 0.0, -0.5), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5))
MIN_GOODWIN_STAGES = 2


class GlycolysisElement(DeclaredElement):
    """Yeast glycolysis in a flow reactor.

    It has the fifteen metabolites of the model of Nielsen and coworkers.

    The model of Nielsen, Sørensen, Hynne, and Busse (1998), BioModels BIOMD0000000042, follows glucose `GLC` through
    hexokinase, phosphofructokinase with its inhibition by the ratio of `ATP` to `AMP`, aldolase, glyceraldehyde
    phosphate dehydrogenase, phosphoglycerate kinase, pyruvate kinase, pyruvate decarboxylase, alcohol dehydrogenase,
    adenylate kinase, and a glycerol branch to `P`. The reactor is fed with `ATP`, `ADP`, `NADH`, `NAD`, and glucose
    at fixed concentrations and washed out at the rate `flow`. The concentrations, in mM, and the rate constants, in mM
    and per minute, are declared variables named as in the source, and `variables` replaces any of them.

    Args:
        label:
            The label of the cell extract.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        variables:
            The declared states and parameters that replace those of the source, by name, or `None` for none.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import GlycolysisElement
        >>> GlycolysisElement('extract').get_value('V4')
        10.0
    """

    law_classes = (RateLaw,)

    def _declare_variables(self) -> Mapping[str, DeclaredVariableSpec]:
        """Return the concentrations and the rate constants of the source.

        They are updated by `variables`.
        """
        nonnegative = RealSet.closed_open(0, oo)
        declared = {
            name: DeclaredVariableSpec(value, StateSpec(domain=nonnegative)) for name, value in CONCENTRATIONS.items()
        }
        declared |= {
            name: DeclaredVariableSpec(value, ParameterSpec(domain=nonnegative)) for name, value in CONSTANTS.items()
        }
        return {**declared, **(self.variables or {})}

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the metabolites.

        They are the reactions of glycolysis and the exchange with the feed.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        v1 = s.V1 * s.ATP * s.GLC / ((s.K1GLC + s.GLC) * (s.K1ATP + s.ATP))
        inhibition = s.K2 * (1 + s.k2 * (s.ATP / s.AMP) ** 2)
        v2 = s.V2 * s.ATP * s.F6P**2 / ((inhibition + s.F6P**2) * (s.K2ATP + s.ATP))
        v3 = s.k3f * s.FBP - s.k3b * s.GAP**2
        v4 = s.V4 * s.NAD * s.GAP / ((s.K4GAP + s.GAP) * (s.K4NAD + s.NAD))
        v5 = s.k5f * s.DPG * s.ADP - s.k5b * s.PEP * s.ATP
        v6 = s.V6 * s.ADP * s.PEP / ((s.K6PEP + s.PEP) * (s.K6ADP + s.ADP))
        v7 = s.V7 * s.PYR / (s.K7PYR + s.PYR)
        v8 = s.k8f * s.NADH * s.ACA - s.k8b * s.NAD * s.EtOH
        v9 = s.k9f * s.AMP * s.ATP - s.k9b * s.ADP**2
        v10 = s.k10 * s.F6P
        reactions = {
            'ATP': -v1 - v2 + v5 + v6 - v9,
            'ADP': v1 + v2 - v5 - v6 + 2 * v9,
            'AMP': -v9,
            'GLC': -v1,
            'F6P': v1 - v2 - v10,
            'FBP': v2 - v3,
            'GAP': 2 * v3 - v4,
            'NAD': -v4 + v8,
            'NADH': v4 - v8,
            'DPG': v4 - v5,
            'PEP': v5 - v6,
            'PYR': v6 - v7,
            'ACA': v7 - v8,
            'EtOH': v8,
            'P': v10,
        }
        laws: list[BaseLaw] = []
        for name, rate in reactions.items():
            state = getattr(s, name)
            exchange = s.flow * (FEEDS.get(name, 0.0) - state)
            laws.append(RateLaw(state, rate + exchange))
        return laws


class PhosphorylationLoopElement(Element):
    """Three proteins that regulate each other.

    Each is phosphorylated and dephosphorylated by Michaelis-Menten kinetics.

    The states are the unphosphorylated amounts, and each total is conserved. With `u_T = A_T - A`, and so on:

    - `dA/dt = kp1 u_T / (K1 + u_T) - kd1 C A / (K1 + A)`,
    - `dB/dt = kp2 A (B_T - B) / (K2 + B_T - B) - kd2 B / (K2 + B)`,
    - `dC/dt = kp3 B (C_T - C) / (K3 + C_T - C) - kd3 C / (K3 + C)`.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `PhosphorylationOscillatorSystem` read its parameters.

    Args:
        label:
            The label of the loop.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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
        >>> from skcomplex.systems.biology.systems_biology import PhosphorylationLoopElement
        >>> PhosphorylationLoopElement('loop', A=0.5).A
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        A: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        B: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        C: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        kp1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        kd1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        kp2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        kd2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        kp3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        kd3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        K1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.01,
        K2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.01,
        K3: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.01,
        A_T: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        B_T: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        C_T: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.A = A
        self.B = B
        self.C = C
        self.kp1 = kp1
        self.kd1 = kd1
        self.kp2 = kp2
        self.kd2 = kd2
        self.kp3 = kp3
        self.kd3 = kd3
        self.K1 = K1
        self.K2 = K2
        self.K3 = K3
        self.A_T = A_T
        self.B_T = B_T
        self.C_T = C_T


class GeneralizedMassActionInteraction(Interaction):
    """Rates of a biochemical system in the generalized mass action form.

    The form is that of biochemical systems theory.

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

    law_classes = (RateLaw,)

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
        """Return the product `prod_k x_k^f_k` of the values raised to their kinetic orders.

        Zero orders are left out.
        """
        return prod((value**order for value, order in zip(values, orders, strict=True) if order), start=SR(1))

    def _check_shape(self, name: str, rows: int, columns: int) -> None:
        """Raise an error unless a matrix argument has the given shape.

        The shape is the given numbers of rows and columns.
        """
        matrix = getattr(self, name)
        if len(matrix) == rows and all(len(row) == columns for row in matrix):
            return
        error_msg = (
            f'The matrix {name} of {self.label!r} must have {rows} rows of {columns} entries, one row per flux or '
            'species.'
        )
        raise ValidationError(error_msg)

    def _check_matrices(self) -> None:
        """Raise an error for a stoichiometry or kinetic orders of the wrong shape.

        They must match the species and the fluxes.
        """
        self._check_shape('stoichiometry', len(self.species), len(self.gamma))
        self._check_shape('orders', len(self.gamma), len(self.species))

    def _get_fluxes(self, s: SimpleNamespace, values: Sequence[Expression]) -> list[Expression]:
        """Return the power-law fluxes `gamma_j prod_k x_k^f_jk`."""
        return [rate * self._get_power_product(values, row) for rate, row in zip(self.gamma, self.orders, strict=True)]

    def _get_stoichiometry(self) -> Sequence[Sequence[float]]:
        """Return the stoichiometric coefficient of every species in every flux.

        There is one row per species.
        """
        return self.stoichiometry

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Raise an error for matrices that do not match the species and the fluxes.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                The matrices do not have one row per flux or species and one entry per species or flux.
        """
        super()._prepare(context)
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


class SSystemInteraction(GeneralizedMassActionInteraction):
    """Rates of a biochemical system in the S-system form.

    The form is that of biochemical systems theory.

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
        """Return the coefficients of every species in its production and degradation.

        The coefficient is 1 in its production and -1 in its degradation.
        """
        size = len(self.species)
        return [
            [1.0 if flux == 2 * row else -1.0 if flux == 2 * row + 1 else 0.0 for flux in range(2 * size)]
            for row in range(size)
        ]


class PhosphorylationOscillatorSystem(BaseSystem):
    """Three proteins whose phosphorylation cycles regulate each other in a loop.

    The loop oscillates.

    C converts A, A converts B, and B converts C, each reaction with Michaelis-Menten kinetics and each total conserved,
    so the loop is a negative feedback of three steps. The book prints the equations and no parameter values. The
    defaults are a choice: totals of 1, Michaelis constants `K` far below the totals, which make every cycle an
    ultrasensitive switch, and rates that put every switch at half the total of its enzyme. The proteins then switch on
    and off in turn on a limit cycle of period about 4.2. With large `K` the switches become graded and the loop settles
    on a fixed point. The amounts are fractions of the totals. The system has the amounts `loop.A`, `loop.B`, and
    `loop.C`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import PhosphorylationOscillatorSystem
        >>> PhosphorylationOscillatorSystem(K=0.1).prepare().system_.elements[0].K2
        0.1
    """

    field = 'biology/systems_biology'
    topics = ('oscillator', 'phosphorylation', 'Michaelis-Menten', 'negative feedback', 'ode')
    references = (RAMAN, 'Mogilner, Wollman, and Marshall 2006')
    until = 40.0
    dt = 0.05

    def __init__(
        self,
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
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.A = A
        self.B = B
        self.C = C
        self.kp1 = kp1
        self.kd1 = kd1
        self.kp2 = kp2
        self.kd2 = kd2
        self.kp3 = kp3
        self.kd3 = kd3
        self.K = K

    def _build_system(self) -> CompositeSystem:
        """Return the components of three proteins."""
        loop = PhosphorylationLoopElement(
            'loop',
            A=self.A,
            B=self.B,
            C=self.C,
            kp1=self.kp1,
            kd1=self.kd1,
            kp2=self.kp2,
            kd2=self.kd2,
            kp3=self.kp3,
            kd3=self.kd3,
            K1=self.K,
            K2=self.K,
            K3=self.K,
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
        return CompositeSystem(self.label, elements=[loop], interactions=reactions)


class BiochemicalSystemsTheorySystem(BaseSystem):
    """A biochemical system of power-law rates.

    It takes the S-system or the generalized mass action form.

    The S-system form gives every species one production and one degradation, `dx_i/dt = alpha_i prod_j x_j^g_ij -
    beta_i prod_j x_j^h_ij`. The generalized mass action form sums power-law fluxes, `dx_i/dt = sum_j s_ij gamma_j
    prod_k x_k^f_jk`. The book states only the two forms. The default is a choice, an unbranched pathway `X1 -> X2 ->
    X3` whose end product inhibits the supply of `X1`, written in both forms with the same rates, `dX1/dt = 2 X3^-0.5 -
    X1^0.5`, `dX2/dt = X1^0.5 - X2^0.5`, and `dX3/dt = X2^0.5 - X3^0.5`. The steady state of an S-system solves the
    linear system `(g - h) log x = log(beta / alpha)`, here `x = (2, 2, 2)`. The system has the concentrations `X1.x`,
    `X2.x`, and so on.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ValidationError:
            The form is unknown.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import BiochemicalSystemsTheorySystem
        >>> BiochemicalSystemsTheorySystem(form='gma').prepare().system_.interactions[0].gamma
        (2.0, 1.0, 1.0, 1.0)
    """

    field = 'biology/systems_biology'
    topics = ('biochemical systems theory', 'power law', 'S-system', 'generalized mass action', 'ode')
    references = (RAMAN, 'Savageau 1976', 'Voit 2000')
    until = 20.0
    dt = 0.05

    def __init__(
        self,
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
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.x = x
        self.form = form
        self.alpha = alpha
        self.beta = beta
        self.g = g
        self.h = h
        self.stoichiometry = stoichiometry
        self.gamma = gamma
        self.orders = orders

    def _build_system(self) -> CompositeSystem:
        """Return the components of a biochemical system of power-law rates."""
        species = [SpeciesElement(f'X{index}', x=float(value)) for index, value in enumerate(self.x, start=1)]
        paths = [item.label for item in species]
        if self.form == 's-system':
            rates: BaseInteraction = SSystemInteraction(
                'rates',
                species=paths,
                alpha=[float(value) for value in self.alpha],
                beta=[float(value) for value in self.beta],
                g=tuple(tuple(row) for row in self.g),
                h=tuple(tuple(row) for row in self.h),
            )
        elif self.form == 'gma':
            rates = GeneralizedMassActionInteraction(
                'rates',
                species=paths,
                stoichiometry=tuple(tuple(row) for row in self.stoichiometry),
                gamma=tuple(self.gamma),
                orders=tuple(tuple(row) for row in self.orders),
            )
        else:
            error_msg = f'Unknown form {self.form!r}. Use s-system or gma.'
            raise ValidationError(error_msg)
        return CompositeSystem(self.label, elements=species, interactions=[rates])


class GoodwinOscillatorSystem(BaseSystem):
    """The Goodwin oscillator, a chain of stages.

    The last stage represses the synthesis of the first.

    The first stage, an mRNA, is synthesized at `a / (1 + (x_N / K)^n)` and every later stage at `k` times the stage
    before it, and every stage decays at rate `b`: `dx_1/dt = a / (1 + (x_N / K)^n) - b x_1` and `dx_i/dt = k x_(i-1) -
    b x_i`. The book points to BioModels and prints no equations, so the system takes the standard form of Goodwin with
    the degradation of Griffith. With equal degradation rates the fixed point loses stability when the loop gain passes
    `sec(pi / N)^N`, 8 for three stages. When `a` is large the repressor saturates and the gain is close to `n`, so
    three stages oscillate for Hill coefficients above about 8 and settle for lower ones. The concentrations and the
    time are dimensionless. The system has the concentrations `x1.x` to `x<variables>.x`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ValidationError:
            There are fewer than two stages, or the initial concentrations do not match them.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import GoodwinOscillatorSystem
        >>> [stage.label for stage in GoodwinOscillatorSystem(4).prepare().system_.elements]
        ['x1', 'x2', 'x3', 'x4']
    """

    field = 'biology/systems_biology'
    topics = ('oscillator', 'negative feedback', 'Hill function', 'gene regulation', 'ode')
    references = (RAMAN, 'Goodwin 1965', 'Griffith 1968')
    until = 60.0
    dt = 0.1

    def __init__(
        self,
        variables: int = 3,
        x: Sequence[float] | None = None,
        *,
        a: float = 100.0,
        b: float = 1.0,
        K: float = 1.0,
        n: float = 12.0,
        k: float = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.variables = variables
        self.x = x
        self.a = a
        self.b = b
        self.K = K
        self.n = n
        self.k = k

    def _build_system(self) -> CompositeSystem:
        """Return the components of the Goodwin oscillator."""
        initial = [0.1 * index for index in range(1, self.variables + 1)] if self.x is None else list(self.x)
        if self.variables < MIN_GOODWIN_STAGES or len(initial) != self.variables:
            error_msg = (
                f'The Goodwin loop needs at least 2 stages and one initial value per stage, not {self.variables} '
                'stages.'
            )
            raise ValidationError(error_msg)
        labels = [f'x{index}' for index in range(1, self.variables + 1)]
        stages = [SpeciesElement(label, x=float(value)) for label, value in zip(labels, initial, strict=True)]
        repression = RepressionInteraction(
            'repression',
            products={(labels[0], 'x'): 1},
            modifiers={(labels[-1], 'x'): 1},
            alpha=self.a,
            K=self.K,
            n=self.n,
        )
        syntheses = [
            ReactionInteraction(
                f'synthesis_{target}',
                products={(target, 'x'): 1},
                modifiers={(source, 'x'): 1},
                k=self.k,
            )
            for source, target in itertools.pairwise(labels)
        ]
        degradations = [
            ReactionInteraction(f'degradation_{label}', reactants={(label, 'x'): 1}, k=self.b) for label in labels
        ]
        return CompositeSystem(self.label, elements=stages, interactions=[repression, *syntheses, *degradations])


class GlycolysisOscillationsSystem(BaseSystem):
    """The glycolysis of a yeast extract in a flow reactor.

    Its NADH oscillates with a period of minutes.

    The model of Nielsen et al. (1998), BioModels BIOMD0000000042, has fifteen metabolites, the inhibition of
    phosphofructokinase by `ATP` and its activation by `AMP`, and the exchange with the feed of the reactor at the rate
    `flow`. With the published parameters the reactor settles on sustained oscillations of `NADH` and of all the other
    metabolites. Raman (2021, exercise 5.7) varies `V4`, `K4GAP`, `K4NAD`, `k8f`, and `k8b` by half to study their
    effect on `NAD` and `NADH`. The system has the concentrations `extract.<metabolite>`, such as `extract.NADH`, in mM.

    Args:
        flow:
            The flow rate of the reactor, per minute.

        V4:
            The largest rate of glyceraldehyde phosphate dehydrogenase, in mM per minute.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import GlycolysisOscillationsSystem
        >>> GlycolysisOscillationsSystem(flow=0.02).prepare().system_.elements[0].get_value('flow')
        0.02
    """

    field = 'biology/systems_biology'
    topics = ('glycolysis', 'metabolic oscillations', 'nadh', 'biomodels', 'ode')
    references = ('Nielsen et al. 1998', 'Raman 2021')
    until = 300.0
    dt = 0.5

    def __init__(
        self,
        *,
        flow: float = 0.011,
        V4: float = 10.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.flow = flow
        self.V4 = V4

    def _build_system(self) -> CompositeSystem:
        """Return the components of the glycolysis of a yeast extract in a flow reactor."""
        variables = {
            'flow': DeclaredVariableSpec(self.flow, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'V4': DeclaredVariableSpec(self.V4, ParameterSpec(domain=RealSet.closed_open(0, oo))),
        }
        return CompositeSystem(self.label, elements=[GlycolysisElement('extract', variables=variables)])

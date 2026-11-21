"""Microbial communities of species that grow and interact in pairs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import RateLaw
from ....core.components.laws.utils import total
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.errors import ValidationError
from ....core.systems import BaseSystem, System
from ....core.types import Role

COMMUNITY_GROWTH = (1.0, 0.8, 0.6)
COMMUNITY_INTERACTIONS = ((-1.0, -0.3, 0.2), (-0.2, -1.0, -0.3), (0.1, -0.2, -1.0))


class MicrobeElement(Element):
    """Species of a microbial community.

    It has an abundance and an intrinsic growth rate.

    Args:
        label:
            The label of the species or of the group of species.

        x:
            The initial abundance.

        b:
            The intrinsic growth rate.

        size:
            The number of species, or `None` for one.

    Examples:
        >>> from skcomplex.systems.biology.ecology import MicrobeElement
        >>> MicrobeElement('species', x=0.1, b=0.8).b
        0.8
    """

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[float | npt.ArrayLike, RealSet.closed_open(0, oo)] = 0.1,
        b: float | npt.ArrayLike = 1.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.x = x
        self.b = b


class MicrobialGrowthInteraction(Interaction):
    """Growth of every species of a group at its own intrinsic rate, `dx_i/dt += b_i x_i`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of species, whose members hold the abundance `x` and the growth rate `b`.

    Examples:
        >>> from skcomplex.systems.biology.ecology import MicrobialGrowthInteraction
        >>> MicrobialGrowthInteraction('growth', target='species').target
        'species'
    """

    law_classes = (RateLaw,)

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the growth of the abundance of every species.

        Args:
            context:
                The namespace of the species, the time `t`, and the parameters.
        """
        super()._prepare(context)
        self.laws_ = [RateLaw(context.target.x, context.target.b * context.target.x)]


class GeneralizedLotkaVolterraInteraction(Interaction):
    """Interactions of the species of a group, `dx_i/dt += x_i sum_j a_ij x_j`.

    They act through an interaction matrix `a`.

    Every species `i` couples to every species `j` of the group, itself included, through the term `x_i x_j` weighted
    by the entry `a_ij` of the interaction matrix, a parameter with one row and one column per species, so a schedule or
    an experiment changes it. The diagonal entries are the interactions of a species with itself, negative for
    self-limitation. With the growth `b_i x_i` as a rate, it gives the generalized Lotka-Volterra equations
    `dx_i/dt = x_i (b_i + sum_j a_ij x_j)`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of species.

        a:
            The interaction matrix, where `a_ij` is the effect of species `j` on the growth of species `i`, or a number
            for every entry.

        state:
            The name of the state that holds the abundance of a species.

    Examples:
        >>> from skcomplex.systems.biology.ecology import GeneralizedLotkaVolterraInteraction
        >>> matrix = ((-1.0, 0.5), (0.5, -1.0))
        >>> GeneralizedLotkaVolterraInteraction('glv', target='species', a=matrix).a
        ((-1.0, 0.5), (0.5, -1.0))
    """

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        a: object = -1.0,
        state: str = 'x',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.a = a
        self.state = state

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the coupling of the abundance of every species to the abundances of the group.

        Args:
            context:
                The namespace of the species, the time `t`, and the matrix.
        """
        super()._prepare(context)
        abundance = getattr(context.target, self.state)
        self.laws_ = [RateLaw(abundance, abundance * total(abundance, weights=context.a))]


class GeneralizedLotkaVolterraSystem(BaseSystem):
    """The generalized Lotka-Volterra system of a microbial community.

    It follows `dx_i/dt = x_i (b_i + sum_j a_ij x_j)`.

    Every species grows at its intrinsic rate `b_i` and its growth changes with the abundance of every species through
    the pairwise interaction `a_ij`. A feasible interior fixed point solves `b + a x = 0`. The book gives no parameter
    values. The default is a choice, a community of three self-limiting species with weak competition and one weak
    mutualism, whose fixed point `x = -a^-1 b` is positive and stable, so the species coexist. Abundances and time are
    in arbitrary units. The system has the abundances `species.x` of the members of the group `species`, the growth
    `growth`, and the interactions `glv`, whose matrix is the parameter `glv.a`.

    Args:
        x:
            The initial abundances, or `None` for 0.1 each.

        b:
            The intrinsic growth rates.

        a:
            The interaction matrix, one row per species.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ValidationError:
            The growth rates, the matrix, and the abundances do not have one entry per species.

    Examples:
        >>> from skcomplex.systems.biology.ecology import GeneralizedLotkaVolterraSystem
        >>> GeneralizedLotkaVolterraSystem().prepare().system_.elements[0].size
        3
    """

    field = 'biology/ecology'
    topics = ('microbial community', 'Lotka-Volterra', 'interaction matrix', 'ode')
    references = ('Raman 2021', 'Stein et al. 2013')
    until = 30.0
    dt = 0.1

    def __init__(
        self,
        x: Sequence[float] | None = None,
        *,
        b: Sequence[float] = COMMUNITY_GROWTH,
        a: Sequence[Sequence[float]] = COMMUNITY_INTERACTIONS,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.b = b
        self.a = a

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the generalized Lotka-Volterra microbial community.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        size = len(self.b)
        abundances = [0.1] * size if self.x is None else [float(value) for value in self.x]
        if len(abundances) != size or len(self.a) != size or any(len(row) != size for row in self.a):
            error_msg = (
                f'The abundances and the interaction matrix must have one entry per species, {size} growth rates.'
            )
            raise ValidationError(error_msg)
        species = MicrobeElement('species', x=abundances, b=[float(value) for value in self.b], size=size)
        matrix = np.asarray(self.a, dtype=float)
        self.system_ = System(
            self.label,
            elements=[species],
            interactions=[
                MicrobialGrowthInteraction('growth', target='species'),
                GeneralizedLotkaVolterraInteraction('glv', target='species', a=matrix),
            ],
        )

"""Microbial communities of species that grow and interact in pairs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import CouplingInteraction, RateInteraction
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.errors import ValidationError
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec

COMMUNITY_GROWTH = (1.0, 0.8, 0.6)

COMMUNITY_INTERACTIONS = ((-1.0, -0.3, 0.2), (-0.2, -1.0, -0.3), (0.1, -0.2, -1.0))


class MicrobeElement(Element):
    """Species of a microbial community.

    It has an abundance and an intrinsic growth rate.

    Args:
        label:
            The label of the species or of the group of species.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial abundance.

        b:
            The intrinsic growth rate.

    Examples:
        >>> from skcomplex.systems.biology.ecology import MicrobeElement
        >>> MicrobeElement('species', x=0.1, b=0.8).b
        0.8
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        b: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.b = b


class GeneralizedLotkaVolterraInteraction(CouplingInteraction):
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
        target: Annotated[str, RoleSpec()],
        a: Annotated[object, ParameterSpec(shape=('target', 'target'))] = -1.0,
        state: str = 'x',
    ) -> None:
        super().__init__(
            label,
            target=target,
            read=state,
            write=state,
            function=self._product,
            mode='group',
            weights='a',
            include_self=True,
        )
        self.a = a
        self.state = state

    def _product(self, namespace: SimpleNamespace) -> object:
        """Return the product `x_i x_j` of the abundances of two species.

        The species are the member `i` and the other species `j`.
        """
        return getattr(namespace.member, self.state) * getattr(namespace.other, self.state)


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

    Raises:
        ValidationError:
            The growth rates, the matrix, and the abundances do not have one entry per species.

    Examples:
        >>> from skcomplex.systems.biology.ecology import GeneralizedLotkaVolterraSystem
        >>> GeneralizedLotkaVolterraSystem().prepare().system_.elements[0].prepare().size_
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
    ) -> None:
        super().__init__(label)
        self.x = x
        self.b = b
        self.a = a

    @staticmethod
    def _growth(namespace: SimpleNamespace) -> object:
        """Return the intrinsic growth `b x` of a species."""
        return namespace.target.b * namespace.target.x

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the generalized Lotka-Volterra microbial community."""
        size = len(self.b)
        abundances = [0.1] * size if self.x is None else [float(value) for value in self.x]
        if len(abundances) != size or len(self.a) != size or any(len(row) != size for row in self.a):
            error_msg = (
                f'The abundances and the interaction matrix must have one entry per species, {size} growth rates.'
            )
            raise ValidationError(error_msg)
        species = GroupElement(
            'species',
            element=MicrobeElement,
            size=size,
            values={'x': abundances, 'b': [float(value) for value in self.b]},
        )
        matrix = tuple(tuple(float(value) for value in row) for row in self.a)
        return CompositeSystem(
            self.label,
            elements=[species],
            interactions=[
                RateInteraction('growth', target='species', rate=self._growth),
                GeneralizedLotkaVolterraInteraction('glv', target='species', a=matrix),
            ],
        )

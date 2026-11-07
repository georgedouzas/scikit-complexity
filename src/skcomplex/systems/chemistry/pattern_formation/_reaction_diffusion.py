"""Pattern formation systems: reaction-diffusion fields whose sites react and whose states diffuse."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression

from ....core.components.elements import Element, FieldElement
from ....core.components.interactions import DiffusionInteraction, ReactionInteraction
from ....core.components.spaces import LatticeSpace
from ....core.components.system import System
from ....core.engine.solvers import RK4Solver
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ....core.types import Array
from ..._catalog import register_system
from ..common.kinetics import BrusselatorReactionElement, CubicAutocatalysisInteraction


class BrusselatorElement(BrusselatorReactionElement):
    """Site of the Brusselator, the reactor of the Brusselator with the diffusion coefficients of its intermediates.

    The four reactions of `brusselator` run at every site, and its diffusions read `Du` and `Dv`.

    Args:
        label:
            The label of the site or the field.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        u:
            The initial concentration of `u`.

        v:
            The initial concentration of `v`.

        a:
            The feed of `u`.

        b:
            The conversion of `u` into `v`.

        Du:
            The diffusion of `u`.

        Dv:
            The diffusion of `v`.

    Examples:
        >>> from skcomplex.systems.chemistry.pattern_formation import BrusselatorElement
        >>> BrusselatorElement('cell').Dv
        8.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: float = 1.0,
        v: float = 3.0,
        a: float = 1.0,
        b: float = 3.0,
        Du: Annotated[object, ParameterSpec()] = 1.0,
        Dv: Annotated[object, ParameterSpec()] = 8.0,
    ) -> None:
        super().__init__(label, visual=visual, u=u, v=v, a=a, b=b)
        self.Du = Du
        self.Dv = Dv


class GrayScottElement(Element):
    """Site of the Gray-Scott system, the substrate `u` and the activator `v`.

    The reactions of `gray_scott` at every site, the cubic autocatalysis of the activator on the substrate, the feed
    `F` of the substrate, and the removal `F + k` of the activator, read `F` and `k`, and its diffusions read `Du` and
    `Dv`.

    Args:
        label:
            The label of the site or the field.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        u:
            The initial substrate.

        v:
            The initial activator.

        Du:
            The diffusion of the substrate.

        Dv:
            The diffusion of the activator.

        F:
            The feed rate.

        k:
            The kill rate.

    Examples:
        >>> from skcomplex.systems.chemistry.pattern_formation import GrayScottElement
        >>> GrayScottElement('cell').k
        0.065
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[object, StateSpec()] = 1.0,
        v: Annotated[object, StateSpec()] = 0.0,
        Du: Annotated[object, ParameterSpec()] = 0.16,
        Dv: Annotated[object, ParameterSpec()] = 0.08,
        F: Annotated[object, ParameterSpec()] = 0.035,
        k: Annotated[object, ParameterSpec()] = 0.065,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u
        self.v = v
        self.Du = Du
        self.Dv = Dv
        self.F = F
        self.k = k


class SchnakenbergElement(Element):
    """Site of the Schnakenberg system, the activator `u` and the inhibitor `v`.

    The reactions of `schnakenberg` at every site, the productions `a` of `u` and `b` of `v`, the decay of `u`, and
    the autocatalysis `2U + V -> 3U`, read `a` and `b`, and its diffusions read `Du` and `Dv`.

    Args:
        label:
            The label of the site or the field.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        u:
            The initial activator.

        v:
            The initial inhibitor.

        a:
            The production of `u`.

        b:
            The production of `v`.

        Du:
            The diffusion of the activator.

        Dv:
            The diffusion of the inhibitor.

    Examples:
        >>> from skcomplex.systems.chemistry.pattern_formation import SchnakenbergElement
        >>> SchnakenbergElement('cell').Dv
        40.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[object, StateSpec()] = 1.0,
        v: Annotated[object, StateSpec()] = 0.9,
        a: Annotated[object, ParameterSpec()] = 0.1,
        b: Annotated[object, ParameterSpec()] = 0.9,
        Du: Annotated[object, ParameterSpec()] = 1.0,
        Dv: Annotated[object, ParameterSpec()] = 40.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u
        self.v = v
        self.a = a
        self.b = b
        self.Du = Du
        self.Dv = Dv


def _get_u_diffusion(s: SimpleNamespace) -> Expression:
    """Return the diffusion coefficient `Du` of the sites."""
    return s.target.Du


def _get_v_diffusion(s: SimpleNamespace) -> Expression:
    """Return the diffusion coefficient `Dv` of the sites."""
    return s.target.Dv


def _get_diffusions() -> list[DiffusionInteraction]:
    """Return the diffusions of `u` and `v` on the field `cells`, at the coefficients `Du` and `Dv` of its sites."""
    return [
        DiffusionInteraction('u_diffusion', target='cells', state='u', coefficient=_get_u_diffusion),
        DiffusionInteraction('v_diffusion', target='cells', state='v', coefficient=_get_v_diffusion),
    ]


def _get_gray_scott_reactions() -> list[ReactionInteraction]:
    """Return the reactions of the Gray-Scott system at every site of `cells`, the kinetics of `cubic_autocatalysis`."""
    return [
        CubicAutocatalysisInteraction('autocatalysis', target='cells', substrate='u', autocatalyst='v'),
        ReactionInteraction('feed', target='cells', products={'u': 1}, modifiers={'F': 1}),
        ReactionInteraction('washout', target='cells', reactants={'u': 1}, modifiers={'F': 1}),
        ReactionInteraction('outflow', target='cells', reactants={'v': 1}, modifiers={'F': 1}),
        ReactionInteraction('decay', target='cells', reactants={'v': 1}, modifiers={'k': 1}),
    ]


def _get_brusselator_reactions() -> list[ReactionInteraction]:
    """Return the reactions of the Brusselator at every site of `cells`, the kinetics of `brusselator_ode`."""
    return [
        ReactionInteraction('feed', target='cells', products={'u': 1}, modifiers={'a': 1}),
        ReactionInteraction('conversion', target='cells', reactants={'u': 1}, products={'v': 1}, modifiers={'b': 1}),
        CubicAutocatalysisInteraction('autocatalysis', target='cells', substrate='v', autocatalyst='u'),
        ReactionInteraction('decay', target='cells', reactants={'u': 1}),
    ]


def _get_schnakenberg_reactions() -> list[ReactionInteraction]:
    """Return the reactions of the Schnakenberg system at every site of `cells`."""
    return [
        ReactionInteraction('production_u', target='cells', products={'u': 1}, modifiers={'a': 1}),
        ReactionInteraction('decay', target='cells', reactants={'u': 1}),
        CubicAutocatalysisInteraction('autocatalysis', target='cells', substrate='v', autocatalyst='u'),
        ReactionInteraction('production_v', target='cells', products={'v': 1}, modifiers={'b': 1}),
    ]


def _noise(shape: tuple[int, ...], scale: float, random_state: int | np.random.Generator | None) -> Array:
    """Return small uniform noise of a shape."""
    return scale * np.random.default_rng(random_state).uniform(-1.0, 1.0, size=shape)


@register_system(
    field='chemistry/pattern_formation',
    tags=('field', 'reaction-diffusion', 'self-replication', 'pde'),
    references=('Gray and Scott 1984', 'Pearson 1993'),
    defaults={'until': 2000.0, 'dt': 20.0, 'solver': RK4Solver(dt=1.0)},
)
def gray_scott(
    shape: Sequence[int] = (128, 128),
    *,
    Du: float = 0.16,
    Dv: float = 0.08,
    F: float = 0.035,
    k: float = 0.065,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Gray-Scott reaction-diffusion system, seeded in its middle, which grows spots that replicate.

    Args:
        shape:
            The number of rows and columns of the periodic lattice.

        Du:
            The diffusion of the substrate `u`.

        Dv:
            The diffusion of the activator `v`.

        F:
            The feed rate.

        k:
            The kill rate.

        random_state:
            The seed or generator of the noise on the seed.

    Returns:
        system:
            The system, with the substrate `cells.u` and the activator `cells.v`.

    Examples:
        >>> from skcomplex.systems.chemistry.pattern_formation import gray_scott
        >>> gray_scott((32, 32), F=0.04).elements[0].values['F']
        0.04
    """
    shape = tuple(shape)
    u, v = np.ones(shape), np.zeros(shape)
    middle = tuple(slice(size // 2 - size // 10, size // 2 + size // 10) for size in shape)
    u[middle], v[middle] = 0.5, 0.25
    noise = _noise(shape, 0.01, random_state)
    values = {'u': u + noise, 'v': v + np.abs(noise), 'Du': Du, 'Dv': Dv, 'F': F, 'k': k}
    return System(
        elements=[FieldElement('cells', element=GrayScottElement, space=LatticeSpace(shape), values=values)],
        interactions=[*_get_gray_scott_reactions(), *_get_diffusions()],
    )


@register_system(
    field='chemistry/pattern_formation',
    tags=('field', 'reaction-diffusion', 'Turing', 'oscillation', 'pde'),
    references=('Prigogine and Lefever 1968', 'Gros 2015'),
    defaults={'until': 100.0, 'dt': 1.0, 'solver': RK4Solver(dt=0.01)},
)
def brusselator(
    shape: Sequence[int] = (64, 64),
    *,
    a: float = 1.0,
    b: float = 3.0,
    Du: float = 1.0,
    Dv: float = 8.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Brusselator reaction-diffusion system from small noise around its steady state `(a, b / a)`.

    Args:
        shape:
            The number of rows and columns of the periodic lattice.

        a:
            The feed of `u`.

        b:
            The conversion of `u` into `v`.

        Du:
            The diffusion of `u`.

        Dv:
            The diffusion of `v`.

        random_state:
            The seed or generator of the noise.

    Returns:
        system:
            The system, with the concentrations `cells.u` and `cells.v`.

    Examples:
        >>> from skcomplex.systems.chemistry.pattern_formation import brusselator
        >>> brusselator((8, 8), b=2.5).elements[0].values['b']
        2.5
    """
    shape = tuple(shape)
    values = {
        'u': a + _noise(shape, 0.01, random_state),
        'v': np.full(shape, b / a),
        'a': a,
        'b': b,
        'Du': Du,
        'Dv': Dv,
    }
    return System(
        elements=[FieldElement('cells', element=BrusselatorElement, space=LatticeSpace(shape), values=values)],
        interactions=[*_get_brusselator_reactions(), *_get_diffusions()],
    )


@register_system(
    field='chemistry/pattern_formation',
    tags=('field', 'reaction-diffusion', 'Turing', 'pde'),
    references=('Schnakenberg 1979', 'Turing 1952', 'Murray 2003'),
    defaults={'until': 200.0, 'dt': 5.0, 'solver': RK4Solver(dt=0.005)},
)
def schnakenberg(
    shape: Sequence[int] = (64, 64),
    *,
    a: float = 0.1,
    b: float = 0.9,
    Du: float = 1.0,
    Dv: float = 40.0,
    spacing: float = 1.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Schnakenberg reaction-diffusion system, which forms Turing patterns from noise when `Dv >> Du`.

    The steady state `u = a + b`, `v = b / (a + b)^2` is stable without diffusion and unstable to perturbations of a
    band of wavelengths with it.

    Args:
        shape:
            The number of sites along every dimension of the periodic lattice.

        a:
            The production of `u`.

        b:
            The production of `v`.

        Du:
            The diffusion of the activator `u`.

        Dv:
            The diffusion of the inhibitor `v`.

        spacing:
            The distance between sites.

        random_state:
            The seed or generator of the noise.

    Returns:
        system:
            The system, with the concentrations `cells.u` and `cells.v`.

    Examples:
        >>> from skcomplex.systems.chemistry.pattern_formation import schnakenberg
        >>> schnakenberg((8, 8)).elements[0].values['Dv']
        40.0
    """
    shape = tuple(shape)
    steady_u, steady_v = a + b, b / (a + b) ** 2
    values = {
        'u': steady_u * (1 + _noise(shape, 0.01, random_state)),
        'v': np.full(shape, steady_v),
        'a': a,
        'b': b,
        'Du': Du,
        'Dv': Dv,
    }
    return System(
        elements=[
            FieldElement(
                'cells',
                element=SchnakenbergElement,
                space=LatticeSpace(shape, spacing=spacing),
                values=values,
            ),
        ],
        interactions=[*_get_schnakenberg_reactions(), *_get_diffusions()],
    )

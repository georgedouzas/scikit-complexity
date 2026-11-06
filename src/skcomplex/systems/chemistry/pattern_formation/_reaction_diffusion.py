"""Pattern formation systems: reaction-diffusion fields whose sites react and whose states diffuse."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import Expression

from ....core.components.elements import FieldElement
from ....core.components.interactions import DiffusionInteraction, ReactionInteraction
from ....core.components.spaces import LatticeSpace
from ....core.components.system import System
from ....core.engine.solvers import RK4Solver
from ....core.types import Array
from ..._catalog import register_system
from ..common.kinetics import CubicAutocatalysisInteraction
from ._brusselator import BrusselatorElement
from ._gray_scott import GrayScottElement
from ._schnakenberg import SchnakenbergElement


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

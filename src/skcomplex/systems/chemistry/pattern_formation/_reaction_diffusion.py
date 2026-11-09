"""Reaction-diffusion fields whose sites react and whose states diffuse."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression

from ....core.components.elements import FieldElement
from ....core.components.interactions import RateInteraction
from ....core.components.laws.utils import laplacian
from ....core.components.spaces import LatticeSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.solvers import RK4Solver
from ....core.specs import DeclaredVariableSpec, ParameterSpec, StateSpec, VisualSpec
from ....core.types import Array
from ...general.dynamics import ReactionDiffusionSystem
from ...general.reactions import ReactionInteraction
from ..common.kinetics import BrusselatorReactionElement, CubicAutocatalysisInteraction


class BrusselatorElement(BrusselatorReactionElement):
    """Site of the Brusselator, its reactor with the diffusion coefficients.

    The coefficients are those of its intermediates.

    The four reactions of `BrusselatorSystem` run at every site, and its diffusions read `Du` and `Dv`.

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


def _get_u_diffusion(s: SimpleNamespace) -> Expression:
    """Return the diffusion coefficient `Du` of the sites."""
    return s.target.Du


def _get_v_diffusion(s: SimpleNamespace) -> Expression:
    """Return the diffusion coefficient `Dv` of the sites."""
    return s.target.Dv


def _get_diffusions() -> list[RateInteraction]:
    """Return the diffusions of `u` and `v` on the field `cells`.

    They run at the coefficients `Du` and `Dv` of its sites.
    """
    return [
        RateInteraction(
            'u_diffusion',
            target='cells',
            state='u',
            rate=lambda s: _get_u_diffusion(s) * laplacian(s.target.u),
        ),
        RateInteraction(
            'v_diffusion',
            target='cells',
            state='v',
            rate=lambda s: _get_v_diffusion(s) * laplacian(s.target.v),
        ),
    ]


def _noise(shape: tuple[int, ...], scale: float, random_state: int | np.random.Generator | None) -> Array:
    """Return small uniform noise of a shape."""
    return scale * np.random.default_rng(random_state).uniform(-1.0, 1.0, size=shape)


class GrayScottSystem(ReactionDiffusionSystem):
    """The Gray-Scott reaction-diffusion system, seeded in its middle.

    It grows spots that replicate.

    Every site of a periodic lattice runs the kinetics of `CubicAutocatalysisSystem`, the cubic autocatalysis of the
    activator on the substrate, the feed `F` of the substrate, and the removal `F + k` of the activator, and both
    diffuse. The substrate is `cells.u` and the activator `cells.v`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.chemistry.pattern_formation import GrayScottSystem
        >>> GrayScottSystem((32, 32), F=0.04).prepare().system_.elements[0].get_declaring_part().get_value('F')
        0.04
    """

    field = 'chemistry/pattern_formation'
    topics = ('field', 'reaction-diffusion', 'self-replication', 'pde')
    references = ('Gray and Scott 1984', 'Pearson 1993')
    until = 2000.0
    dt = 20.0
    solver = RK4Solver(dt=1.0)

    def __init__(
        self,
        shape: Sequence[int] = (128, 128),
        *,
        Du: float = 0.16,
        Dv: float = 0.08,
        F: float = 0.035,
        k: float = 0.065,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.shape = shape
        self.Du = Du
        self.Dv = Dv
        self.F = F
        self.k = k
        self.random_state = random_state

    @staticmethod
    def _get_reactions() -> list[ReactionInteraction]:
        """Return the reactions of the Gray-Scott system at every site of `cells`."""
        return [
            CubicAutocatalysisInteraction('autocatalysis', target='cells', substrate='u', autocatalyst='v'),
            ReactionInteraction('feed', target='cells', products={'u': 1}, modifiers={'F': 1}),
            ReactionInteraction('washout', target='cells', reactants={'u': 1}, modifiers={'F': 1}),
            ReactionInteraction('outflow', target='cells', reactants={'v': 1}, modifiers={'F': 1}),
            ReactionInteraction('decay', target='cells', reactants={'v': 1}, modifiers={'k': 1}),
        ]

    def _get_element_label(self) -> str:
        """Return the label of the field of the sites."""
        return 'cells'

    def _get_space(self) -> LatticeSpace:
        """Return the periodic lattice of the shape."""
        return LatticeSpace(tuple(self.shape))

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the fields seeded in their middle, and the parameters."""
        shape = tuple(self.shape)
        u, v = np.ones(shape), np.zeros(shape)
        middle = tuple(slice(size // 2 - size // 10, size // 2 + size // 10) for size in shape)
        u[middle], v[middle] = 0.5, 0.25
        noise = _noise(shape, 0.01, self.random_state)
        return {
            'u': DeclaredVariableSpec(u + noise, StateSpec()),
            'v': DeclaredVariableSpec(v + np.abs(noise), StateSpec()),
            'Du': DeclaredVariableSpec(self.Du, ParameterSpec()),
            'Dv': DeclaredVariableSpec(self.Dv, ParameterSpec()),
            'F': DeclaredVariableSpec(self.F, ParameterSpec()),
            'k': DeclaredVariableSpec(self.k, ParameterSpec()),
        }

    def _get_diffusivities(self) -> dict[str, Callable[[SimpleNamespace], object]]:
        """Return the diffusivities `Du` and `Dv` of the sites."""
        return {'u': _get_u_diffusion, 'v': _get_v_diffusion}

    def _get_equation_interactions(self) -> list[ReactionInteraction]:
        """Return the reactions at every site."""
        return self._get_reactions()


class BrusselatorSystem(BaseSystem):
    """The Brusselator reaction-diffusion system from noise around its steady state.

    The noise is small and the steady state is `(a, b / a)`.

    Every site of a periodic lattice runs the kinetics of `BrusselatorODESystem`, and both intermediates diffuse. The
    concentrations are `cells.u` and `cells.v`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.chemistry.pattern_formation import BrusselatorSystem
        >>> BrusselatorSystem((8, 8), b=2.5).prepare().system_.elements[0].values['b']
        2.5
    """

    field = 'chemistry/pattern_formation'
    topics = ('field', 'reaction-diffusion', 'Turing', 'oscillation', 'pde')
    references = ('Prigogine and Lefever 1968', 'Gros 2015')
    until = 100.0
    dt = 1.0
    solver = RK4Solver(dt=0.01)

    def __init__(
        self,
        shape: Sequence[int] = (64, 64),
        *,
        a: float = 1.0,
        b: float = 3.0,
        Du: float = 1.0,
        Dv: float = 8.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.a = a
        self.b = b
        self.Du = Du
        self.Dv = Dv
        self.random_state = random_state

    @staticmethod
    def _get_reactions() -> list[ReactionInteraction]:
        """Return the reactions of the Brusselator at every site of `cells`."""
        return [
            ReactionInteraction('feed', target='cells', products={'u': 1}, modifiers={'a': 1}),
            ReactionInteraction(
                'conversion',
                target='cells',
                reactants={'u': 1},
                products={'v': 1},
                modifiers={'b': 1},
            ),
            CubicAutocatalysisInteraction('autocatalysis', target='cells', substrate='v', autocatalyst='u'),
            ReactionInteraction('decay', target='cells', reactants={'u': 1}),
        ]

    def _build_system(self) -> CompositeSystem:
        """Return the field near its steady state, its reactions, and its diffusions."""
        shape = tuple(self.shape)
        values = {
            'u': self.a + _noise(shape, 0.01, self.random_state),
            'v': np.full(shape, self.b / self.a),
            'a': self.a,
            'b': self.b,
            'Du': self.Du,
            'Dv': self.Dv,
        }
        return CompositeSystem(
            self.label,
            elements=[FieldElement('cells', element=BrusselatorElement, space=LatticeSpace(shape), values=values)],
            interactions=[*self._get_reactions(), *_get_diffusions()],
        )


class SchnakenbergSystem(ReactionDiffusionSystem):
    """The Schnakenberg reaction-diffusion system.

    It forms Turing patterns from noise when `Dv >> Du`.

    The steady state `u = a + b`, `v = b / (a + b)^2` is stable without diffusion and unstable to perturbations of a
    band of wavelengths with it. The concentrations are `cells.u` and `cells.v`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.chemistry.pattern_formation import SchnakenbergSystem
        >>> SchnakenbergSystem((8, 8)).prepare().system_.elements[0].get_declaring_part().get_value('Dv')
        40.0
    """

    field = 'chemistry/pattern_formation'
    topics = ('field', 'reaction-diffusion', 'Turing', 'pde')
    references = ('Schnakenberg 1979', 'Turing 1952', 'Murray 2003')
    until = 200.0
    dt = 5.0
    solver = RK4Solver(dt=0.005)

    def __init__(
        self,
        shape: Sequence[int] = (64, 64),
        *,
        a: float = 0.1,
        b: float = 0.9,
        Du: float = 1.0,
        Dv: float = 40.0,
        spacing: float = 1.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.shape = shape
        self.a = a
        self.b = b
        self.Du = Du
        self.Dv = Dv
        self.spacing = spacing
        self.random_state = random_state

    @staticmethod
    def _get_reactions() -> list[ReactionInteraction]:
        """Return the reactions of the Schnakenberg system at every site of `cells`."""
        return [
            ReactionInteraction('production_u', target='cells', products={'u': 1}, modifiers={'a': 1}),
            ReactionInteraction('decay', target='cells', reactants={'u': 1}),
            CubicAutocatalysisInteraction('autocatalysis', target='cells', substrate='v', autocatalyst='u'),
            ReactionInteraction('production_v', target='cells', products={'v': 1}, modifiers={'b': 1}),
        ]

    def _get_element_label(self) -> str:
        """Return the label of the field of the sites."""
        return 'cells'

    def _get_space(self) -> LatticeSpace:
        """Return the periodic lattice of the shape."""
        return LatticeSpace(tuple(self.shape))

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the fields near its steady state, and the parameters."""
        shape = tuple(self.shape)
        steady_u, steady_v = self.a + self.b, self.b / (self.a + self.b) ** 2
        return {
            'u': DeclaredVariableSpec(steady_u * (1 + _noise(shape, 0.01, self.random_state)), StateSpec()),
            'v': DeclaredVariableSpec(np.full(shape, steady_v), StateSpec()),
            'a': DeclaredVariableSpec(self.a, ParameterSpec()),
            'b': DeclaredVariableSpec(self.b, ParameterSpec()),
            'Du': DeclaredVariableSpec(self.Du, ParameterSpec()),
            'Dv': DeclaredVariableSpec(self.Dv, ParameterSpec()),
        }

    def _get_diffusivities(self) -> dict[str, Callable[[SimpleNamespace], object]]:
        """Return the diffusivities `Du` and `Dv` of the sites."""
        return {'u': _get_u_diffusion, 'v': _get_v_diffusion}

    def _get_equation_interactions(self) -> list[ReactionInteraction]:
        """Return the reactions at every site."""
        return self._get_reactions()

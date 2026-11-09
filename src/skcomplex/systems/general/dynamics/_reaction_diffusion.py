"""Define reaction-diffusion systems.

A reaction-diffusion system puts the states of a dynamical system as fields on the sites of a lattice, with local
kinetics and a diffusion of every field that has a diffusivity.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace

from ....core.components.elements import BaseElement, DeclaredElement, FieldElement
from ....core.components.interactions import BaseInteraction, RateInteraction
from ....core.components.laws.utils import laplacian
from ....core.components.spaces import LatticeSpace
from ....core.errors import ValidationError
from ....core.specs import StateSpec
from ._dynamical import DynamicalSystem
from .types import Equations, Noises


class ReactionDiffusionSystem(DynamicalSystem):
    """Fields on the sites of a lattice with local kinetics and diffusion.

    The states and the parameters are those of a `DeclaredElement`, a member of a `FieldElement` on a `LatticeSpace`
    of a shape, a spacing, and a boundary. The equations are the local kinetics of every site, and every field `u` with
    a diffusivity `D` gains the rate `D laplacian(u)`, a `RateInteraction` apart from the kinetics, so a split solver
    can treat it. A diffusivity is a number or a function of the namespace, such as one that reads a parameter of the
    sites. An initial value is a number for every site or an array of the shape of the lattice. The system runs in
    continuous time.

    The Gray-Scott model of chemistry and the front of Fisher and Kolmogorov in ecology are reaction-diffusion systems.
    A subclass returns the diffusivities from `_get_diffusivities`, the lattice from `_get_space`, and the boundary of
    every field from `_get_boundaries`, besides the hooks of `DynamicalSystem`. A system whose kinetics are reactions
    returns them from `_get_equation_interactions`.

    Attributes:
        system_:
            The composite of the field, its kinetics, and its diffusion built at the last `prepare`.

        laws_:
            The laws the system contributes for the namespace of its last run.

    Examples:
        The front of Fisher and Kolmogorov, which invades the empty right side of a line:

        >>> import numpy as np
        >>> from skcomplex.systems.general.dynamics import ReactionDiffusionSystem
        >>> from skcomplex.core.engine.runtime import simulate
        >>> initial = np.zeros(50)
        >>> initial[:5] = 1.0
        >>> front = ReactionDiffusionSystem(
        ...     states={'u': initial},
        ...     equations={'u': lambda s: s.target.r * s.target.u * (1 - s.target.u)},
        ...     diffusivities={'u': 1.0},
        ...     shape=50,
        ...     boundary='reflecting',
        ...     parameters={'r': 1.0},
        ... )
        >>> trajectory = simulate(front, until=10.0, dt=1.0)
        >>> bool(trajectory.values('medium.u')[-1][20] > trajectory.values('medium.u')[0][20])
        True

        The Gray-Scott model, whose substrate `u` feeds the autocatalysis of the activator `v`:

        >>> u, v = np.ones(32), np.zeros(32)
        >>> u[12:20], v[12:20] = 0.5, 0.25
        >>> gray_scott = ReactionDiffusionSystem(
        ...     states={'u': u, 'v': v},
        ...     equations={
        ...         'u': lambda s: -s.target.u * s.target.v**2 + s.target.F * (1 - s.target.u),
        ...         'v': lambda s: s.target.u * s.target.v**2 - (s.target.F + s.target.k) * s.target.v,
        ...     },
        ...     diffusivities={'u': 0.2, 'v': 0.1},
        ...     shape=32,
        ...     parameters={'F': 0.04, 'k': 0.06},
        ... )
        >>> trajectory = simulate(gray_scott, until=20.0, dt=5.0)
        >>> bool(trajectory.values('medium.v')[-1].max() > 0)
        True
    """

    def __init__(
        self,
        states: Mapping[str, object] | None = None,
        equations: Equations | None = None,
        diffusivities: Mapping[str, Callable[[SimpleNamespace], object] | float] | None = None,
        *,
        shape: Sequence[int] | int = 64,
        spacing: float = 1.0,
        boundary: str = 'periodic',
        discretization: str = 'finite-difference',
        parameters: Mapping[str, object] | None = None,
        noises: Noises | None = None,
        convention: str = 'ito',
        element: str = 'medium',
        label: str = 'system',
    ) -> None:
        """Store the fields, the kinetics, the diffusivities, the lattice, and the rest of the data.

        Args:
            states:
                The initial value of every field by name, a number for every site or an array of the shape.

            equations:
                The local kinetics of every field by name, as a function of the namespace, or `None`.

            diffusivities:
                The diffusivity of every field that diffuses, by name, a number or a function of the namespace, or
                `None` for none.

            shape:
                The number of sites along every axis of the lattice, or one number for a line.

            spacing:
                The distance between neighbouring sites.

            boundary:
                The boundary of the lattice, `'periodic'`, `'fixed'`, or `'reflecting'`.

            discretization:
                The discretization of the field, `'finite-difference'` or `'spectral'`.

            parameters:
                The value of every parameter by name, or `None` for none.

            noises:
                The noise amplitude of every field that has one, by name, or `None` for none.

            convention:
                The stochastic calculus of the noises, `'ito'` or `'stratonovich'`.

            element:
                The label of the field.

            label:
                The label of the system, which names it when it is nested in another one.
        """
        super().__init__(
            states,
            equations,
            parameters=parameters,
            noises=noises,
            convention=convention,
            element=element,
            label=label,
        )
        self.diffusivities = diffusivities
        self.shape = shape
        self.spacing = spacing
        self.boundary = boundary
        self.discretization = discretization

    def _get_time(self) -> str:
        """Return the continuous time of the rates."""
        return 'continuous'

    def _get_diffusivities(self) -> Mapping[str, Callable[[SimpleNamespace], object] | float]:
        """Return the diffusivity of every field that diffuses.

        A subclass with diffusivities of its own returns them.
        """
        return self.diffusivities or {}

    def _get_space(self) -> LatticeSpace:
        """Return the lattice of the sites.

        A subclass with a lattice of its own returns it.
        """
        return LatticeSpace(self.shape, spacing=self.spacing, boundary=self.boundary)

    def _get_boundaries(self) -> Mapping[str, object] | None:
        """Return the boundary of every field, or `None` for the boundary of the lattice.

        A subclass with boundaries per field returns them.
        """
        return None

    def _check_equations(self) -> None:
        """Raise an error for the equation, the noise, or the diffusivity of an unknown state.

        Raises:
            ValidationError:
                An equation, a noise, or a diffusivity names a state the system does not declare.
        """
        super()._check_equations()
        states = [name for name, variable in self._get_variables().items() if isinstance(variable.spec, StateSpec)]
        unknown = [name for name in self._get_diffusivities() if name not in states]
        if unknown:
            error_msg = (
                f'{type(self).__name__} has a diffusivity for the state {unknown[0]!r}, which it does not declare. '
                f'The declared states are {states}.'
            )
            raise ValidationError(error_msg)

    def _get_element(self) -> BaseElement:
        """Return the field on the lattice whose members hold the states and the parameters.

        The initial values of the states are also given to the field, which reads an array of the shape of the
        lattice as one value per site.
        """
        variables = self._get_variables()
        states = {name: variable.value for name, variable in variables.items() if isinstance(variable.spec, StateSpec)}
        return FieldElement(
            self._get_element_label(),
            element=DeclaredElement,
            space=self._get_space(),
            boundary=self._get_boundaries(),
            values={'variables': variables, **states},
            discretization=self.discretization,
        )

    def _get_diffusion_interactions(self) -> list[RateInteraction]:
        """Return the diffusion `D laplacian(u)` of every field that diffuses, labelled `<field>_diffusion`."""
        element = self._get_element_label()
        return [
            RateInteraction(
                f'{state}_diffusion',
                target=element,
                state=state,
                rate=self._get_diffusion_rate(state, diffusivity),
            )
            for state, diffusivity in self._get_diffusivities().items()
        ]

    @staticmethod
    def _get_diffusion_rate(
        state: str,
        diffusivity: Callable[[SimpleNamespace], object] | float,
    ) -> Callable[[SimpleNamespace], object]:
        """Return the rate `D laplacian(u)` of a field as a function of the namespace."""
        if callable(diffusivity):
            return lambda s: diffusivity(s) * laplacian(getattr(s.target, state))
        return lambda s: diffusivity * laplacian(getattr(s.target, state))

    def _get_interactions(self) -> list[BaseInteraction]:
        """Return the local kinetics and the diffusion of the fields."""
        return [*super()._get_interactions(), *self._get_diffusion_interactions()]

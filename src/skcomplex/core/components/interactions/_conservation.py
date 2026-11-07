"""Define conservation laws, which change a conserved state of a field by the divergence of a flux and a source."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

from sage.modules.free_module_element import FreeModuleElement

from ...errors import ValidationError
from ...specs import RoleSpec, Tags
from ...types import ElementKind
from ..base import BasePart
from ..laws import BaseLaw, FluxLaw, RateLaw
from ..laws.utils import divergence
from ._base import RATE_DOMAINS
from ._interaction import Interaction

LIMITERS = ('minmod', 'van-leer', None)
RIEMANN_SOLVERS = ('rusanov', 'godunov', None)


class ConservationInteraction(Interaction):
    """Conservation law `du/dt + divergence(flux(ns)) = source(ns)` of a state `u` of a field.

    The flux is one expression for a field of one dimension, or one expression per dimension, of the states of the
    field, and the interaction lowers it to a finite-volume flux law with a slope limiter and a numerical flux, or,
    without a numerical flux, to the rate `-divergence(flux)` that the field discretizes by its own operators, as a
    spectral or finite-difference field does. Without a source the total of the state on a periodic field stays
    constant. The conservation laws of several states of one field form one system, whose numerical flux reads the
    wave speeds of all of them, and a flux may read parameters that vary across the cells. A flux or a source function
    takes the namespace of the interaction, with the namespace `target` of the field, the time `t`, and the symbols of
    the parameters a subclass declares.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the field whose state is conserved.

        state:
            The name of the conserved state of the field.

        flux:
            A function of the namespace that returns the flux, or one flux per dimension.

        source:
            A function of the namespace that returns the source, or `None` for none.

        limiter:
            The slope limiter of the reconstruction, `'minmod'`, `'van-leer'`, or `None` for first order.

        riemann:
            The numerical flux of a finite-volume field, `'rusanov'` or `'godunov'`, the exact solver for one state, or
            `None` for the divergence of the flux on a field of another discretization.

        sources:
            The paths of other elements the flux and the source read, as the namespaces `sources`. A field over the
            position lattice of a target on a phase space is read broadcast along the class axes.

    Raises:
        ValidationError:
            The state is not a name, a function is not callable, or the limiter or the numerical flux is unknown.

    Examples:
        >>> from typing import Annotated
        >>> import numpy as np
        >>> from skcomplex.core.components.elements import Element, FieldElement
        >>> from skcomplex.core.components.interactions import ConservationInteraction
        >>> from skcomplex.core.components.spaces import LatticeSpace
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.core.engine.runtime import Simulation
        >>> from skcomplex.core.specs import StateSpec
        >>> class CellElement(Element):
        ...     def __init__(self, label, u: Annotated[float, StateSpec()] = 0.0):
        ...         super().__init__(label)
        ...         self.u = u
        >>> space = LatticeSpace((50,), spacing=0.1, boundary='periodic')
        >>> values = {'u': 1.0 + np.sin(2 * np.pi * np.arange(50) / 50)}
        >>> volumes = 'finite-volume'
        >>> field = FieldElement('wave', element=CellElement, space=space, values=values, discretization=volumes)
        >>> burgers = ConservationInteraction('burgers', target='wave', state='u', flux=lambda s: s.target.u**2 / 2)
        >>> system = System(elements=[field], interactions=[burgers])
        >>> trajectory = Simulation(system, dt=0.01).prepare().run(until=0.5).trajectory_
        >>> round(float(trajectory.values('wave.u')[-1].sum()), 6)
        50.0
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (FluxLaw, RateLaw)

    def _get_tags(self) -> Tags:
        """Return the tags of a flux with a Riemann solver or of a rate, with a rate for a source."""
        transport = (FluxLaw,) if self._get_argument('riemann') is not None else (RateLaw,)
        source = () if self._get_argument('source') is None else (RateLaw,)
        return self._get_variant_tags((*transport, *source))

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec(elements=(ElementKind.ANY,), states=('state',), domains=RATE_DOMAINS)],
        state: str = 'u',
        flux: Callable[[SimpleNamespace], object] | None = None,
        source: Callable[[SimpleNamespace], object] | None = None,
        limiter: str | None = 'minmod',
        riemann: str | None = 'rusanov',
        sources: Annotated[Sequence[str], RoleSpec(variadic=True, elements=(ElementKind.ANY,))] = (),
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.flux = flux
        self.source = source
        self.limiter = limiter
        self.riemann = riemann
        self.sources = sources

    def _check(self) -> None:
        """Raise an error for a state that is not a name, a function that is not callable, or an unknown scheme.

        Raises:
            ValidationError:
                The type declares states, the state is not a name, a function is not callable, or the limiter or the
                numerical flux is unknown.
        """
        super()._check()
        if not isinstance(self.state, str) or not self.state.isidentifier():
            error_msg = f'{type(self).__name__} {self.label!r} conserves the state {self.state!r}, which is no name.'
            raise ValidationError(error_msg)
        for name in ('flux', 'source'):
            value = getattr(self, name)
            if value is not None and not callable(value):
                error_msg = f'The {name} of {self.label!r} is {value!r}. Give a function of the namespace.'
                raise ValidationError(error_msg)
        if self.limiter not in LIMITERS:
            error_msg = f'Unknown limiter {self.limiter!r} of {self.label!r}. Use minmod, van-leer, or None.'
            raise ValidationError(error_msg)
        if self.riemann not in RIEMANN_SOLVERS:
            error_msg = f'Unknown numerical flux {self.riemann!r} of {self.label!r}. Use rusanov, godunov, or None.'
            raise ValidationError(error_msg)

    def _get_flux(self, namespace: SimpleNamespace) -> object:
        """Return the flux, or `None` for none, the hook a subclass overrides to derive it from its arguments."""
        return None if self.flux is None else self.flux(namespace)

    def _get_transport_laws(self, namespace: SimpleNamespace) -> list[BasePart]:
        """Return the laws of the divergence of the flux, the hook a subclass overrides to lower its own flux."""
        flux = self._get_flux(namespace)
        if flux is None:
            return []
        state = getattr(self._get_member(namespace, 'target'), self.state)
        if self.riemann is None:
            components = list(flux) if isinstance(flux, FreeModuleElement | list | tuple) else [flux]
            return [RateLaw(state, -divergence(components))]
        return [FluxLaw(state, flux, limiter=self.limiter, riemann=self.riemann)]

    def _get_laws(self, namespace: SimpleNamespace) -> list[BasePart]:
        """Return the laws of the divergence of the flux and the rate of the source.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with the namespace of the field.

        Returns:
            laws:
                The flux law, and the rate law of the source when there is one.
        """
        laws = self._get_transport_laws(namespace)
        if self.source is not None:
            laws.append(RateLaw(getattr(self._get_member(namespace, 'target'), self.state), self.source(namespace)))
        return laws

"""Nanoparticle systems: the corona of blood proteins that forms on a nanoparticle."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import NonnegativeElement
from ....core.components.interactions import ReactionInteraction
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.solvers import ScipyODESolver
from ....core.specs import StateSpec, VisualSpec

PROTEINS = ('albumin', 'lipoprotein', 'fibrinogen')
LITRES_PER_NANOMOLE = 1e-9


class SpeciesElement(NonnegativeElement):
    """Chemical species with a nonnegative concentration `x`.

    It is a free nanoparticle, a free protein, or a complex of the two. A species has no laws of its own, and the
    binding and unbinding reactions move its amount.

    Args:
        label:
            The label of the species.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial concentration.

    Examples:
        >>> from skcomplex.systems.chemistry.kinetics import SpeciesElement
        >>> SpeciesElement('albumin', x=600.0).x
        600.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual, x=x)


class NanoparticleCoronaSystem(BaseSystem):
    """The corona that plasma proteins form on a nanoparticle.

    The proteins are albumin, high-density lipoprotein, and fibrinogen.

    Each protein `i` binds a free nanoparticle at the rate `n_i alpha_i U P_i` and leaves it at the rate
    `delta_i B_i`, where `n_i = 4 (R + r_i)^2 / r_i^2` is the number of its sites on a nanoparticle of radius `R`.
    The book gives only the balance of the complexes, and the system takes the free nanoparticles and the free proteins
    from mass action, so `U + sum B_i` and every `P_i + B_i` are conserved. The parameters and initial
    concentrations are the book's, without the binding to a target. Concentrations are in nM, association rates in
    1/(M s), and time is in seconds. Albumin, the most abundant, covers the nanoparticles first, and lipoprotein,
    which binds more tightly, replaces it within hours, which is the Vroman effect. The free nanoparticles are
    `nanoparticle.x`, the free proteins `albumin.x`, `lipoprotein.x`, and `fibrinogen.x`, and the complexes
    `albumin_corona.x`, `lipoprotein_corona.x`, and `fibrinogen_corona.x`.

    Args:
        nanoparticle:
            The initial concentration of free nanoparticles.

        radius:
            The radius of a nanoparticle, in nm.

        proteins:
            The initial concentrations of free albumin, lipoprotein, and fibrinogen.

        corona:
            The initial concentrations of nanoparticles bound to each protein.

        protein_radii:
            The radii of the proteins, in nm.

        association:
            The association rates of the proteins.

        dissociation:
            The dissociation rates of the proteins, per second.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.chemistry.kinetics import NanoparticleCoronaSystem
        >>> round(NanoparticleCoronaSystem().prepare().system_.interactions[0].k, 7)
        0.0006144
    """

    field = 'chemistry/kinetics'
    topics = ('nanomedicine', 'protein corona', 'mass action', 'ode', 'Vroman effect')
    references = ('Padmanabhan, Meskin, and Al Moustafa 2021', 'Dell\'Orco et al. 2010')
    until = 10800.0
    dt = 60.0
    solver = ScipyODESolver(method='LSODA', rtol=1e-8, atol=1e-12)

    def __init__(
        self,
        nanoparticle: float = 18.0,
        *,
        radius: float = 35.0,
        proteins: Sequence[float] = (6e5, 1.5e4, 8.8e3),
        corona: Sequence[float] = (14.0, 4.0, 0.0),
        protein_radii: Sequence[float] = (5.0, 4.0, 8.3),
        association: Sequence[float] = (2.4e3, 3e4, 2e3),
        dissociation: Sequence[float] = (2e-3, 3e-5, 2e-3),
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.nanoparticle = nanoparticle
        self.radius = radius
        self.proteins = proteins
        self.corona = corona
        self.protein_radii = protein_radii
        self.association = association
        self.dissociation = dissociation

    @staticmethod
    def _count_binding_sites(radius: float, protein_radius: float) -> float:
        """Return the number of sites for a protein on a nanoparticle.

        It is `4 pi (R + r)^2 / (pi r^2)`.
        """
        return 4 * (radius + protein_radius) ** 2 / protein_radius**2

    def _build_system(self) -> CompositeSystem:
        """Return the nanoparticles, the proteins, and the binding and release reactions.

        The proteins are free and bound, and every protein binds and is released.
        """
        elements = [SpeciesElement('nanoparticle', x=self.nanoparticle)]
        interactions: list[ReactionInteraction] = []
        rows = zip(
            PROTEINS,
            self.proteins,
            self.corona,
            self.protein_radii,
            self.association,
            self.dissociation,
            strict=True,
        )
        for name, free, bound, protein_radius, alpha, delta in rows:
            complex_ = f'{name}_corona'
            elements += [SpeciesElement(name, x=free), SpeciesElement(complex_, x=bound)]
            k = self._count_binding_sites(self.radius, protein_radius) * alpha * LITRES_PER_NANOMOLE
            interactions += [
                ReactionInteraction(
                    f'{name}_binding',
                    reactants={('nanoparticle', 'x'): 1, (name, 'x'): 1},
                    products={(complex_, 'x'): 1},
                    k=k,
                ),
                ReactionInteraction(
                    f'{name}_release',
                    reactants={(complex_, 'x'): 1},
                    products={('nanoparticle', 'x'): 1, (name, 'x'): 1},
                    k=delta,
                ),
            ]
        return CompositeSystem(self.label, elements=elements, interactions=interactions)

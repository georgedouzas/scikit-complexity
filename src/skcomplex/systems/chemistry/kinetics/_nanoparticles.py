"""Nanoparticle systems: the corona of blood proteins that forms on a nanoparticle."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

from ....core.components.elements import CompartmentElement
from ....core.components.interactions import ReactionInteraction
from ....core.components.systems import BaseSystem, System
from ....core.engine.solvers import ScipyODESolver

PROTEINS = ('albumin', 'lipoprotein', 'fibrinogen')
LITRES_PER_NANOMOLE = 1e-9


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
    `nanoparticle.n`, the free proteins `albumin.n`, `lipoprotein.n`, and `fibrinogen.n`, and the complexes
    `albumin_corona.n`, `lipoprotein_corona.n`, and `fibrinogen_corona.n`.

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

    def _build_system(self) -> System:
        """Return the nanoparticles, the proteins, and the binding and release reactions.

        The proteins are free and bound, and every protein binds and is released.
        """
        elements = [CompartmentElement('nanoparticle', n=self.nanoparticle)]
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
            elements += [CompartmentElement(name, n=free), CompartmentElement(complex_, n=bound)]
            k = self._count_binding_sites(self.radius, protein_radius) * alpha * LITRES_PER_NANOMOLE
            interactions += [
                ReactionInteraction(
                    f'{name}_binding',
                    reactants={('nanoparticle', 'n'): 1, (name, 'n'): 1},
                    products={(complex_, 'n'): 1},
                    k=k,
                ),
                ReactionInteraction(
                    f'{name}_release',
                    reactants={(complex_, 'n'): 1},
                    products={('nanoparticle', 'n'): 1, (name, 'n'): 1},
                    k=delta,
                ),
            ]
        return System(self.label, elements=elements, interactions=interactions)

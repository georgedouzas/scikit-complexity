"""Nanoparticle models: the corona of blood proteins that forms on a nanoparticle."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

from ....core.model import System
from ....core.runtime import register_model
from ....core.solvers import ScipyODE
from ._reactions import Binding, Dissociation, Species

PROTEINS = ('albumin', 'lipoprotein', 'fibrinogen')
LITRES_PER_NANOMOLE = 1e-9


def _count_binding_sites(radius: float, protein_radius: float) -> float:
    """Return the number of sites for a protein on a nanoparticle, `4 pi (R + r)^2 / (pi r^2)`."""
    return 4 * (radius + protein_radius) ** 2 / protein_radius**2


@register_model(
    field='chemistry/kinetics',
    tags=('nanomedicine', 'protein corona', 'mass action', 'ode', 'Vroman effect'),
    references=('Padmanabhan, Meskin, and Al Moustafa 2021', 'Dell\'Orco et al. 2010'),
    defaults={'until': 10800.0, 'dt': 60.0, 'solver': ScipyODE(method='LSODA', rtol=1e-8, atol=1e-12)},
)
def nanoparticle_corona(
    nanoparticle: float = 18.0,
    *,
    radius: float = 35.0,
    proteins: Sequence[float] = (6e5, 1.5e4, 8.8e3),
    corona: Sequence[float] = (14.0, 4.0, 0.0),
    protein_radii: Sequence[float] = (5.0, 4.0, 8.3),
    association: Sequence[float] = (2.4e3, 3e4, 2e3),
    dissociation: Sequence[float] = (2e-3, 3e-5, 2e-3),
) -> System:
    """Return the corona that albumin, high-density lipoprotein, and fibrinogen form on a nanoparticle in plasma.

    Each protein `i` binds a free nanoparticle at the rate `n_i alpha_i U P_i` and leaves it at the rate
    `delta_i B_i`, where `n_i = 4 (R + r_i)^2 / r_i^2` is the number of its sites on a nanoparticle of radius `R`.
    The book gives only the balance of the complexes, and the model takes the free nanoparticles and the free proteins
    from mass action, so `U + sum B_i` and every `P_i + B_i` are conserved. The parameters and initial
    concentrations are the book's, without the binding to a target. Concentrations are in nM, association rates in
    1/(M s), and time is in seconds. Albumin, the most abundant, covers the nanoparticles first, and lipoprotein,
    which binds more tightly, replaces it within hours, which is the Vroman effect.

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

    Returns:
        system:
            The model, with the free nanoparticles `nanoparticle.n`, the free proteins `albumin.n`,
            `lipoprotein.n`, and `fibrinogen.n`, and the complexes `albumin_corona.n`, `lipoprotein_corona.n`, and
            `fibrinogen_corona.n`.

    Examples:
        >>> from skcomplex.models.chemistry.kinetics import nanoparticle_corona
        >>> round(nanoparticle_corona().interactions[0].k, 7)
        0.0006144
    """
    elements = [Species('nanoparticle', n=nanoparticle)]
    interactions: list[Binding | Dissociation] = []
    rows = zip(PROTEINS, proteins, corona, protein_radii, association, dissociation, strict=True)
    for name, free, bound, protein_radius, alpha, delta in rows:
        complex_ = f'{name}_corona'
        elements += [Species(name, n=free), Species(complex_, n=bound)]
        k = _count_binding_sites(radius, protein_radius) * alpha * LITRES_PER_NANOMOLE
        interactions += [
            Binding(f'{name}_binding', first='nanoparticle', second=name, product=complex_, k=k),
            Dissociation(f'{name}_release', complex=complex_, first='nanoparticle', second=name, k=delta),
        ]
    return System(elements=elements, interactions=interactions)

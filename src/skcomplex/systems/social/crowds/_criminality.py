"""Crowd systems: the kinetic theory of criminality in a society and the security forces that act against it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from functools import partial
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression

from ....core.components.elements import DensityElement, GroupElement
from ....core.components.laws.utils import moment
from ....core.components.spaces import ContinuousTime, LatticeSpace
from ....core.components.system import System
from ....core.specs import ObservableSpec
from ..._catalog import register_system
from ._criminal_contagion import CriminalContagionInteraction
from ._deterrence import DeterrenceInteraction
from ._recruitment import RecruitmentInteraction, criminality_grid
from ._security_agents import SecurityAgentsElement

CRIMINAL_THRESHOLD = 0.5


CRIMINALITY_REFERENCES = ('Bellomo et al. 2017', 'Bellomo, Colasuonno, Knopoff, and Soler 2015')


def mean_criminality(s: SimpleNamespace, criminality: npt.NDArray[np.float64]) -> Expression:
    """Return the mean criminality of a distribution of citizens.

    Args:
        s:
            The namespace of one member of the distribution.

        criminality:
            The criminality of every class.

    Returns:
        mean:
            The normalized first moment.
    """
    return moment(s.f, 1, criminality, normalized=True)


def criminal_fraction(s: SimpleNamespace, criminality: npt.NDArray[np.float64]) -> Expression:
    """Return the fraction of citizens whose criminality is at least one half.

    Args:
        s:
            The namespace of one member of the distribution.

        criminality:
            The criminality of every class.

    Returns:
        fraction:
            The moment of the indicator of the criminal classes, divided by the density.
    """
    return moment(s.f, 1, (criminality >= CRIMINAL_THRESHOLD).astype(float), normalized=True)


def security_density(s: SimpleNamespace, forces: int) -> Expression:
    """Return the density of the security forces.

    Args:
        s:
            The namespace of one member of the distribution of the security forces.

        forces:
            The number of efficacy classes.

    Returns:
        density:
            The order-0 moment.
    """
    return moment(s.f, 0, np.arange(forces, dtype=float))


def _sizes_of(system: System) -> tuple[int, int]:
    """Return the number of criminality classes and of efficacy classes of a system."""
    sizes = {item.label: int(item.size or 0) for item in system.elements}
    return sizes['citizens'], sizes['security']


def criminality_observables(classes: int, forces: int) -> list[ObservableSpec]:
    """Return the mean criminality, the criminal fraction, and the density of the security forces of the system.

    Args:
        classes:
            The number of criminality classes.

        forces:
            The number of efficacy classes of the security forces.

    Returns:
        observables:
            The observables, each reading the moments of a distribution.

    Examples:
        >>> from skcomplex.systems.social.crowds import criminality_observables
        >>> [item.label for item in criminality_observables(5, 3)]
        ['mean_criminality', 'criminal_fraction', 'security_forces']
    """
    criminality = criminality_grid(classes)
    return [
        ObservableSpec('mean_criminality', partial(mean_criminality, criminality=criminality), group='citizens'),
        ObservableSpec('criminal_fraction', partial(criminal_fraction, criminality=criminality), group='citizens'),
        ObservableSpec('security_forces', partial(security_density, forces=forces), group='security'),
    ]


@register_system(
    field='social/crowds',
    tags=('criminality', 'security', 'kinetic', 'distribution'),
    references=CRIMINALITY_REFERENCES,
    defaults={'until': 60, 'dt': 0.5},
    observables=lambda system: criminality_observables(*_sizes_of(system)),
)
def criminality(
    classes: int = 5,
    forces: int = 3,
    *,
    criminals: float = 0.2,
    security: float = 0.05,
    attraction: float = 0.3,
    redemption: float = 0.1,
    deterrence: float = 2.0,
    recruitment: float = 0.5,
    retirement: float = 0.1,
) -> System:
    """Return the kinetic system of criminality in a society and the security forces that act against it.

    Citizens are a distribution over criminality classes and security agents a distribution over efficacy classes,
    held by site groups on lattices of the classes. Encounters between citizens spread crime when it attracts more than
    honesty redeems, encounters with security agents move criminal citizens toward honesty, a micro-scale action, and
    the security forces grow in proportion to the mean criminality, a macro-scale action that reads a moment of the
    citizens. The security forces grow while crime is high, crime recedes, and the forces retire, so crime and
    security oscillate toward a balance where some crime persists.

    Args:
        classes:
            The number of criminality classes.

        forces:
            The number of efficacy classes of the security forces.

        criminals:
            The initial fraction of citizens in the most criminal class, the others being honest.

        security:
            The initial density of the security forces, spread evenly over their efficacy classes.

        attraction:
            The probability that a citizen who meets a more criminal one moves one class up.

        redemption:
            The probability that a citizen who meets a more honest one moves one class down.

        deterrence:
            The rate at which the most effective agents deter a criminal citizen.

        recruitment:
            The recruitment rate of security agents per unit of mean criminality.

        retirement:
            The retirement rate of security agents.

    Returns:
        system:
            The system, with the distributions `citizens` and `security` and the interactions `contagion`,
            `deterrence`, and `recruitment`. Its standard observables are the mean criminality, the criminal fraction,
            and the density of the security forces.

    Examples:
        >>> from skcomplex.systems.social.crowds import criminality
        >>> [item.label for item in criminality().interactions]
        ['contagion', 'deterrence', 'recruitment']
    """
    citizens = np.zeros(classes)
    citizens[0], citizens[-1] = 1.0 - criminals, criminals
    return System(
        elements=[
            GroupElement('citizens', element=DensityElement, size=classes, on='sites', values={'f': citizens}),
            GroupElement(
                'security',
                element=SecurityAgentsElement,
                size=forces,
                on='sites',
                space=LatticeSpace(forces, boundary='fixed'),
                values={'f': np.full(forces, security / forces), 'share': np.full(forces, 1.0 / forces)},
            ),
        ],
        interactions=[
            CriminalContagionInteraction(
                'contagion',
                candidate='citizens',
                field='citizens',
                classes=classes,
                attraction=attraction,
                redemption=redemption,
            ),
            DeterrenceInteraction(
                'deterrence',
                candidate='citizens',
                field='security',
                classes=classes,
                forces=forces,
                deterrence=deterrence,
            ),
            RecruitmentInteraction(
                'recruitment',
                target='security',
                citizens='citizens',
                classes=classes,
                recruitment=recruitment,
                retirement=retirement,
            ),
        ],
        space=LatticeSpace(classes, boundary='fixed'),
        time=ContinuousTime(),
    )

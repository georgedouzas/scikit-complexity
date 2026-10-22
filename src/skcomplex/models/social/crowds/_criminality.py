"""Crowd models: the kinetic theory of criminality in a society and the security forces that act against it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from functools import partial

import numpy as np
import numpy.typing as npt

from ....core.base import Domain, Expression, parameter, role, state
from ....core.components import Element, Group
from ....core.interactions import Interaction
from ....core.laws import Encounter, Law, Rate, Symbols, moment
from ....core.runtime import register_model
from ....core.space import ContinuousTime, LatticeSpace
from ....core.system import Observable, System

CRIMINAL_THRESHOLD = 0.5
CRIMINALITY_REFERENCES = ('Bellomo et al. 2017', 'Bellomo, Colasuonno, Knopoff, and Soler 2015')


def criminality_grid(classes: int) -> npt.NDArray[np.float64]:
    """Return the criminality of every class of citizens, from 0 for honest to 1 for criminal.

    Args:
        classes:
            The number of criminality classes.

    Returns:
        criminality:
            The criminality of every class.

    Examples:
        >>> from skcomplex.models.social.crowds import criminality_grid
        >>> criminality_grid(3).tolist()
        [0.0, 0.5, 1.0]
    """
    return np.linspace(0.0, 1.0, classes)


def contagion_table(classes: int, attraction: object, redemption: object) -> npt.NDArray[np.object_]:
    """Return the transition table of encounters between citizens, `A[h, k, i]`.

    A citizen of class `h` who meets a more criminal citizen moves one class up with probability `attraction`, and one
    who meets a more honest citizen moves one class down with probability `redemption`.

    Args:
        classes:
            The number of criminality classes.

        attraction:
            The probability that crime attracts, a number or a parameter.

        redemption:
            The probability that honesty redeems, a number or a parameter.

    Returns:
        table:
            The transition probabilities, which sum to one over the outcomes.

    Examples:
        >>> from skcomplex.models.social.crowds import contagion_table
        >>> contagion_table(3, 0.3, 0.1)[0, 2].astype(float).tolist()
        [0.7, 0.3, 0.0]
    """
    table = np.zeros((classes, classes, classes), dtype=object)
    for candidate in range(classes):
        for field in range(classes):
            if field > candidate:
                table[candidate, field, candidate + 1] += attraction
            elif field < candidate:
                table[candidate, field, candidate - 1] += redemption
            table[candidate, field, candidate] += 1 - table[candidate, field].sum()
    return table


def deterrence_table(classes: int, forces: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the rate pattern and the transition table of the encounters of citizens with security forces.

    A citizen of a criminal class `h > 0` who meets a security agent of efficacy `v_k` moves one class down at a rate
    proportional to `v_k`. Honest citizens are left alone.

    Args:
        classes:
            The number of criminality classes.

        forces:
            The number of efficacy classes of the security forces.

    Returns:
        pattern:
            The rate pattern `eta[h, k]`, the efficacy `v_k` for criminal classes and 0 for honest ones.

        table:
            The transition table `A[h, k, i]`.

    Examples:
        >>> from skcomplex.models.social.crowds import deterrence_table
        >>> deterrence_table(2, 2)[0].tolist()
        [[0.0, 0.0], [0.5, 1.0]]
    """
    efficacy = np.linspace(1.0 / forces, 1.0, forces)
    pattern = np.zeros((classes, forces))
    pattern[1:] = efficacy
    table = np.zeros((classes, forces, classes))
    for candidate in range(classes):
        table[candidate, :, max(candidate - 1, 0)] = 1.0
    return pattern, table


def mean_criminality(s: Symbols, criminality: npt.NDArray[np.float64]) -> Expression:
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


def criminal_fraction(s: Symbols, criminality: npt.NDArray[np.float64]) -> Expression:
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


def security_density(s: Symbols, forces: int) -> Expression:
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


class Citizens(Element):
    """Density of citizens at one criminality class, a member of a distribution over criminality.

    Args:
        label:
            The label of the distribution.

        f:
            The initial density.

    Examples:
        >>> from skcomplex.models.social.crowds import Citizens
        >>> Citizens.group('citizens', 5, f=0.2).size
        5
    """

    f: float = state(default=0.0, domain=Domain.NONNEGATIVE)


class SecurityAgents(Element):
    """Density of security agents at one efficacy class, with the share of new recruits that join the class.

    Args:
        label:
            The label of the distribution.

        f:
            The initial density.

        share:
            The share of new recruits that join the efficacy class.

    Examples:
        >>> from skcomplex.models.social.crowds import SecurityAgents
        >>> SecurityAgents.group('security', 3, share=[0.5, 0.3, 0.2]).size
        3
    """

    f: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    share: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)


class CriminalContagion(Interaction):
    """Conservative encounters between citizens, in which crime attracts and honesty redeems.

    Args:
        label:
            The label of the interaction.

        citizens:
            The path of the distribution of citizens.

        classes:
            The number of criminality classes.

        eta:
            The encounter rate.

        attraction:
            The probability that a citizen who meets a more criminal one moves one class up.

        redemption:
            The probability that a citizen who meets a more honest one moves one class down.

    Examples:
        >>> from skcomplex.models.social.crowds import CriminalContagion
        >>> CriminalContagion('contagion', citizens='citizens').attraction
        0.3
    """

    citizens: str = role()
    classes: int = parameter(default=5, domain=Domain.POSITIVE)
    eta: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    attraction: float = parameter(default=0.3, domain=Domain.NONNEGATIVE)
    redemption: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the encounter of the citizens with themselves.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        table = contagion_table(int(self.classes), s.attraction, s.redemption)
        return [Encounter(s.citizens.f, s.citizens.f, s.eta, table)]


class Deterrence(Interaction):
    """Encounters of citizens with security forces, which move criminal citizens toward honesty.

    Args:
        label:
            The label of the interaction.

        citizens:
            The path of the distribution of citizens.

        security:
            The path of the distribution of the security forces over efficacy.

        classes:
            The number of criminality classes.

        forces:
            The number of efficacy classes of the security forces.

        deterrence:
            The rate at which the most effective agents deter a criminal citizen.

    Examples:
        >>> from skcomplex.models.social.crowds import Deterrence
        >>> Deterrence('deterrence', citizens='citizens', security='security').deterrence
        2.0
    """

    citizens: str = role()
    security: str = role()
    classes: int = parameter(default=5, domain=Domain.POSITIVE)
    forces: int = parameter(default=3, domain=Domain.POSITIVE)
    deterrence: float = parameter(default=2.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the encounter of the citizens with the security forces.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        pattern, table = deterrence_table(int(self.classes), int(self.forces))
        return [Encounter(s.citizens.f, s.security.f, pattern * s.deterrence, table)]


class Recruitment(Interaction):
    """Recruitment of security agents in proportion to the mean criminality, a macro-scale external action.

    The security forces grow at the rate `recruitment m share_k` in efficacy class `k`, where `m` is the mean
    criminality of the citizens, a moment of their distribution, and retire at the rate `retirement`.

    Args:
        label:
            The label of the interaction.

        citizens:
            The path of the distribution of citizens.

        security:
            The path of the distribution of the security forces over efficacy.

        classes:
            The number of criminality classes.

        recruitment:
            The recruitment rate per unit of mean criminality.

        retirement:
            The retirement rate.

    Examples:
        >>> from skcomplex.models.social.crowds import Recruitment
        >>> Recruitment('recruitment', citizens='citizens', security='security').retirement
        0.1
    """

    citizens: str = role()
    security: str = role()
    classes: int = parameter(default=5, domain=Domain.POSITIVE)
    recruitment: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)
    retirement: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the rate of the security forces.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        criminality = moment(s.citizens.f, 1, criminality_grid(int(self.classes)), normalized=True)
        return [Rate(s.security.f, s.recruitment * criminality * s.security.share - s.retirement * s.security.f)]


def _sizes_of(system: System) -> tuple[int, int]:
    """Return the number of criminality classes and of efficacy classes of a model."""
    sizes = {item.label: int(item.size or 0) for item in system.components}
    return sizes['citizens'], sizes['security']


def criminality_observables(classes: int, forces: int) -> list[Observable]:
    """Return the mean criminality, the criminal fraction, and the density of the security forces of the model.

    Args:
        classes:
            The number of criminality classes.

        forces:
            The number of efficacy classes of the security forces.

    Returns:
        observables:
            The observables, each reading the moments of a distribution.

    Examples:
        >>> from skcomplex.models.social.crowds import criminality_observables
        >>> [item.label for item in criminality_observables(5, 3)]
        ['mean_criminality', 'criminal_fraction', 'security_forces']
    """
    criminality = criminality_grid(classes)
    return [
        Observable('mean_criminality', partial(mean_criminality, criminality=criminality), group='citizens'),
        Observable('criminal_fraction', partial(criminal_fraction, criminality=criminality), group='citizens'),
        Observable('security_forces', partial(security_density, forces=forces), group='security'),
    ]


@register_model(
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
    """Return the kinetic model of criminality in a society and the security forces that act against it.

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
            The model, with the distributions `citizens` and `security` and the interactions `contagion`,
            `deterrence`, and `recruitment`. Its standard observables are the mean criminality, the criminal fraction,
            and the density of the security forces.

    Examples:
        >>> from skcomplex.models.social.crowds import criminality
        >>> [item.label for item in criminality().interactions]
        ['contagion', 'deterrence', 'recruitment']
    """
    citizens = np.zeros(classes)
    citizens[0], citizens[-1] = 1.0 - criminals, criminals
    return System(
        components=[
            Group('citizens', element=Citizens, size=classes, on='sites', values={'f': citizens}),
            Group(
                'security',
                element=SecurityAgents,
                size=forces,
                on='sites',
                space=LatticeSpace(forces, boundary='fixed'),
                values={'f': np.full(forces, security / forces), 'share': np.full(forces, 1.0 / forces)},
            ),
        ],
        interactions=[
            CriminalContagion(
                'contagion',
                citizens='citizens',
                classes=classes,
                attraction=attraction,
                redemption=redemption,
            ),
            Deterrence(
                'deterrence',
                citizens='citizens',
                security='security',
                classes=classes,
                forces=forces,
                deterrence=deterrence,
            ),
            Recruitment(
                'recruitment',
                citizens='citizens',
                security='security',
                classes=classes,
                recruitment=recruitment,
                retirement=retirement,
            ),
        ],
        space=LatticeSpace(classes, boundary='fixed'),
        time=ContinuousTime(),
    )

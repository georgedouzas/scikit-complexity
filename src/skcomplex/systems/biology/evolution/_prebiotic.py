"""Evolution systems: quasispecies and hypercycles.

Quasispecies evolve under mutation and selection, and hypercycles are made of catalytic molecules.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Sequence
from functools import partial
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import MapInteraction
from ....core.components.laws.utils import total
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.errors import ValidationError
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..common.populations import ReplicatorInteraction


class GenotypeClassElement(Element):
    """Share of a population some number of point mutations from the master sequence.

    Args:
        label:
            The label of the class.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial share.

    Examples:
        >>> from skcomplex.systems.biology.evolution import GenotypeClassElement
        >>> GenotypeClassElement('k1', x=0.1).x
        0.1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x


class MasterSequenceElement(GenotypeClassElement):
    """Share of the population with the fittest genome, in a sharp-peak landscape.

    In the landscape every other genome is less fit.

    The master sequence has fitness 1 and every other genome `1 - sigma`. A genome mutates at one locus with
    probability `u` per generation. The master sequence holds the parameters of the landscape that the maps of every
    class read.

    Args:
        label:
            The label of the class.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial share.

        u:
            The mutation rate per genome and generation.

        sigma:
            The selective disadvantage of the other genomes.

    Examples:
        >>> from skcomplex.systems.biology.evolution import MasterSequenceElement
        >>> MasterSequenceElement('master', u=0.2).u
        0.2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: float = 0.0,
        u: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual, x=x)
        self.u = u
        self.sigma = sigma


class MoleculeElement(Element):
    """Self-replicating molecule of a prebiotic soup.

    It has its concentration `x` and its own growth rate.

    Args:
        label:
            The label of the molecule or of the group of molecules.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial concentration.

        growth:
            The rate of replication without catalysis.

    Examples:
        >>> from skcomplex.systems.biology.evolution import MoleculeElement
        >>> MoleculeElement('molecule', growth=0.1).growth
        0.1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        growth: Annotated[object, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.growth = growth


class CatalysisInteraction(ReplicatorInteraction):
    """Catalysed replication of the molecules of a group.

    A common outflow keeps the total at 1.

    Molecule `i` replicates at the rate `f_i = lambda_i + omega sum_j T_ij x_j`, with its own growth `lambda_i` and the
    0 or 1 topology `T` of the catalytic links read as the weights of a total over the molecules. The outflow is the
    total replication, so the concentrations follow the replicator equation `dx_i/dt = x_i (f_i - sum_j x_j f_j)`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of molecules.

        omega:
            The catalytic rate of a link.

        topology:
            The catalytic links, 1 where molecule `j` catalyses molecule `i`.

    Examples:
        >>> from skcomplex.systems.biology.evolution import CatalysisInteraction
        >>> CatalysisInteraction('catalysis', target='soup', topology=((0, 1), (1, 0))).topology
        ((0, 1), (1, 0))
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        omega: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        topology: Sequence[Sequence[float]] = ((0.0, 1.0), (1.0, 0.0)),
    ) -> None:
        super().__init__(label, target=target)
        self.omega = omega
        self.topology = topology

    def _check_fitness(self) -> None:
        """Leave the fitness unchecked, since the catalysis derives it from its topology."""

    def _get_fitness(self, namespace: SimpleNamespace) -> Expression:
        """Return the catalysed growth `lambda_i + omega sum_j T_ij x_j` of a molecule.

        Args:
            namespace:
                The namespace of the symbols the fitness can use, with the namespace of the target.

        Returns:
            fitness:
                The fitness of the molecule.
        """
        molecule = namespace.target
        fitness: Expression = molecule.growth + namespace.omega * total(
            molecule.x,
            weights=np.asarray(self.topology, dtype=float),
        )
        return fitness


class EigenQuasispeciesSystem(BaseSystem):
    """The linear chain system of a population around a master sequence.

    It holds the shares of a population at every number of mutations from the master sequence.

    Below the error threshold, `u < sigma`, a quasispecies forms around the master sequence, whose share settles at `x0
    = 1 - u / sigma`. Above it the master sequence is lost, the error catastrophe. Back mutations and multiple mutations
    are neglected, so the chain holds only forward mutations, and the last class collects the far mutants. The system
    has the shares `master.x` and `k1.x` to `k<classes>.x`.

    Args:
        classes:
            The number of classes of mutants after the master sequence.

        u:
            The mutation rate per genome and generation.

        sigma:
            The selective disadvantage of the mutants.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.evolution import EigenQuasispeciesSystem
        >>> [element.label for element in EigenQuasispeciesSystem(2).prepare().system_.elements]
        ['master', 'k1', 'k2']
    """

    field = 'biology/evolution'
    topics = ('quasispecies', 'error threshold', 'discrete map', 'mutation', 'selection')
    references = ('Eigen 1971', 'Gros 2015')
    until = 100
    dt = 1

    def __init__(
        self,
        classes: int = 12,
        *,
        u: float = 0.3,
        sigma: float = 0.5,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.classes = classes
        self.u = u
        self.sigma = sigma

    @staticmethod
    def _get_mean_fitness(master: SimpleNamespace) -> Expression:
        """Return the mean fitness of a population around a master sequence.

        The mean fitness is `x0 + (1 - sigma) (1 - x0)`.
        """
        return master.x + (1 - master.sigma) * (1 - master.x)

    def _replicate_master(self, s: SimpleNamespace) -> Expression:
        """Return the next share of the master sequence, replicated without mutation.

        The share is `x0 (1 - u) / <W>`.
        """
        master = s.target
        return master.x * (1 - master.u) / self._get_mean_fitness(master)

    def _replicate_mutants(self, s: SimpleNamespace, *, first: bool, last: bool) -> Expression:
        """Return the next share of a class of mutants.

        The share is `(u W_{k-1} x_{k-1} + kept (1 - sigma) x_k) / <W>`.

        The sources are the class one mutation closer to the master sequence and the master sequence. The fitness of the
        source is 1 for the first class, whose source is the master sequence, and the last class keeps its own mutants.
        """
        source, master = s.sources
        u, sigma = master.u, master.sigma
        source_fitness = 1 if first else 1 - sigma
        kept = 1 if last else 1 - u
        gained = u * source_fitness * source.x + kept * (1 - sigma) * s.target.x
        return gained / self._get_mean_fitness(master)

    def _build_system(self) -> CompositeSystem:
        """Return the components of the shares of a population around a master sequence.

        The shares are at every number of mutations from the master sequence.
        """
        labels = ['master', *(f'k{k}' for k in range(1, self.classes + 1))]
        elements = [
            MasterSequenceElement('master', x=1.0, u=self.u, sigma=self.sigma),
            *(GenotypeClassElement(label) for label in labels[1:]),
        ]
        selection = MapInteraction('selection', target='master', state='x', function=self._replicate_master)
        mutations = [
            MapInteraction(
                f'mutation_{target}',
                target=target,
                state='x',
                function=partial(self._replicate_mutants, first=source == 'master', last=target == labels[-1]),
                sources=(source, 'master'),
            )
            for source, target in itertools.pairwise(labels)
        ]
        interactions = [selection, *mutations]
        return CompositeSystem(self.label, elements=elements, interactions=interactions)


class HypercyclesSystem(BaseSystem):
    """A hypercycle of self-replicating molecules.

    The catalysis of the network keeps a set of them alive.

    The growth rates rise as `lambda_i = alpha i`. In a `'cycle'`, every molecule catalyses the next one. Up to four
    molecules the hypercycle keeps all of them at a fixed point, and from five some members die out and the others
    settle at a fixed point. In a `'complete'` network every molecule catalyses every other, and about `sqrt(2 omega /
    alpha)` of the fastest replicators survive while the others die out. The system has the concentrations `soup.x` of
    the molecules, members of the group `soup`.

    Args:
        molecules:
            The number of molecules.

        alpha:
            The step of the autocatalytic growth rates.

        omega:
            The catalytic rate of a link.

        topology:
            `'cycle'` or `'complete'`.

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ValidationError:
            The topology is unknown.

    Examples:
        >>> from skcomplex.systems.biology.evolution import HypercyclesSystem
        >>> HypercyclesSystem(3).prepare().system_.interactions[0].topology
        ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    """

    field = 'biology/evolution'
    topics = ('hypercycle', 'autocatalysis', 'prebiotic', 'ode')
    references = ('Eigen and Schuster 1977', 'Gros 2015')
    until = 50.0
    dt = 0.5

    def __init__(
        self,
        molecules: int = 4,
        *,
        alpha: float = 0.1,
        omega: float = 1.0,
        topology: str = 'cycle',
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.molecules = molecules
        self.alpha = alpha
        self.omega = omega
        self.topology = topology

    def _build_system(self) -> CompositeSystem:
        """Return the components of a hypercycle of self-replicating molecules.

        The catalysis of the network keeps a set of them alive.
        """
        if self.topology == 'cycle':
            links = np.roll(np.eye(self.molecules), 1, axis=0)
        elif self.topology == 'complete':
            links = 1 - np.eye(self.molecules)
        else:
            error_msg = f'Unknown topology {self.topology!r}. Use cycle or complete.'
            raise ValidationError(error_msg)
        soup = GroupElement(
            'soup',
            element=MoleculeElement,
            size=self.molecules,
            values={
                'x': [1 / self.molecules] * self.molecules,
                'growth': [float(self.alpha * index) for index in range(1, self.molecules + 1)],
            },
        )
        catalysis = CatalysisInteraction(
            'catalysis',
            target='soup',
            omega=self.omega,
            topology=tuple(tuple(float(value) for value in row) for row in links),
        )
        return CompositeSystem(self.label, elements=[soup], interactions=[catalysis])

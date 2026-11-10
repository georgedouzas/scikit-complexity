# Quantum systems

This page builds a quantum system, such as a chain of spins, and follows its state in time or finds its spectrum. A
quantum system is a set of quantum subsystems whose state is one vector of their joint Hilbert space, the complex
vector space whose unit vectors are the possible states.

## A spectrum from the catalog

This section finds the two lowest energies of a chain of four spins, the Heisenberg chain of the catalog. The analysis
`operator_measures` diagonalises the generator of the system exactly, and `n_states` sets how many of the lowest
eigenvalues it returns.

```python
from skcomplex.core.analysis import operator_measures
from skcomplex.systems.physics.statistical_mechanics import QuantumHeisenbergSystem

spectrum = operator_measures(QuantumHeisenbergSystem(n_sites=4), n_states=2)
spectrum.energies.round(6).tolist()
```

The two lowest energies are -2.0 and -1.0, in units of the coupling.

The core names this machinery for its mathematics, linear operator dynamics on tensor products, while the physics
systems keep the words of physics. For example,
[`GeneratorLaw`](../core_components/laws.md#generatorlaw) takes a Hermitian operator, which physics calls a
Hamiltonian.

Every subsystem holds its state `psi` in the Hilbert space of a `HilbertSpec`, which names its operators. A subsystem
of your own declares the operators it needs. Typical subsystems have these operators.

- A spin has the components `x`, `y`, `z` and the ladder operators `plus` and `minus` of spin `j`.
- A qubit has the Pauli matrices.
- A mode has `annihilate`, `create`, and `number`, truncated at `quanta`.
- A particle on the sites of a lattice or a network has its `laplacian` and its `adjacency`.

A group of subsystems, such as a [`GroupElement`](../core_components/elements.md#groupelement) of spins on `'sites'`,
is the tensor product of the spaces of its members, in member order. The initial state of every subsystem is a basis
label, an index, or a vector.

## Operators and laws

This section writes the generator of a system from the operators of its subsystems. A spin one half is an
[`Element`](../core_components/elements.md#element) whose state `psi` lives in the two-dimensional space of a
`HilbertSpec` with the three spin components. Neighbouring spins on a ring of six sites exchange their spin with the
coupling `J`, the Heisenberg exchange `J (Sx Sx + Sy Sy + Sz Sz)`. A
[`NeighborInteraction`](../core_components/interactions.md#neighborinteraction) writes the exchange once for a bond and
returns it as a `GeneratorLaw`. The chain starts in the alternating state up, down, up, down, up, down, and the
observable `sz` records the expectation of `Sz` at every site.

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import CC

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import NeighborInteraction
from skcomplex.core.components.laws import GeneratorLaw
from skcomplex.core.components.laws.utils import local_operator
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import HilbertSpec, ObservableSpec, ParameterSpec, RoleSpec, StateSpec

PAULI = {
    'x': (np.array([[0, 0.5], [0.5, 0]]), True),
    'y': (np.array([[0, -0.5j], [0.5j, 0]]), True),
    'z': (np.array([[0.5, 0], [0, -0.5]]), True),
}
SPIN_HALF = HilbertSpec(2, PAULI, basis={'up': 0, 'down': 1})


class SpinElement(Element):
    def __init__(self, label, psi: Annotated[object, StateSpec(domain=CC)] = 'up', *, hilbert=SPIN_HALF):
        super().__init__(label)
        self.psi = psi
        self.hilbert = hilbert


class ExchangeInteraction(NeighborInteraction):
    def __init__(self, label, *, target: Annotated[str, RoleSpec()], J: Annotated[float, ParameterSpec()] = 1.0):
        super().__init__(label)
        self.target = target
        self.J = J

    def _get_laws(self, s):
        a, b = s.source.psi, s.target.psi
        exchange = sum(local_operator(a, name) * local_operator(b, name) for name in ('x', 'y', 'z'))
        return [GeneratorLaw(s.J * exchange)]


def sz(member):
    return local_operator(member.psi, 'z')


def sites(values):
    return values


chain = CompositeSystem(
    elements=[
        GroupElement(
            'chain', element=SpinElement, space=LatticeSpace(6), on='sites', values={'psi': ['up', 'down'] * 3}
        ),
    ],
    interactions=[ExchangeInteraction('exchange', target='chain')],
)
observables = [ObservableSpec('sz', sz, group='chain', reduce=sites)]
trajectory = simulate(chain, until=2.0, dt=0.5, observables=observables)
trajectory.get_values('sz').shape, round(float(np.ptp(trajectory.get_values('energy'))), 8)
```

The record of `sz` has the shape `(5, 6)`, five output times by six sites. The energy varies by 0.0 over the run, so
the unitary evolution conserves it exactly.

Laws name the operators of a subsystem with `local_operator(s.psi, name)`, from
`skcomplex.core.components.laws.utils`. Operators combine by sums, numbers, and symbolic coefficients of parameters
and time. `operator_product` and `dagger` keep the order of products and adjoints. Two laws build the dynamics.

- `GeneratorLaw(operator)` adds to the generator, with `ħ = 1`.
- [`DissipatorLaw(operator, rate)`](../core_components/laws.md#dissipatorlaw) adds a Lindblad dissipator, a term that
  lets the system lose energy or coherence to its surroundings, which makes the system open.

A `NeighborInteraction` is written once for a bond, with `s.source` and `s.target`. It acts on every pair of
neighbours of the space of the group that its one role binds, here the
[`LatticeSpace`](../core_components/spaces.md#latticespace) of six sites. Compiling checks that the generator is
Hermitian, and that the dimension stays within the `max_dimension` of the operator solver.

A whole-system initial state, such as an entangled vector or `'ground'`, is given to `simulate` as
`initial_state={'<system label>.state': value}`.

## Solvers, observables, and spectra

This section chooses how the state evolves, records the expectations of operators, and diagonalises the generator
exactly. `simulate` compiles a system of quantum elements to the operator form. Its state is stored as complex pairs
under `<system>.state`, a pure state for a closed system and a density matrix for an open one. Its solvers hand every
frame to QuTiP.

- `UnitarySolver` solves the Schrödinger equation.
- `LindbladSolver`, the default, solves the master equation.
- `JumpTrajectoriesSolver` draws quantum jumps from the random stream of the run.

An `ObservableSpec` whose expression is an operator records its expectation. One over a group gives one value per
member when its `reduce` keeps the values, as `sites` does above. Every quantum run also records its `energy`, and its
`purity` when it is open. Groups on a lattice stream the space-time image of their local expectations to Rerun.

`operator_measures(system, kind=...)` analyses quantum systems by its kind.

- `'spectrum'` and `'ground-state'` diagonalise the generator exactly, optionally in the sector of a conserved
  diagonal operator.
- `'purity'` and `'entanglement'` read trajectories.

`symbolic(system, kind='hamiltonian')` prints the Hamiltonian in operator notation.

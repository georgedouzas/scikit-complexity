"""Define the compiler of quantum systems, with their generators and dissipators."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from .....core.components.laws import BaseLaw
from .....core.engine.compile import OperatorCompiler
from .....core.engine.problems import BaseProblemSpec
from .....core.engine.types import Dissipators, Generators
from ._dissipator import DissipatorLaw
from ._generator import GeneratorLaw
from ._lindblad import LindbladSolver


class QuantumCompiler(OperatorCompiler):
    """A compiler of quantum systems into problems of the operator form.

    It adds every `GeneratorLaw` to the Hamiltonian and every `DissipatorLaw` to the dissipators of the master
    equation. A system runs with `LindbladSolver` unless it is given a solver. The Hilbert space of a subsystem names
    this compiler, so a system of quantum elements compiles with it by itself.

    Args:
        backend:
            The backend requested for the problem, which the operator form does not use.

        solver:
            The solver the problem runs on. A solver with a largest dimension of its own limits the Hilbert space.

        observables:
            The observables to compute at every frame, labelled by their path from the root.

        max_dimension:
            The largest dimension of the Hilbert space, for a solver without a largest dimension of its own.

    Examples:
        >>> from skcomplex.core.components.systems import CompositeSystem
        >>> from skcomplex.systems.physics.common.quantum import QuantumCompiler, QubitElement
        >>> compiler = QuantumCompiler().prepare(CompositeSystem(elements=[QubitElement('q')])).run()
        >>> compiler.problem_.dimensions, type(compiler.solver_).__name__
        ((2,), 'LindbladSolver')
    """

    def _add_law(self, law: BaseLaw, owner: str, generators: Generators, dissipators: Dissipators) -> None:
        """Add a generator law to the Hamiltonian and a dissipator law to the dissipators.

        Every other law raises an error.

        Args:
            law:
                The law.

            owner:
                The description of the component that returns the law.

            generators:
                The owner and the Hermitian operator of every generator term found so far.

            dissipators:
                The owner, the collapse operator, and the rate of every dissipator found so far.

        Raises:
            IncompatibleLawsError:
                The law is neither a generator nor a dissipator.
        """
        if isinstance(law, GeneratorLaw):
            generators.append((owner, law.operator))
        elif isinstance(law, DissipatorLaw):
            dissipators.append((owner, law.operator, law.rate))
        else:
            super()._add_law(law, owner, generators, dissipators)

    def _get_solver(self, problem: BaseProblemSpec) -> object:
        """Return `LindbladSolver`, the default solver of quantum systems.

        Args:
            problem:
                The compiled problem.

        Returns:
            solver:
                The master equation solver, which runs closed pure systems with the Schrödinger equation.
        """
        del problem
        return LindbladSolver()

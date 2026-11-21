"""Define the errors and warnings that the library raises."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


class SkcomplexError(Exception):
    """An error at the root of every error that the library raises.

    Every error of the library derives from it, so `except SkcomplexError` catches them all in one place, whether a
    model of an epidemic, a market, or a magnet failed. Its subclasses say what went wrong and how to fix it.

    Read more in the user guide, under Troubleshooting.

    See Also:
        ValidationError: Raised for a system or a value that breaks a rule.
        SimulationError: Raised for a run used in a state that forbids the call.

    Examples:
        >>> from skcomplex.core.errors import SkcomplexError, ValidationError
        >>> issubclass(ValidationError, SkcomplexError)
        True
    """


class NotPreparedError(SkcomplexError):
    """An error for an object that runs before it is prepared.

    A system simulates and an engine object runs only after `prepare`, which checks it and compiles its laws, so
    `system.simulate(until=1.0)` before `system.prepare()` raises it. Call `prepare` first. It stops a predator and prey
    model, a lattice of spins, or a market from running unchecked.

    Read more in the user guide, under Troubleshooting.

    See Also:
        ContextError: Raised for an object prepared without what it needs.

    Examples:
        >>> from skcomplex.systems.biology.ecology import LotkaVolterraSystem
        >>> LotkaVolterraSystem().simulate(until=1.0)
        Traceback (most recent call last):
            ...
        skcomplex.core.errors.NotPreparedError: The system 'system' must be prepared before it simulates...
    """


class ValidationError(SkcomplexError):
    """An error for a system, a value, or a choice of solver that is not valid.

    `prepare` and the compiler raise it when an argument breaks a rule of its class, or when a system does not fit its
    solver and backend, such as a lattice of an unknown shape. The message names the component and the fix. It catches
    a negative rate of growth, an unknown boundary of a grid, and a solver that cannot run a market.

    Read more in the user guide, under Troubleshooting.

    See Also:
        SkcomplexError: The root of every error of the library.
        LabelError: Raised for a label used twice.

    Examples:
        >>> from skcomplex.core.components.spaces import LatticeSpace
        >>> LatticeSpace((3,), geometry='hexagon').prepare()
        Traceback (most recent call last):
            ...
        skcomplex.core.errors.ValidationError: Unknown lattice geometry 'hexagon'...
    """


class ContextError(ValidationError):
    """An error for an object prepared without the context it needs.

    Some objects prepare on something else, such as a solver on the compiled problem of a system, so
    `RK4Solver().prepare()` raises it. Prepare the object inside its owner, or pass the context to `prepare`. It applies
    alike to the solver of a pendulum, a reaction, or a flock.

    Read more in the user guide, under Troubleshooting.

    See Also:
        NotPreparedError: Raised for an object that runs before it is prepared.

    Examples:
        >>> from skcomplex.core.engine.solvers import RK4Solver
        >>> RK4Solver().prepare()
        Traceback (most recent call last):
            ...
        skcomplex.core.errors.ContextError: A solver prepares on a compiled problem...
    """


class AlgebraicStateError(ValidationError):
    """An error for a rate law on a state that equations fix.

    A state fixed at every time by the equations of its system, such as the level held by `ConstraintLaw(x, 2.0)`, has
    no rate of its own, so a rate law on it raises this error. Remove the rate law, or declare the state with an order.
    It guards the voltages of a circuit, the balance of a chemical equilibrium, and the prices that clear a market.

    Read more in the user guide, under Troubleshooting.

    See Also:
        IncompatibleLawsError: Raised for laws that no single problem can hold.

    Examples:
        >>> from skcomplex.core.errors import AlgebraicStateError, ValidationError
        >>> issubclass(AlgebraicStateError, ValidationError)
        True
    """


class ConservationError(ValidationError):
    """An error for a conserved total or a fixed quantity that drifts too far in a run.

    A conserved total may change only by its flows, and a quantity that should stay fixed only within its tolerance, so
    a run with `invariants={'energy': 1e-3}` raises it when the energy drifts further. Use a smaller step, a tighter
    tolerance, or a solver that keeps the quantity. It watches the energy of a pendulum, the mass of a reaction, and the
    money of a closed economy.

    Read more in the user guide, under Troubleshooting.

    See Also:
        SolverError: Raised for a solver that fails to advance a problem.

    Examples:
        >>> from skcomplex.core.errors import ConservationError, ValidationError
        >>> issubclass(ConservationError, ValidationError)
        True
    """


class TensorShapeError(ValidationError):
    """An error for a matrix operation given an expression of the wrong shape.

    An operation on matrices reads a matrix of a given shape, so the divergence of a matrix given a single number raises
    it. Pass a matrix of expressions of the shape the operation reads. It guards the stresses of a solid, the flows of a
    fluid, and the order of a liquid crystal.

    Read more in the user guide, under Troubleshooting.

    See Also:
        TensorSymmetryError: Raised for a rate that breaks the symmetry of a matrix value.

    Examples:
        >>> from skcomplex.core.errors import TensorShapeError, ValidationError
        >>> issubclass(TensorShapeError, ValidationError)
        True
    """


class TensorSymmetryError(ValidationError):
    """An error for a rate law that breaks the symmetry of a matrix value.

    A matrix value declared symmetric, whose entry at `(i, j)` equals the one at `(j, i)`, cannot take a rate that is
    not, so an antisymmetric rate raises it. Write the rate with the same symmetry, such as with `symmetric_part`. It
    guards the stress of a solid, the strain of a sheet, and the order of a liquid crystal.

    Read more in the user guide, under Troubleshooting.

    See Also:
        TensorShapeError: Raised for a matrix operation given the wrong shape.

    Examples:
        >>> from skcomplex.core.errors import TensorSymmetryError, ValidationError
        >>> issubclass(TensorSymmetryError, ValidationError)
        True
    """


class RewritingError(ValidationError):
    """An error for a malformed rewriting system or a word that grows too long.

    A generation that rewrites a word or a tiling raises it when the result would pass its largest size, such as a word
    of algae that doubles beyond `max_size=4`. Raise the limit or run fewer generations. It bounds the growth of a
    plant, a sentence of a grammar, and a tiling of the plane.

    Read more in the user guide, under Troubleshooting.

    See Also:
        SpaceTooLargeError: Raised for a space that would hold too many sites.

    Examples:
        >>> from skcomplex.core.errors import RewritingError, ValidationError
        >>> issubclass(RewritingError, ValidationError)
        True
    """


class ParameterError(ValidationError):
    """An error for a parameter name or value that is not valid.

    A run reads parameters by name, so a schedule of a name the system does not have, or a value outside its range,
    raises it. Read the names from `get_params` and give a value the parameter accepts. It catches a misspelled rate of
    an epidemic, a negative mass of a pendulum, and a matrix of trade of the wrong size.

    Read more in the user guide, under Troubleshooting.

    See Also:
        ParameterShapeError: Raised for an array argument that no group shapes.

    Examples:
        >>> from skcomplex.core.errors import ParameterError, ValidationError
        >>> issubclass(ParameterError, ValidationError)
        True
    """


class ParameterShapeError(ParameterError):
    """An error for an array argument of an interaction that no group shapes.

    An array argument is a parameter when it is a vector, a square matrix of 2 or 3 rows, or an array with one axis per
    role whose group has that many members, so a 4 by 4 matrix of couplings on a single element raises it. Bind the
    roles of its axes to groups of those lengths. It guards the couplings of oscillators, the contacts of an epidemic,
    and the trade between the sectors of an economy.

    Read more in the user guide, under Troubleshooting.

    See Also:
        ParameterError: Raised for a parameter name or value that is not valid.

    Examples:
        >>> from skcomplex.core.errors import ParameterError, ParameterShapeError
        >>> issubclass(ParameterShapeError, ParameterError)
        True
    """


class LabelError(ValidationError):
    """An error for a label that is missing, used twice, or taken by another name.

    Every component needs a label unique among the components of its system, so two elements labelled `'a'` raise it.
    Rename one of the components. It keeps apart the species of an ecosystem, the firms of a market, and the bodies of
    a planetary system.

    Read more in the user guide, under Troubleshooting.

    See Also:
        RoleError: Raised for a role that names a missing element.

    Examples:
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.systems import System
        >>> System(elements=[Element('a', values={'x': 1.0}), Element('a', values={'x': 2.0})]).prepare()
        Traceback (most recent call last):
            ...
        skcomplex.core.errors.LabelError: Label 'a' is used by more than one child of the root system.
    """


class RoleError(ValidationError):
    """An error for a role of an interaction that names a missing element.

    A role binds an interaction to the path of an element, so `roles={'source': 'lynx'}` in a system with no lynx
    raises it. Bind the role to the path of an element of the system, such as `'forest/foxes'` for an element of a
    nested system. It catches a predator with no prey, a reaction with a missing species, and a trade with no buyer.

    Read more in the user guide, under Troubleshooting.

    See Also:
        RoleElementError: Raised for a role bound to an element it does not accept.
        LabelError: Raised for a label used twice.

    Examples:
        >>> from skcomplex.core.errors import RoleError, ValidationError
        >>> issubclass(RoleError, ValidationError)
        True
    """


class RoleElementError(RoleError):
    """An error for a role bound to an element it does not accept.

    A role accepts some kinds of element and reads some of their values, so a law that reads `n` of an element with only
    `x` raises it. Bind an element of the kind the role names, or give the element the value. It catches a growth law
    on a population with no size, a force on a body with no mass, and a vote by a voter with no opinion.

    Read more in the user guide, under Troubleshooting.

    See Also:
        RoleError: Raised for a role that names a missing element.

    Examples:
        >>> from skcomplex.core.errors import RoleElementError, RoleError
        >>> issubclass(RoleElementError, RoleError)
        True
    """


class TagError(ValidationError):
    """An error for tags of a system that disagree or that a solver does not support.

    The components of a system may need different times, or a solver may not run the kinds of law the system holds, so
    a solver of maps given rate laws raises it. Read the tags with `get_tags` and choose a solver that accepts them. It
    catches a stochastic epidemic sent to a deterministic solver and a lattice sent to a solver with no space.

    Read more in the user guide, under Troubleshooting.

    See Also:
        SolverError: Raised for a solver that does not fit a problem.
        IncompatibleLawsError: Raised for laws that no single problem can hold.

    Examples:
        >>> from skcomplex.core.errors import TagError, ValidationError
        >>> issubclass(TagError, ValidationError)
        True
    """


class SpaceError(ValidationError):
    """An error for a component, a vector, or a location that does not fit the space.

    A component may need a kind of space the system lacks, a vector value may have no space to give its length, or a
    location may lie outside the space, such as a triangular grid turned into a complex of faces. Give the system or the
    group the space the component needs. It catches agents in a park off the map, a flock with no box, and faces on a
    network that has no such links.

    Read more in the user guide, under Troubleshooting.

    See Also:
        SpaceTooLargeError: Raised for a space that would hold too many sites.

    Examples:
        >>> from skcomplex.core.components.spaces import CellComplexSpace, LatticeSpace
        >>> CellComplexSpace(LatticeSpace((3, 3), geometry='triangular')).prepare()
        Traceback (most recent call last):
            ...
        skcomplex.core.errors.SpaceError: A cell complex of a lattice takes a square lattice...
    """


class SpaceTooLargeError(SpaceError):
    """An error for a space that would hold too many sites.

    A space that would hold more sites than its limit, or values that would overflow their integer type, raises it,
    such as a grid of a million by a million cells. Use a smaller space, or raise the limit when the memory allows. It
    bounds a fine map of a forest, a large lattice of spins, and a dense mesh of a machine part.

    Read more in the user guide, under Troubleshooting.

    See Also:
        PhaseSpaceTooLargeError: Raised for a phase space past its limit.
        SpaceError: Raised for a component that does not fit its space.

    Examples:
        >>> from skcomplex.core.errors import SpaceError, SpaceTooLargeError
        >>> issubclass(SpaceTooLargeError, SpaceError)
        True
    """


class PhaseSpaceTooLargeError(SpaceTooLargeError):
    """An error for a phase space that would hold more sites than its limit.

    A phase space holds one site per position and cell of every extra axis, so 100 positions with 100 velocities pass
    a `limit` of 10. Use fewer cells on the extra axes or the positions, or raise `limit`. It bounds the gas molecules
    of kinetic theory, the swimming bacteria of a colony, and the ages of a population.

    Read more in the user guide, under Troubleshooting.

    See Also:
        SpaceTooLargeError: Raised for any space that would hold too many sites.

    Examples:
        >>> from skcomplex.core.components.spaces import AxisSpec, LatticeSpace, PhaseSpace
        >>> PhaseSpace(LatticeSpace(100), [AxisSpec('v', 100, 0.0, 1.0)], limit=10).prepare()
        Traceback (most recent call last):
            ...
        skcomplex.core.errors.PhaseSpaceTooLargeError: The phase space would take 10000 sites, above its limit of 10.
    """


class UnregisteredLawError(ValidationError):
    """An error for a law of a class that the engine cannot compile.

    The engine compiles the law classes it knows, so a law of a new class raises it even when it derives from a law of
    the core. Write the law with a class the core provides, such as `RateLaw` or `UpdateLaw`. It applies alike to the
    laws of a reaction, an epidemic, or a flock.

    Read more in the user guide, under Troubleshooting.

    See Also:
        IncompatibleLawsError: Raised for laws that no single problem can hold.

    Examples:
        >>> from skcomplex.core.errors import UnregisteredLawError, ValidationError
        >>> issubclass(UnregisteredLawError, ValidationError)
        True
    """


class IncompatibleLawsError(ValidationError):
    """An error for a system that mixes kinds of law that no single problem can hold.

    A rate law in continuous time and an agent rule in discrete time, for example, cannot form one problem. Split the
    system into nested systems, or write the laws in one kind. It catches a growing population of cells whose agents
    step while their nutrients flow, and a market whose traders step while prices flow.

    Read more in the user guide, under Troubleshooting.

    See Also:
        TagError: Raised for tags that disagree or that a solver does not support.

    Examples:
        >>> from skcomplex.core.errors import IncompatibleLawsError, ValidationError
        >>> issubclass(IncompatibleLawsError, ValidationError)
        True
    """


class BackendError(SkcomplexError):
    """An error for a backend that is unknown or cannot write a problem.

    A backend writes some kinds of problem only, so a system with no laws given `JaxBackend()`, which writes rate
    equations and maps, raises it. Choose a backend such as `NumpyBackend()`, or one that accepts the laws of the
    system. It applies alike to an epidemic, a lattice of spins, or a network of neurons.

    Read more in the user guide, under Troubleshooting.

    See Also:
        SolverError: Raised for a solver that does not fit a problem.

    Examples:
        >>> from skcomplex.core.errors import BackendError, SkcomplexError
        >>> issubclass(BackendError, SkcomplexError)
        True
    """


class PipelineError(SkcomplexError):
    """An error for steps of a compilation that cannot be put in order.

    The compiler runs its steps in an order that gives each what it reads, so a step that reads what no step writes, or
    steps that wait on each other, raise it. Add the step that writes it, or break the cycle. It concerns those who
    extend the compiler for a new kind of model, in any field.

    Read more in the user guide, under Troubleshooting.

    See Also:
        UnregisteredLawError: Raised for a law the engine cannot compile.

    Examples:
        >>> from skcomplex.core.errors import PipelineError, SkcomplexError
        >>> issubclass(PipelineError, SkcomplexError)
        True
    """


class SolverError(SkcomplexError):
    """An error for a solver that does not fit a problem or fails to advance it.

    A solver runs some kinds of problem only, and it may fail on a stiff problem whose values change at very different
    speeds, as a map solver given rate laws does. Choose a solver that accepts the problem, or a smaller step or a
    solver for stiff problems. It catches a stiff chemical network, a pendulum with too large a step, and an epidemic
    sent to the wrong solver.

    Read more in the user guide, under Troubleshooting.

    See Also:
        InfeasibleError: Raised for a linear program with no solution.
        BackendError: Raised for a backend that cannot write a problem.

    Examples:
        >>> from skcomplex.core.errors import SkcomplexError, SolverError
        >>> issubclass(SolverError, SkcomplexError)
        True
    """


class InfeasibleError(SolverError):
    """An error for a linear program with no solution, when its law asks to raise.

    A linear program has no solution when its constraints contradict, such as a demand above the supply of a network.
    Loosen the constraints, or set `infeasible='hold'` to keep the last values instead of raising. It catches a
    metabolism that cannot grow, a supply chain that cannot deliver, and a power grid that cannot meet its load.

    Read more in the user guide, under Troubleshooting.

    See Also:
        SolverError: Raised for any solver that fails to advance a problem.

    Examples:
        >>> from skcomplex.core.errors import InfeasibleError, SolverError
        >>> issubclass(InfeasibleError, SolverError)
        True
    """


class SimulationError(SkcomplexError):
    """An error for a simulation used in a state that does not allow the call.

    Some calls need a run behind them, so reading the results of a simulation before it has run raises it. Call `run`
    first, as the message says. It applies alike to a run of a reaction, a market, or a flock.

    Read more in the user guide, under Troubleshooting.

    See Also:
        CapacityError: Raised for a group that runs out of rows.
        NotPreparedError: Raised for an object that runs before it is prepared.

    Examples:
        >>> from skcomplex.core.errors import SimulationError, SkcomplexError
        >>> issubclass(SimulationError, SkcomplexError)
        True
    """


class CapacityError(SimulationError):
    """An error for a change of a network that needs more rows than a group holds.

    A group given a `capacity` holds that many rows for its members, so a request to add more raises it, with the path
    of the group and its number of rows. A group without a capacity grows as a Python list does and never raises it.
    Raise the capacity, or leave it out. It bounds a growing social network, a dividing population of cells, and a
    spreading web of roads.

    Read more in the user guide, under Troubleshooting.

    Args:
        message:
            The message.

        path:
            The path of the full group, such as `'nodes'`.

        rows:
            The number of rows the group holds.

    See Also:
        SimulationError: Raised for a run used in a state that forbids the call.

    Examples:
        >>> from skcomplex.core.errors import CapacityError
        >>> CapacityError('The group is full.', path='nodes', rows=8).rows
        8
    """

    def __init__(self, message: str, *, path: str = '', rows: int = 0) -> None:
        """Keep the path of the full group and its number of rows.

        Args:
            message:
                The message.

            path:
                The path of the full group, such as `'nodes'`.

            rows:
                The number of rows the group holds.
        """
        super().__init__(message)
        self.path = path
        self.rows = rows


class ExperimentError(SkcomplexError):
    """An error for an experiment that cannot run its tasks.

    An experiment runs many simulations in worker processes, and a worker may die, such as for lack of memory. Use fewer
    workers or a smaller design. It concerns the seeds of an epidemic, the sweep of a climate model, and the trials of a
    market alike.

    Read more in the user guide, under Troubleshooting.

    See Also:
        SimulationError: Raised for a run used in a state that forbids the call.

    Examples:
        >>> from skcomplex.core.errors import ExperimentError, SkcomplexError
        >>> issubclass(ExperimentError, SkcomplexError)
        True
    """


class AnalysisError(SkcomplexError):
    """An error for an analysis that does not apply to the system or the data.

    An analysis reads a kind of data, such as a series or a network, so it raises this error when the system or the
    data are of another kind. Use `applicable_analyses` to find the analyses that apply. It catches an entropy of a
    network, a spectrum of a single number, and a network measure of a price series.

    Read more in the user guide, under Troubleshooting.

    See Also:
        EnumerationLimitError: Raised for an exact count that runs out of time.

    Examples:
        >>> from skcomplex.core.errors import AnalysisError, SkcomplexError
        >>> issubclass(AnalysisError, SkcomplexError)
        True
    """


class EnumerationLimitError(AnalysisError):
    """An error for an exact enumeration that does not finish within its time limit.

    An analysis that lists every case, such as every loop of a reaction network, raises it when the list takes longer
    than its limit. Raise the time limit, or use a smaller system or a sampling method. It bounds the loops of a food
    web, the pathways of a metabolism, and the cycles of a trade network.

    Read more in the user guide, under Troubleshooting.

    See Also:
        AnalysisError: Raised for an analysis that does not apply.

    Examples:
        >>> from skcomplex.core.errors import AnalysisError, EnumerationLimitError
        >>> issubclass(EnumerationLimitError, AnalysisError)
        True
    """


class MissingExtraError(SkcomplexError):
    """An error for a class that needs an optional extra that is not installed.

    Some classes read a package that an optional extra installs, such as the relativity extra, so they raise this error
    without it. Install the extra that the message names. It applies to the curved spacetime of relativity and to other
    models with packages of their own.

    Read more in the user guide, under Troubleshooting.

    See Also:
        RegistryError: Raised for a name registered twice.

    Examples:
        >>> from skcomplex.core.errors import MissingExtraError, SkcomplexError
        >>> issubclass(MissingExtraError, SkcomplexError)
        True
    """


class RegistryError(SkcomplexError):
    """An error for a name registered twice or a registration that is not valid.

    A name registered twice with different meanings, or a registration that misses what it needs, raises it. Choose a
    new name, or complete the registration. It concerns those who add new kinds of law, solver, or visual, in any
    field.

    Read more in the user guide, under Troubleshooting.

    See Also:
        PipelineError: Raised for steps of a compilation that cannot be ordered.

    Examples:
        >>> from skcomplex.core.errors import RegistryError, SkcomplexError
        >>> issubclass(RegistryError, SkcomplexError)
        True
    """


class BackendWarning(UserWarning):
    """A warning that part of a problem runs on another backend or in plain Python.

    A backend that cannot write a law hands it to another backend, or a compiled function falls back to Python, so the
    run still works but more slowly. A law given as a Python function in a reaction, a swarm, or a market sends this
    warning under `NumbaBackend()`.

    Read more in the user guide, under Troubleshooting.

    See Also:
        BackendError: Raised for a backend that cannot write a problem at all.

    Examples:
        >>> from skcomplex.core.errors import BackendWarning
        >>> issubclass(BackendWarning, UserWarning)
        True
    """


class UnobservedStatesWarning(UserWarning):
    """A warning that an analysis drops states with no observed transitions.

    An analysis that counts the moves between states drops the states it never sees leave, so its result covers fewer
    states. Run longer or from more initial states so every state is visited. It shows in the moods of a market, the
    folds of a protein, and the stages of a disease.

    Read more in the user guide, under Troubleshooting.

    See Also:
        AnalysisError: Raised for an analysis that does not apply.

    Examples:
        >>> from skcomplex.core.errors import UnobservedStatesWarning
        >>> issubclass(UnobservedStatesWarning, UserWarning)
        True
    """


class ValidationWarning(UserWarning):
    """A warning of a problem that does not stop compilation.

    The check of a system reports problems that let it compile, such as a value no law changes, with a message that
    names the problem and its fix. It flags a forgotten rate of a predator, an unused price of a market, and an idle
    spin of a lattice.

    Read more in the user guide, under Troubleshooting.

    See Also:
        ValidationError: Raised for a problem that stops compilation.

    Examples:
        >>> from skcomplex.core.errors import ValidationWarning
        >>> issubclass(ValidationWarning, UserWarning)
        True
    """

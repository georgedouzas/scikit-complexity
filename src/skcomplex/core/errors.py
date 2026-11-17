"""Define the errors and warnings that the library raises."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


class SkcomplexError(Exception):
    """An error at the root of every error that the library raises.

    Catch it to handle any error of the library in one place.
    """


class NotPreparedError(SkcomplexError):
    """An error for an object that runs before it is prepared.

    `run` raises it when `prepare` has not been called. Call `prepare` first, or use `simulate`, which prepares the
    system itself.
    """


class ValidationError(SkcomplexError):
    """An error for a system, a value, or a choice of solver that is not valid.

    `prepare` and the compiler raise it when an argument breaks a rule of its class, or a system does not fit its solver
    and backend. The message names the component and the fix.
    """


class ContextError(ValidationError):
    """An error for an object prepared without the context it needs.

    For example, a solver is prepared without its problem. Prepare the object inside its owner, or pass the context to
    `prepare`.
    """


class AlgebraicStateError(ValidationError):
    """An error for a law that integrates a state fixed by equations.

    An algebraic state is fixed at every time by the equations of its system, so a rate law on it has no meaning. Remove
    the rate law, or declare the state with an order.
    """


class ConservationError(ValidationError):
    """An error for a conserved quantity or invariant that changes too much.

    A conserved quantity may change only by its flows, and an invariant only within its tolerance. A run raises it when
    either drifts further. Use a smaller step, a tighter tolerance, or a solver that keeps the quantity.
    """


class TensorShapeError(ValidationError):
    """An error for a tensor operation given an expression of the wrong shape.

    An example is the tensor divergence of a scalar. Pass a matrix of expressions of the shape the operation reads.
    """


class TensorSymmetryError(ValidationError):
    """An error for a rate law that breaks the symmetry of a tensor state.

    A tensor state declared symmetric, for example, cannot take an antisymmetric rate. Write the rate with the same
    symmetry, such as with `symmetric_part`.
    """


class RewritingError(ValidationError):
    """An error for a malformed rewriting system or a word that grows too long.

    A generation of a string rewriting system or a tiling raises it when it would exceed its maximum size. Raise the
    limit or run fewer generations.
    """


class ParameterError(ValidationError):
    """An error for a parameter name or value that is not valid.

    The name is not part of the problem, or the value is not valid for its system or matrix. Read the parameter names
    from `get_params` and give a value in the domain of the parameter.
    """


class LabelError(ValidationError):
    """An error for a label that is missing, duplicated, or collides with a name.

    Every component needs a label that is unique among the components of its system. Rename one of the components.
    """


class RoleError(ValidationError):
    """An error for a role of an interaction that names a missing element.

    Bind the role to the path of an element of the system, such as `'forest/foxes'` for an element of a nested system.
    """


class RoleElementError(RoleError):
    """An error for a role bound to an element it does not accept.

    The element is of a kind the role does not accept, or lacks a state the role reads. Bind an element of the kind the
    role names, or give the element the state.
    """


class TagError(ValidationError):
    """An error for tags of a system that disagree or that a solver does not support.

    The components of a system may disagree on a capability tag, such as two components that need different times, or a
    solver or backend may not support the tags. Read the tags with `get_tags` and choose a solver whose tags accept
    them.
    """


class SpaceError(ValidationError):
    """An error for a component, a vector, or a location that does not fit the space.

    A component needs a kind of space the system lacks, a vector state has no space to give its dimension, or a location
    lies outside its space. Give the system or the group the space the component needs.
    """


class SpaceTooLargeError(SpaceError):
    """An error for a space that would hold too many sites.

    It would hold more sites than its limit, or its elements would overflow their integer type. Use a smaller space, or
    raise the limit when the memory allows.
    """


class PhaseSpaceTooLargeError(SpaceTooLargeError):
    """An error for a phase space that would hold more sites than its limit.

    Use fewer cells on the class axes or the position lattice, or raise `limit`.
    """


class UnregisteredLawError(ValidationError):
    """An error for a law of a class that the engine has no form for.

    No form of the engine takes its class, even when it subclasses a core law, so no compiler can lower it. Write the
    law with a class the core provides, such as `RateLaw` or `UpdateLaw`.
    """


class IncompatibleLawsError(ValidationError):
    """An error for a system that mixes law kinds no single form supports.

    For example, a rate law in continuous time and an agent rule in discrete time cannot form one problem. Split the
    system into partitions, or write the laws in one kind.
    """


class BackendError(SkcomplexError):
    """An error for a backend that is not registered or cannot compile a problem.

    Choose a registered backend, such as `'numpy'`, or one whose tags accept the laws of the system.
    """


class PipelineError(SkcomplexError):
    """An error for passes of a compilation that cannot be ordered.

    A pass reads a field that no pass writes, or the passes form a cycle. Add the pass that writes the field, or break
    the cycle.
    """


class SolverError(SkcomplexError):
    """An error for a solver that does not fit a problem or fails to advance it.

    Choose a solver whose tags accept the problem, or a smaller step or a stiff solver when it fails to advance.
    """


class InfeasibleError(SolverError):
    """An error for a linear program with no solution, when its law asks to raise.

    Loosen the constraints, or set `infeasible='hold'` to keep the last values instead of raising.
    """


class SimulationError(SkcomplexError):
    """An error for a simulation used in a state that does not allow the operation.

    For example, the results of a simulation are read before it has run. Call `run` first, as the message says.
    """


class CapacityError(SimulationError):
    """An error for a topology change that needs more rows than a group holds.

    A variable group on a network holds a fixed number of rows for its members, and a request to add more raises it.
    Raise `capacity` of the network space.
    """


class ExperimentError(SkcomplexError):
    """An error for an experiment that cannot run its tasks.

    Its worker processes may have died, for example for lack of memory. Use fewer workers or a smaller design.
    """


class AnalysisError(SkcomplexError):
    """An error for an analysis that does not apply to the system or the data.

    The tags of the system or the kind of data do not match what the analysis needs. Use `applicable_analyses` to find
    the analyses that apply.
    """


class EnumerationLimitError(AnalysisError):
    """An error for an exact enumeration that does not finish in its time limit.

    Raise the time limit, or use a smaller system or a sampling method.
    """


class MissingExtraError(SkcomplexError):
    """An error for a class that needs an optional extra that is not installed.

    Install the extra that the message names.
    """


class RegistryError(SkcomplexError):
    """An error for a name registered twice or a registration that is not valid.

    A name is registered twice with different meanings, or a registration misses what it needs. Choose a new name, or
    complete the registration.
    """


class BackendWarning(UserWarning):
    """A warning that part of a problem runs on another backend or in Python.

    The other backend is not the one requested, or a compiled kernel falls back to Python and runs slower.
    """


class UnobservedStatesWarning(UserWarning):
    """A warning that an analysis drops states with no observed transitions.

    Run longer or from more initial states so that every state is visited.
    """


class ValidationWarning(UserWarning):
    """A warning of a problem that does not stop compilation.

    The message names the problem and its fix.
    """

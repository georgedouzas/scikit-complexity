"""Define the compiled problem that solvers evaluate."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression, I
from sage.modules.free_module_element import FreeModuleElement

from ...core.elements import GroupElement
from ...core.laws import BaseLaw
from ...core.spaces import LatticeSpace
from ...types import Array, VectorFunction
from .._utils import declarations_of
from ._accelerated import Accelerated
from ._base import BaseKernel, BaseProblem
from ._breakpoints import _Breakpoints
from ._delay import _Delay
from ._draw_kernel import DrawKernel
from ._event_kernel import EventKernel
from ._fractional_rows import FractionalRows
from ._jump_channels import _JumpChannels
from ._layout import Layout
from ._reduction_kernel import ReductionKernel
from ._variable import Variable
from ._variable_kind import VariableKind
from ._wiener_block import WienerBlock

if TYPE_CHECKING:
    from ._jacobian_kernel import JacobianKernel


@dataclass(frozen=True)
class Problem(BaseProblem):
    """System compiled to one mathematical form with flat state and parameter vectors.

    Only the functions of the problem's form are set. `drift`, `diffusion`, and `noise_rows` belong to the ODE and SDE
    forms, `propensities` and `jumps` to the jump process form, and `update` to the discrete map form. `boundary`
    belongs to any form whose system has boundary laws.

    Args:
        form:
            The mathematical form.

        states:
            The layout of the state vector.

        parameters:
            The layout of the parameter vector.

        initial_state:
            The initial state vector.

        initial_parameters:
            The initial parameter vector.

        time:
            The time symbol.

        symbols:
            A mapping from dotted name to the symbol of a scalar or the column of symbols of a vector. The symbols
            of a group stand for one member.

        equations:
            A mapping from state component name to its symbolic drift, or to its next value for the discrete form.
            Group components are named with the member index `i`. Components with a callable law, a sum over group
            members, or a sum over pairs are left out.

        equation_symbols:
            A mapping from the names of `equations` to the symbol of the component.

        has_callables:
            Whether any law is a callable.

        has_group_sums:
            Whether any law sums over group members or pairs.

        elements:
            The compiled elements and groups in state vector order.

        interactions:
            The compiled interactions in tree order.

        system:
            The system the problem was compiled from.

        agents:
            The agent elements and groups of an agent-based problem.

        model_rules:
            The rules of interactions of an agent-based problem, with the path of their interaction.

        agent_space:
            The space the agents of an agent-based problem live in, or `None`.

        companions:
            A mapping from the name of every companion state to its kind, `'derivative'` or `'lag'`, the name of the
            state it belongs to, and the order of the derivative or the number of steps back.

        partitions:
            The partitions of a co-simulation problem, in system order.

        delays:
            The delayed states that laws read, which solvers fill from the run's history.

        events:
            The discrete events, with their guard and reset functions.

        breakpoints:
            The times where time-dependent inputs jump, at which solvers restart.

        history:
            The values of states before the start, by dotted state path, as numbers or functions of time.

        lattice:
            The steps that change the states of sites, such as the local rules of a lattice problem, in the order
            they apply within a step.

        linear:
            The linear terms with constant coefficients of the rates of fields, which implicit and exponential solvers
            treat exactly while the rest of the rates is the nonlinear remainder.

        volumes:
            The conservation laws of finite-volume fields, whose flux differences the drift includes.

        constrained:
            The state indices held at their initial values, the fixed boundary nodes of finite element fields.

        reductions:
            The reductions that laws read, in the order they are evaluated before the kernels.

        jacobian:
            The Jacobian of the drift of an ODE problem, derived from its laws: its sparsity pattern and, for laws with
            a symbolic form, its exact value. `None` for other forms.

        stable_step:
            The largest step of explicit fixed-step solvers for which the diffusion of fields is stable,
            `h^2 / (degree * D)`, or `None` when the system has no diffusion with numeric coefficients.

        diffusive_limits:
            The stability limits of the diffusion of finite-volume fields, which depend on the parameters.

        drift:
            The function `f(t, x, p)` that returns the time derivative or the drift.

        diffusion:
            The function `g(t, x, p)` that returns every noise amplitude, one per state and process it couples.

        noise_rows:
            The state index each amplitude of `diffusion` drives.

        noise_columns:
            The Wiener process each amplitude of `diffusion` draws, which several amplitudes share for shared noise.

        n_processes:
            The number of standard normal increments drawn per step.

        noise_blocks:
            The blocks of correlated processes, whose increments their covariance factor scales.

        propensities:
            The function `a(t, x, p)` that returns the rate of every jump channel.

        jumps:
            The channels of every jump law.

        update:
            The function `F(t, x, p)` that returns the next state vector, or `F(t, x, p, rng)` for a stochastic map.

        boundary:
            The function `B(t, x, p)` that returns the state after the boundary laws.

        draws:
            The random values the update draws at every step, or `None` for a deterministic map.

        matrices:
            The transition and emission matrices of the Markov laws, checked when their parameters change.

        renewals:
            The renewal laws, whose members renew on their own clocks.

        fractional:
            The noise rows that fractional noise laws drive, with their Hurst indices.

        orders:
            The order of the Caputo derivative that the drift gives for every state component of a problem of the
            fractional form, 1 for the ordinary states, and empty for the other forms.

        transfer_network:
            The quantities, flows, objectives, and feedback terms of the system, or `None` for partitioned and Mesa
            problems.

        backend:
            The name of the backend that produced the numeric functions.

        backends:
            The backend that runs every part of the problem: its form by the form's name, and the kernels of every law
            kind by the kind's name, `'python'` for kernels that a compiling backend runs in Python.

        accelerated:
            The compiled stepper loops, or `None` when the backend has none for this problem.

        topology:
            The topology requests of the laws and the networks they change, which the stepper applies between steps,
            or `None` for a problem without them.
    """

    equations: Mapping[str, Expression]
    equation_symbols: Mapping[str, Expression]
    has_callables: bool
    has_group_sums: bool
    model_rules: tuple[tuple[str, BaseLaw], ...] = ()
    delays: tuple[_Delay, ...] = ()
    events: tuple[EventKernel, ...] = ()
    breakpoints: _Breakpoints = field(default_factory=_Breakpoints)
    history: Mapping[str, Any] = field(default_factory=dict)
    lattice: tuple[BaseKernel, ...] = field(default=(), repr=False)
    stable_step: float | None = None
    diffusive_limits: tuple[Any, ...] = field(default=(), repr=False)
    linear: tuple[Any, ...] = field(default=(), repr=False)
    volumes: tuple[Any, ...] = field(default=(), repr=False)
    constrained: npt.NDArray[np.intp] = field(default_factory=lambda: np.zeros(0, dtype=np.intp), repr=False)
    reductions: tuple[ReductionKernel, ...] = field(default=(), repr=False)
    jacobian: 'JacobianKernel | None' = field(default=None, repr=False)
    drift: VectorFunction | None = field(default=None, repr=False)
    diffusion: VectorFunction | None = field(default=None, repr=False)
    noise_rows: npt.NDArray[np.intp] = field(default_factory=lambda: np.zeros(0, dtype=np.intp), repr=False)
    noise_columns: npt.NDArray[np.intp] = field(default_factory=lambda: np.zeros(0, dtype=np.intp), repr=False)
    n_processes: int = 0
    noise_blocks: tuple[WienerBlock, ...] = field(default=(), repr=False)
    propensities: VectorFunction | None = field(default=None, repr=False)
    jumps: tuple[_JumpChannels, ...] = field(default=(), repr=False)
    update: Callable[..., Array] | None = field(default=None, repr=False)
    boundary: VectorFunction | None = field(default=None, repr=False)
    draws: DrawKernel | None = field(default=None, repr=False)
    fractional: tuple[FractionalRows, ...] = field(default=(), repr=False)
    orders: Array = field(default_factory=lambda: np.zeros(0), repr=False)

    backend: str = 'numpy'
    backends: Mapping[str, str] = field(default_factory=dict)
    accelerated: Accelerated | None = field(default=None, repr=False)

    def _flatten_symbols(self, layout: Layout) -> list[Expression]:
        """Return the symbols of the entries of a layout as a flat list."""
        return [item for entry in layout.entries for item in self._real_components_of(self.symbols[entry.name])]

    @property
    def adaptive_parameters(self) -> dict[str, Variable]:
        """Declarations of the adaptive parameters of the elements and groups, by dotted state name."""
        found = {}
        for compiled in self.elements:
            component = compiled.element
            kind = component.element if isinstance(component, GroupElement) else type(component)
            for name, declaration in declarations_of(kind, VariableKind.STATE).items():
                if declaration.adaptive and name in compiled.states:
                    found[compiled.states[name]] = declaration
        return found

    @property
    def stochastic(self) -> bool:
        """Whether the update draws random values, so it takes the random number generator."""
        return self.draws is not None

    @property
    def state_symbols(self) -> list[Expression]:
        """Symbols of the state components of single elements and of one member of every group."""
        return self._flatten_symbols(self.states)

    @property
    def parameter_symbols(self) -> list[Expression]:
        """Symbols of the parameter components of interactions, single elements, and one member of every group."""
        return self._flatten_symbols(self.parameters)

    def max_speed(self, x: Array, p: Array) -> float | None:
        """Return the largest wave speed of the conservation laws, or `None` without them.

        Args:
            x:
                The state vector.

            p:
                The parameter vector.

        Returns:
            speed:
                The largest wave speed over the cells, the dimensions, and the finite-volume fields.
        """
        if not self.volumes:
            return None
        return max(volume.max_speed(x, p) for volume in self.volumes)

    def diffusive_step(self, p: Array) -> float | None:
        """Return the largest stable explicit step for the diffusion of finite-volume fields, or `None` without it.

        Args:
            p:
                The parameter vector.

        Returns:
            step:
                The smallest limit over the finite-volume fields at the parameter values.
        """
        if not self.diffusive_limits:
            return None
        return min(limit.step(p) for limit in self.diffusive_limits)

    def lattice_shape(self, name: str) -> tuple[int, ...] | None:
        """Return the lattice shape of a state that lattice laws read or change, or `None` for other states.

        Args:
            name:
                The dotted name of the state.

        Returns:
            shape:
                The number of sites along every dimension.
        """
        for step in self.lattice:
            for state in step.targets:
                if state.name == name:
                    return state.shape
        owner = name.rpartition('.')[0]
        for element in self.elements:
            space = getattr(element.element, 'space', None)
            if element.path == owner and isinstance(space, LatticeSpace):
                return space.shape_  # type: ignore[union-attr]
        return None

    def draw_increments(self, rng: np.random.Generator, scale: float) -> Array:
        """Return the Wiener increment that every noise amplitude multiplies, drawn for one step.

        The increments of the processes are normal with the standard deviation `scale`, correlated within every block
        by its covariance factor. Amplitudes that share a process take its increment.

        Args:
            rng:
                The random number generator of the run.

            scale:
                The square root of the step.

        Returns:
            increments:
                One increment per noise amplitude, in the order of `noise_rows`.
        """
        increments = rng.normal(0.0, scale, size=self.n_processes)
        for block in self.noise_blocks:
            part = slice(block.start, block.start + block.size)
            increments[part] = block.factor @ increments[part]
        return increments[self.noise_columns]

    @property
    def stoichiometry(self) -> Array:
        """Increments of the state vector, one row per jump channel."""
        n_channels = sum(channels.size for channels in self.jumps)
        matrix = np.zeros((n_channels, self.states.size))
        for channels in self.jumps:
            rows = channels.offset + np.arange(channels.size)
            for targets, increment in zip(channels.targets, channels.increments, strict=True):
                kept = targets >= 0
                np.add.at(matrix, (rows[kept], targets[kept]), increment)
        return matrix

    @staticmethod
    def _real_components_of(value: Any) -> list[Any]:  # noqa: ANN401
        """Return the components of a value with every complex component split into its real and imaginary parts.

        Args:
            value:
                A vector, an expression, or a number.

        Returns:
            components:
                The components, two in a row for a complex component.

        Examples:
            >>> from sage.all__sagemath_symbolics import I
            >>> from skcomplex.engine._utils import symbol, vector
            >>> from skcomplex.engine.problems import Problem
            >>> x, y = symbol('x'), symbol('y')
            >>> Problem._real_components_of(vector([x + I * y, 2 * x]))
            [x, y, 2*x]
        """
        components = []
        for item in (list(value) if isinstance(value, FreeModuleElement) else [value]):
            if isinstance(item, Expression) and item.has(I):
                components += [item.real_part(), item.imag_part()]
            else:
                components.append(item)
        return components

"""Define the system of stocks that change only through named flows."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import TransferInteraction
from ....core.components.laws import Sink, Source
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.errors import ValidationError
from ....core.systems import BaseSystem, System
from .types import FlowTable


class StockFlowSystem(BaseSystem):
    """A system of stocks that change only through named flows between them.

    The stocks and parameters are the values of one element. Every flow leaves a stock or a source outside
    the system and enters a stock or a sink outside the system, at a rate that is a function of a namespace whose
    `source` holds the stocks and parameters and whose `t` holds the time. A labelled `Source` or `Sink` in place of
    `None` names the environment. A flow is a `TransferInteraction` from the element to itself, so the trajectory
    reports it as `<element>.<flow>`, and a flow between two stocks keeps their total.

    A subclass returns the stocks, flows, and parameters from the hooks `_get_stocks`, `_get_flows`, and
    `_get_parameters`, and the label of the element from `_get_element_label`.

    Unlike `DynamicalSystem`, which writes a rate per state, it writes flows between stocks with a
    `TransferInteraction`, so what leaves one stock enters another. Choose it when a model is drawn as stocks and flows,
    as in system dynamics.

    It appears in many fields. A bathtub fills and drains in engineering, nations build arms in the arms race of
    Richardson, and the models of system dynamics of Forrester follow stocks of people, capital, and goods.

    Attributes:
        system_:
            The `System` of the element and its flows built from the parameters at the last `prepare`.

    Examples:
        A bathtub whose faucet fills it as fast as its drain empties it:

        >>> from skcomplex.systems.general.dynamics import StockFlowSystem
        >>> tub = StockFlowSystem(
        ...     stocks={'water': 50.0},
        ...     flows={
        ...         'faucet': (None, 'water', lambda s: s.source.inflow),
        ...         'drain': ('water', None, lambda s: s.source.outflow),
        ...     },
        ...     parameters={'inflow': 5.0, 'outflow': 5.0},
        ...     element='tub',
        ... )
        >>> trajectory = tub.prepare().simulate(until=10.0, dt=5.0)
        >>> trajectory.get_values('tub.water').tolist(), trajectory.get_values('tub.faucet').tolist()
        ([50.0, 50.0, 50.0], [5.0, 5.0, 5.0])

        The arms race of two nations, each arming faster the more the other holds:

        >>> race = StockFlowSystem(
        ...     stocks={'a': 1.0, 'b': 1.0},
        ...     flows={
        ...         'arming_a': (None, 'a', lambda s: s.source.k * s.source.b),
        ...         'arming_b': (None, 'b', lambda s: s.source.k * s.source.a),
        ...     },
        ...     parameters={'k': 0.1},
        ...     element='nations',
        ... )
        >>> trajectory = race.prepare().simulate(until=10.0, dt=10.0)
        >>> round(float(trajectory.get_values('nations.a')[-1]), 3)
        2.718
    """

    def __init__(
        self,
        stocks: Mapping[str, float] | None = None,
        flows: FlowTable | None = None,
        *,
        parameters: Mapping[str, object] | None = None,
        element: str = 'stocks',
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        """Store the stocks, the flows, the parameters, and the labels.

        Args:
            stocks:
                The initial value of every stock, by name, or `None` for no stocks.

            flows:
                The stock it leaves, or `None` or a labelled `Source` for the environment, the stock it enters, or
                `None` or a labelled `Sink` for the environment, and the rate of every flow, by name, or `None` for no
                flows.

            parameters:
                The value of every parameter the rates read, by name, or `None` for none.

            element:
                The label of the element that holds the stocks and the parameters.

            label:
                The label of the system, which names it when it is nested in another one.

            solver:
                The solver of the problem, or `'auto'` for the one the system chooses.

            backend:
                The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.
        """
        super().__init__(label, solver=solver, backend=backend)
        self.stocks = stocks
        self.flows = flows
        self.parameters = parameters
        self.element = element

    def _get_stocks(self) -> Mapping[str, object]:
        """Return the initial value of every stock by name.

        A subclass with parameters of its own returns it.
        """
        return self.stocks or {}

    def _get_flows(self) -> FlowTable:
        """Return the stock each flow leaves, the stock it enters, and its rate, by name.

        A subclass returns them.
        """
        return self.flows or {}

    def _get_parameters(self) -> Mapping[str, object]:
        """Return the value of every parameter by name.

        A subclass with parameters of its own returns them.
        """
        return self.parameters or {}

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the stocks and the parameters.

        A subclass returns the label of its model.
        """
        return self.element

    def _get_solver(self) -> BaseSolver | str:
        """Return the tuned solver, or `'auto'` for the one the problem chooses.

        A subclass whose flows need a solver of their own returns it. A `solver` given to the system replaces it.
        """
        return 'auto'

    def _check_flows(self) -> None:
        """Raise an error for a flow of an unknown stock or a flow between no stocks.

        Raises:
            ValidationError:
                A flow names a stock the system does not declare, or neither leaves nor enters a stock.
        """
        stocks = list(self._get_stocks())
        for name, (leaves, enters, _) in self._get_flows().items():
            if not isinstance(leaves, str) and not isinstance(enters, str):
                error_msg = f'The flow {name!r} of {type(self).__name__} neither leaves nor enters a stock.'
                raise ValidationError(error_msg)
            unknown = [stock for stock in (leaves, enters) if isinstance(stock, str) and stock not in stocks]
            if unknown:
                error_msg = (
                    f'The flow {name!r} of {type(self).__name__} names the stock {unknown[0]!r}, which it does not '
                    f'declare. The declared stocks are {stocks}.'
                )
                raise ValidationError(error_msg)

    def _get_flow_interactions(self) -> list[TransferInteraction]:
        """Return a transfer from the element to itself for every flow.

        Each transfer is labelled and named after its flow.

        Returns:
            interactions:
                The flows, in their order.
        """
        element = self._get_element_label()
        return [
            TransferInteraction(
                name,
                source=element,
                target=element,
                state=Source() if leaves is None else leaves,
                into=Sink() if enters is None else enters,
                amount=rate,
                flow=name,
            )
            for name, (leaves, enters, rate) in self._get_flows().items()
        ]

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Check the flows and set `system_` to the stocks and their flows.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        self._check_flows()
        element = Element(self._get_element_label(), values={**self._get_stocks(), **self._get_parameters()})
        self.system_ = System(
            self.label,
            elements=[element],
            interactions=self._get_flow_interactions(),
            solver=self._get_solver(),
        )

"""Define stock-flow systems.

A stock-flow system is a set of stocks that change only through named flows between them or with the environment.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping

from ....core.components.elements import DeclaredElement
from ....core.components.interactions import TransferInteraction
from ....core.components.laws import Sink, Source
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.errors import ValidationError
from ....core.specs import DeclaredVariableSpec, ParameterSpec, StateSpec
from .types import FlowTable


class StockFlowSystem(BaseSystem[CompositeSystem]):
    """Stocks that change only through named flows.

    The stocks and the parameters are the variables of one `DeclaredElement`. Every flow leaves a stock, or a source
    outside the system, and enters a stock, or a sink outside the system, at a rate that is a function of a namespace
    whose `source` holds the stocks and the parameters and whose `t` holds the time. A labelled `Source` or `Sink` in
    place of `None` names the environment a flow draws from or releases to. A flow is a `TransferInteraction`
    from the element to itself, so the trajectory reports it as `<element>.<flow>`, and a flow between two stocks keeps
    their total.

    The bathtub of engineering and the arms race of social organizations are stock-flow systems, as are the models of
    system dynamics. A subclass takes the parameters of its model under its own names and returns the stocks, the
    flows, and the parameters from the hooks `_get_stocks`, `_get_flows`, and `_get_parameters`, the label of the
    element from `_get_element_label`, and the specs of the variables, such as stocks that cannot fall below zero, from
    `_get_variables`.

    Attributes:
        system_:
            The composite of the element and its flows built from the parameters at the last `prepare`.

        laws_:
            The laws the system contributes for the namespace of its last run.

    Examples:
        A bathtub whose faucet fills it as fast as its drain empties it:

        >>> from skcomplex.systems.general.dynamics import StockFlowSystem
        >>> from skcomplex.core.engine.runtime import simulate
        >>> tub = StockFlowSystem(
        ...     stocks={'water': 50.0},
        ...     flows={
        ...         'faucet': (None, 'water', lambda s: s.source.inflow),
        ...         'drain': ('water', None, lambda s: s.source.outflow),
        ...     },
        ...     parameters={'inflow': 5.0, 'outflow': 5.0},
        ...     element='tub',
        ... )
        >>> trajectory = simulate(tub, until=10.0, dt=5.0)
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
        >>> trajectory = simulate(race, until=10.0, dt=10.0)
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
        """
        super().__init__(label)
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

    def _get_variables(self) -> Mapping[str, DeclaredVariableSpec]:
        """Return the spec of every stock and parameter, the stocks first.

        The stocks are plain `StateSpec` and the parameters plain `ParameterSpec`. A subclass whose variables have
        domains or quantities returns their specs.
        """
        stocks = {name: DeclaredVariableSpec(value, StateSpec()) for name, value in self._get_stocks().items()}
        parameters = {
            name: DeclaredVariableSpec(value, ParameterSpec()) for name, value in self._get_parameters().items()
        }
        return {**stocks, **parameters}

    def _check_flows(self) -> None:
        """Raise an error for a flow of an unknown stock or a flow between no stocks.

        Raises:
            ValidationError:
                A flow names a stock the system does not declare, or neither leaves nor enters a stock.
        """
        stocks = [name for name, variable in self._get_variables().items() if isinstance(variable.spec, StateSpec)]
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
        """Return a transfer from the element to itself for every flow, labelled and named after the flow.

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

    def _build_system(self) -> CompositeSystem:
        """Return the element of the stocks and the parameters, and the flows between them.

        Returns:
            system:
                The composite of the element and its flows.
        """
        self._check_flows()
        return CompositeSystem(
            self.label,
            elements=[DeclaredElement(self._get_element_label(), variables=self._get_variables())],
            interactions=self._get_flow_interactions(),
        )

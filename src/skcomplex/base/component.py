"""Base classes for systems components."""

# Author: Georgios Douzas <gdouzas@icloud.com> License: MIT

import inspect
from abc import abstractmethod
from typing import Self

from sage.all import var


class BaseSystemComponent:
    """Base class for all system components.

    Args:
        label:
            The label of the component.
    """

    def __init__(self: Self, label: str) -> None:
        """Initialize the system component with a label."""
        self.label = label
        self._init()

    @abstractmethod
    def _init_param(self: Self, param_name: str) -> Self:
        return self

    def _init(self: Self) -> Self:
        """Initialize the component's properties."""
        parameters = inspect.signature(self.__init__).parameters.values()  # type: ignore[misc]
        labels = [
            getattr(self, parameter.name)
            for parameter in parameters
            if parameter.name == 'label' or parameter.name.endswith('_label')
        ]
        for parameter in parameters:
            if parameter.name != 'label' and not parameter.name.endswith('_label'):
                value = getattr(self, parameter.name)
                if value is None:
                    setattr(self, f'{parameter.name}_', var(f'{parameter.name}__{"__".join(labels)}'))
                else:
                    self._init_param(parameter.name)
        return self


class BaseElement(BaseSystemComponent):
    """Base class for all elements."""


class BaseElementsInteraction(BaseSystemComponent):
    """Base class for all between elements interactions.

    Args:
        label:
            The label of the interaction.

        element_1_label:
            The label of the first element.

        element_2_label:
            The label of the second element.
    """

    def __init__(self: Self, label: str, element_1_label: str, element_2_label: str) -> None:
        """Initialize the interaction with interaction and elements labels."""
        self.element_1_label = element_1_label
        self.element_2_label = element_2_label
        super().__init__(label)


class BaseEnvironmentInteraction(BaseSystemComponent):
    """Base class for all elements interactions with the environment.

    Args:
        label:
            The label of the interaction.

        element_label:
            The label of the element.
    """

    def __init__(self: Self, label: str, element_label: str) -> None:
        """Initialize the interaction with interaction and element labels."""
        self.element_label = element_label
        super().__init__(label)


class BaseSpace(BaseSystemComponent):
    """Base class for all spaces."""

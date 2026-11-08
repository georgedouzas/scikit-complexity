"""Test the `_connector_spec` module of `skcomplex.core.specs`."""

import pytest

from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import ConnectorSpec


def test_connector_names_the_variables_of_a_port():
    """Test that a connector holds the names of its potential and its flow."""
    connector = ConnectorSpec('chemical', potential='mu', flow='n')
    assert (connector.name, connector.potential, connector.flow) == ('chemical', 'mu', 'n')


@pytest.mark.parametrize(('potential', 'flow'), [('v', 'v'), ('not a name', 'i'), ('v', '')])
def test_connector_rejects_variables_that_are_not_distinct_names(potential, flow):
    """Test that the potential and the flow are two different names."""
    with pytest.raises(ValidationError, match='Give two different names'):
        ConnectorSpec('broken', potential=potential, flow=flow)._check()

"""Test the `_member` module of `skcomplex.core.components.elements`."""

import pytest

from skcomplex.core.components.elements import Member
from skcomplex.core.errors import ValidationError


def test_member_holds_its_group():
    """Test that a member holds the label of the group it names."""
    assert Member('firms').group == 'firms'


@pytest.mark.parametrize('group', ['', 3])
def test_member_rejects_a_group_that_is_not_a_nonempty_string(group):
    """Test that the check names the expected form of a member."""
    with pytest.raises(ValidationError, match=r"Member\('firms'\)"):
        Member(group)._check()

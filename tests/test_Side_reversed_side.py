import pytest

from optical_tracer import Side, reversed_side


@pytest.mark.slow
@pytest.mark.parametrize('side, expected', [(Side.LEFT, Side.RIGHT),
                                            (Side.RIGHT, Side.LEFT),
                                            ])
def test_reversed_side(side, expected):
    assert reversed_side(side) == expected


@pytest.mark.slow
@pytest.mark.parametrize('side, expected_exception', [('Wrong input', AssertionError),
                                                      (None, AssertionError),
                                                      (Side, AssertionError)])
def test_reversed_side_exception(side, expected_exception):
    with pytest.raises(expected_exception):
        reversed_side(side)
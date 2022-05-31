import pytest

from optical_tracer import ObjectKeyWordsMismatchException, UnspecifiedFieldException


@pytest.mark.parametrize('coords, expected', [({'y': "999.44", 'x': -1, 'z': '-2.374'}, (-1, 999.44, -2.374))])
def test_set_coords(coords, expected, create_point):
    pt = create_point
    pt.set_coords(**coords)
    assert (pt.x, pt.y, pt.z) == expected

@pytest.mark.parametrize('coords, expected_exception', [({'y': "999999", 'z': 0}, ObjectKeyWordsMismatchException),
                                                        ({}, ObjectKeyWordsMismatchException),
                                                        ({'y': "999999", 'z': 0, 'x': 1, 'q': 'buz'}, ObjectKeyWordsMismatchException),
                                                        ({'y': "-inf", 'z': 0, 'x': 1}, ValueError),
                                                        ({'y': [1], 'z': 0, 'x': 1}, TypeError),
                                                        # ({'y': "0", 'z': 0, 'x': 1, 'x': 2, 'y':3, 'z':4}, ObjectKeyWordsMismatchException),
                                                        ({'y': "D", 'z': 0, 'x': 1,}, ValueError),

                                                        ]
                         )
def test_set_coords_exception(coords, expected_exception, create_point):
    pt = create_point
    with pytest.raises(expected_exception):
        pt.set_coords(**coords)


@pytest.mark.current
@pytest.mark.parametrize('coords, expected', [('x', {'x': -1.2}),
                                              ('zxy', {'z':0, 'x': -1.2, 'y': 1}),
                                              ('yz', {'y': 1, 'z': 0}),
                                              ('yzy', {'y': 1, 'z': 0}),
                                              ]
                         )
def test_get_coords(coords, expected, create_point):
    pt = create_point
    assert pt.get_coords(coords) == expected



@pytest.mark.parametrize('coords, expected_exception', [('xyzq', UnspecifiedFieldException),
                                                        ('q', UnspecifiedFieldException),
                                                        ('', UnspecifiedFieldException),
                                                        ]
                         )
def test_get_coords_exception(coords, expected_exception, create_point):
    pt = create_point
    with pytest.raises(expected_exception):
        pt.get_coords(coords)

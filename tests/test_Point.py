import pytest

from core import ObjectKeyWordsMismatchException, Point, UnspecifiedFieldException


@pytest.mark.parametrize('coords, expected',
                         [({'y': '10','x': -0 , 'z': 1.00000001}, {'x': 0, 'y': 10, 'z':1.00000001}),
                          ({'y': '+inf','x': -0, 'z': 1}, {'x': 0, 'y': float('inf'), 'z':1}),
                          ({'y': '-inf','x': 'inf', 'z': '+inf'}, {'x': float('inf'), 'y': float('-inf'), 'z':float('inf') })])
def test___init__(coords, expected):
    pt = Point(**coords)
    assert (pt.x, pt.y, pt.z == expected['x'], expected['y'], expected['z'])


@pytest.mark.parametrize('coords, expected_exceptions', [({'y': 0,'x': 'Wrong', 'z': '-inf'}, ValueError),
                                                         ({'y': 0,'x': 0, 'q': 999}, ObjectKeyWordsMismatchException),
                                                         ({'y': 0,'x': 0, 'z': 0, 'q': 999}, ObjectKeyWordsMismatchException),
                                                         ({'y': 0,'x': 0, }, ObjectKeyWordsMismatchException),
                                                         ({}, ObjectKeyWordsMismatchException),
                                                         ]
                         )
def test___init___exceptions(coords, expected_exceptions):
    with pytest.raises(expected_exceptions):
        Point(**coords)


@pytest.mark.parametrize('coords, expected', [({'y': "999.44", 'x': -1, 'z': '-2.374'}, (-1, 999.44, -2.374))])
def test_set_coords(coords, expected, create_point):
    pt = create_point
    pt.set_coords(**coords)
    assert (pt.x, pt.y, pt.z) == expected

@pytest.mark.parametrize('coords, expected_exception', [({'y': "999999", 'z': 0}, ObjectKeyWordsMismatchException),
                                                        ({}, ObjectKeyWordsMismatchException),
                                                        ({'y': "999999", 'z': 0, 'x': 1, 'q': 'buz'}, ObjectKeyWordsMismatchException),
                                                        ({'y': [1], 'z': 0, 'x': 1}, TypeError),
                                                        ({'y': "D", 'z': 0, 'x': 1,}, ValueError),

                                                        ]
                         )
def test_set_coords_exception(coords, expected_exception, create_point):
    pt = create_point
    with pytest.raises(expected_exception):
        pt.set_coords(**coords)


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

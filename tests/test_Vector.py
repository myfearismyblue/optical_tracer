from math import pi

import pytest
from pytest import approx

from optical_tracer import Vector, Point, UnspecifiedFieldException, ObjectKeyWordsMismatchException
from tests.test_OpticalSystem import TOL

init_point = Point(x=-1, z=99.00000001, y="-0.1")
kwargs_lst1 = [
    {'initial_point': init_point, 'lum': '-0', 'w_length': '1000', 'theta': 0.1 + 4 * pi, 'psi': -0.1 - 3 * pi},
    {'initial_point': init_point, 'lum': 1, 'w_length': 780, 'theta': 2 * pi, 'psi': "0"},
    {'initial_point': init_point, 'lum': 1, 'w_length': 780, 'theta': -6 * pi, 'psi': -2 * pi},
    ]
expected_lst1 = [
    {'initial_point': init_point, 'lum': 0, 'w_length': 1000, 'theta': approx(0.1, TOL), 'psi': approx(pi - 0.1, TOL)},
    {'initial_point': init_point, 'lum': 1, 'w_length': 780, 'theta': approx(0, TOL), 'psi': approx(0, TOL)},
    {'initial_point': init_point, 'lum': 1, 'w_length': 780, 'theta': approx(0, TOL), 'psi': approx(0, TOL)},
    ]


@pytest.mark.parametrize('kwargs, expected', [(*zip(kwargs_lst1, expected_lst1))])
def test___init__(kwargs, expected):
    v = Vector(initial_point=kwargs['initial_point'], lum=kwargs['lum'],
               w_length=kwargs['w_length'], theta=kwargs['theta'], psi=kwargs['psi'])
    expected_attrs = expected.keys()
    all_attr_values_ok = all(getattr(v, attr) == expected[attr] for attr in expected_attrs)
    assert all_attr_values_ok


kwargs_lst2 = [{'Wrong': 1, 'initial_point': init_point, 'lum': 10, 'w_length': 555, 'theta': 2 * pi, 'psi': "0"},
               {'lum': 10, 'w_length': 555, 'theta': 2 * pi, 'psi': "0"},
               {'Wrong': 1, 'lum': 10, 'w_length': 555, 'theta': 2 * pi, 'psi': "0"},

               {'initial_point': None, 'lum': '-0', 'w_length': '1000', 'theta': 0.1 + 4 * pi, 'psi': -0.1 - 3 * pi},
               {'initial_point': init_point, 'lum': -1, 'w_length': 555, 'theta': 2 * pi, 'psi': "0"},
               {'initial_point': init_point, 'lum': 'Wrong', 'w_length': 555, 'theta': -6 * pi, 'psi': -2 * pi},
               {'initial_point': init_point, 'lum': 1, 'w_length': -1, 'theta': -6 * pi, 'psi': -2 * pi},
               {'initial_point': init_point, 'lum': 1, 'w_length': 'Wrong', 'theta': -6 * pi, 'psi': -2 * pi},
               {'initial_point': init_point, 'lum': 1, 'w_length': 555, 'theta': 'Wrong', 'psi': -2 * pi},
               {'initial_point': init_point, 'lum': 1, 'w_length': 555, 'theta': -6 * pi, 'psi': 'Wrong'},
               ]
expected_exception_lst = [ObjectKeyWordsMismatchException,
                          ObjectKeyWordsMismatchException,
                          ObjectKeyWordsMismatchException,
                          UnspecifiedFieldException,
                          UnspecifiedFieldException,
                          ValueError,
                          UnspecifiedFieldException,
                          ValueError,
                          ValueError,
                          ValueError,
                          ]


@pytest.mark.parametrize('kwargs, expected_expection', [(*zip(kwargs_lst2, expected_exception_lst))])
def test___init__exception(kwargs, expected_expection):
    with pytest.raises(expected_expection):
        v = Vector(**kwargs)


@pytest.mark.parametrize('angles, expected', [({'psi': 0, 'theta': 0}, {'psi': 0, 'theta': 0}),
                                              ({'psi': 2 * pi, 'theta': 2 * pi}, {'psi': 0, 'theta': 0}),
                                              ({'psi': pi, 'theta': pi}, {'psi': pi, 'theta': pi}),
                                              ({'psi': -pi, 'theta': -pi}, {'psi': pi, 'theta': pi}),
                                              ]
                         )
def test_direction_setter(angles, expected, create_vector):
    v = create_vector
    v.direction = angles
    assert (v.theta, v.psi == expected['theta'], expected['psi'])


@pytest.mark.parametrize('angles, expected_exception', [({'psi': 'inf', 'theta': 0}, ValueError),
                                                        ({'psi': 2 * pi, 'theta': '-inf'}, ValueError),
                                                        ({'psi': 'nan', 'theta': pi}, ValueError),
                                                        ({'psi': 'Wrong', 'theta': pi}, ValueError),
                                                        ({'theta': pi}, ValueError)
                                                        ]
                         )
def test_direction_setter_exception(angles, expected_exception, create_vector):
    v = create_vector
    with pytest.raises(expected_exception):
        v.direction = angles


@pytest.mark.current
@pytest.mark.parametrize('point, expected', [(Point(x='1', y='1', z='1'), Point(x=1, y=1, z=1)),
                                             ]
                         )
def test_initial_point_setter(point, expected, create_vector):
    v = create_vector
    v.initial_point = point
    assert v.initial_point == expected


@pytest.mark.current
@pytest.mark.parametrize('point, expected_exception', [(None, UnspecifiedFieldException),
                                                       ({'x': 0, 'y': 0, 'z': 0}, UnspecifiedFieldException),
                                                       ]
                         )
def test_initial_point_setter_exception(point, expected_exception, create_vector):
    v = create_vector
    with pytest.raises(expected_exception):
        v.initial_point = point


@pytest.mark.current
@pytest.mark.parametrize('lum, expected', [(0, 0),
                                           ('1.00', 1),
                                           ]
                         )
def test_lum_setter(lum, expected, create_vector):
    v = create_vector
    v.lum = lum
    assert v.lum == expected


@pytest.mark.current
@pytest.mark.parametrize('lum, expected_exception', [(-1, UnspecifiedFieldException),
                                                     ('-inf', ValueError),
                                                     ('nan', ValueError),
                                                     ]
                         )
def test_lum_setter_exception(lum, expected_exception, create_vector):
    v = create_vector
    with pytest.raises(expected_exception):
        v.lum = lum



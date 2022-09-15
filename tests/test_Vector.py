from math import pi

import pytest
from pytest import approx

from core import Vector, Point, UnspecifiedFieldException, ObjectKeyWordsMismatchException
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


@pytest.mark.parametrize('point, expected', [(Point(x='1', y='1', z='1'), Point(x=1, y=1, z=1)),
                                             ]
                         )
def test_initial_point_setter(point, expected, create_vector):
    v = create_vector
    v.initial_point = point
    assert v.initial_point == expected


@pytest.mark.parametrize('point, expected_exception', [(None, UnspecifiedFieldException),
                                                       ({'x': 0, 'y': 0, 'z': 0}, UnspecifiedFieldException),
                                                       ]
                         )
def test_initial_point_setter_exception(point, expected_exception, create_vector):
    v = create_vector
    with pytest.raises(expected_exception):
        v.initial_point = point


@pytest.mark.parametrize('lum, expected', [(0, 0),
                                           ('1.00', 1),
                                           ]
                         )
def test_lum_setter(lum, expected, create_vector):
    v = create_vector
    v.lum = lum
    assert v.lum == expected


@pytest.mark.parametrize('lum, expected_exception', [(-1, UnspecifiedFieldException),
                                                     ('-inf', ValueError),
                                                     ('nan', ValueError),
                                                     ]
                         )
def test_lum_setter_exception(lum, expected_exception, create_vector):
    v = create_vector
    with pytest.raises(expected_exception):
        v.lum = lum


@pytest.mark.parametrize('w_length, expected', [(100, 100),
                                                ('1.00', 1),
                                                ]
                         )
def test_w_length_setter(w_length, expected, create_vector):
    v = create_vector
    v.w_length = w_length
    assert v.w_length == expected


@pytest.mark.parametrize('w_length, expected_exception', [(-1, UnspecifiedFieldException),
                                                          ('-inf', ValueError),
                                                          ('nan', ValueError),
                                                          ]
                         )
def test_w_length_setter_exception(w_length, expected_exception, create_vector):
    v = create_vector
    with pytest.raises(expected_exception):
        v.w_length = w_length

@pytest.mark.parametrize('theta, expected', [(-1, 2*pi - 1),
                                             (2*pi, 0),
                                             (-10*pi, 0),
                                             (pi, pi),
                                             (-pi, pi),
                                             (3*pi/2, 3*pi/2),
                                             (-3*pi/2, pi/2),
                                             (-pi/2, 3*pi/2),
                                             ("-1", 2*pi - 1),
                                             ]
                         )
def test_theta_setter(theta, expected, create_vector):
    v = create_vector
    v.theta = theta
    assert v.theta == expected

@pytest.mark.parametrize('theta, expected_exception', [('Wrong', ValueError),
                                                       ('-inf', ValueError),
                                                       ('nan', ValueError),
                                                       ]
                         )
def test_theta_setter_exception(theta, expected_exception, create_vector):
    v = create_vector
    with pytest.raises(expected_exception):
        v.theta = theta

@pytest.mark.parametrize('psi, expected', [(-1, 2*pi - 1),
                                             (2*pi, 0),
                                             (-10*pi, 0),
                                             (pi, pi),
                                             (-pi, pi),
                                             (3*pi/2, 3*pi/2),
                                             (-3*pi/2, pi/2),
                                             (-pi/2, 3*pi/2),
                                             ("-1", 2*pi - 1),
                                             ]
                         )
def test_psi_setter(psi, expected, create_vector):
    v = create_vector
    v.psi = psi
    assert v.psi == expected

@pytest.mark.parametrize('psi, expected_exception', [('Wrong', ValueError),
                                                       ('-inf', ValueError),
                                                       ('nan', ValueError),
                                                       ]
                         )
def test_psi_setter_exception(psi, expected_exception, create_vector):
    v = create_vector
    with pytest.raises(expected_exception):
        v.psi = psi


@pytest.mark.parametrize('slope_deg, expected', [({'slope': '+inf'}, 0),
                                                 ({'slope': 10**9}, approx(0, abs=TOL)),
                                                 ({'slope': 1}, pi/4),
                                                 ({'slope': 0}, pi/2),
                                                 ({'slope': -1}, 3 * pi / 4),
                                                 ({'slope': -10**9}, approx(pi, abs=TOL)),
                                                 ({'slope': '-inf'}, 0),
                                                 ({'slope': '+inf', 'deg': True}, 0),
                                                 ({'slope': 10 ** 9, 'deg': True}, approx(0, abs=TOL)),
                                                 ({'slope': 1, 'deg': True}, 45),
                                                 ({'slope': 0, 'deg': True}, 90),
                                                 ({'slope': -1, 'deg': True}, 135),
                                                 ({'slope': -10 ** 9, 'deg': True}, approx(180, abs=TOL)),
                                                 ({'slope': '-inf', 'deg': True}, 0),
                                                 ]
                         )
def test_calculate_angles(slope_deg, expected, create_vector):
    v = create_vector
    assert v.calculate_angles(**slope_deg) == expected

@pytest.mark.parametrize('slope_deg, expected_exception', [({'slope': 'Wrong', 'deg': True},  ValueError),
                                                       ({'slope': 'nan', 'deg': True}, ValueError),
                                                    ]
                         )
def test_calculate_angles_exception(slope_deg, expected_exception, create_vector):
    v = create_vector
    with pytest.raises(expected_exception):
        v.calculate_angles(**slope_deg)

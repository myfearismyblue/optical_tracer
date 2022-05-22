from math import pi
from typing import List, Tuple

import pytest
from pytest import approx

from optical_tracer import Layer, Material, OpticalComponent, OpticalSystem, Point, NANOMETRE
from optical_tracer import reversed_side, Side, Vector, VectorOutOfComponentException

deg = 2 * pi / 360
Air = Material(name="Air", transmittance=0, refractive_index=1)
TOL = 0.001


@pytest.fixture
def create_two_lenses_opt_sys():
    def create_first_lense():
        first_lense = OpticalComponent(name='first lense')
        first_lense.material = Material(name='Glass', transmittance=0.9, refractive_index=1.5)
        parabolic_l = Layer(name='parabolic', boundary=lambda y: y ** 2 / 10, side=Side.RIGHT)
        first_lense.add_layer(layer=parabolic_l)
        plane_l = Layer(name='plane', boundary=lambda y: 10, side=Side.LEFT)
        first_lense.add_layer(layer=plane_l)
        return first_lense

    def create_second_lense():
        second_lense = OpticalComponent(name='second lense')
        second_lense.material = Material(name='Glass', transmittance=0.9, refractive_index=1.5)
        plane_sec = Layer(name='plane_sec', boundary=lambda y: 20, side=Side.RIGHT)
        second_lense.add_layer(layer=plane_sec)
        parabolic_sec = Layer(name='parabolic', boundary=lambda y: 30 - y ** 2 / 10, side=Side.LEFT)
        second_lense.add_layer(layer=parabolic_sec)
        return second_lense

    opt_sys = OpticalSystem()
    first_lense = create_first_lense()
    second_lense = create_second_lense()
    opt_sys.add_component(component=first_lense)
    opt_sys.add_component(component=second_lense)
    return opt_sys



@pytest.mark.slow
@pytest.mark.parametrize('vector_angle, normal_angle, refractive_index1, refractive_index2, expected',
                         [(190 * deg, 145 * deg, 1, 1.5, approx(173.1255 * deg, abs=TOL)),
                          (325 * deg, 10 * deg, 1, 1.5, approx(341.8744 * deg, abs=TOL)),
                          (0 * deg, 0 * deg, 1, 1.5, approx(0 * deg, abs=TOL)),
                          (190 * deg, 85 * deg, 1, 1, approx(190 * deg, abs=TOL)),
                          (180 * deg, 0 * deg, 1, 1, approx(180 * deg, abs=TOL)),
                          ]
                         )
def test__get_refract_angle(vector_angle, normal_angle, prev_index, next_index, expected, create_two_lenses_opt_sys):
    opt_sys = create_two_lenses_opt_sys
    assert opt_sys._get_refract_angle(vector_angle=vector_angle, normal_angle=normal_angle,
                                      prev_index=prev_index,
                                      next_index=next_index) == expected


parabolic_l = Layer(name='parabolic',
                    boundary=lambda y: y ** 2,
                    side=Side.RIGHT,
                    )
plane_l = Layer(name='plane',
                boundary=lambda y: 11,
                side=Side.LEFT,
                )

point = [Point(x=0, y=0.0001, z=0),
         Point(x=0, y=0, z=0),
         Point(x=0, y=-0.0001, z=0),
         Point(x=0, y=100, z=0),
         Point(x=0, y=-100, z=0),
         Point(x=0, y=11, z=0),
         ]


@pytest.mark.slow
@pytest.mark.parametrize('layer, point, expected',
                         [(parabolic_l, point[0], approx(3.1413926, abs=TOL)),  # ~<pi - approaching zero from plus
                          (parabolic_l, point[1], approx(0, abs=TOL)),  # parallel to z-axis
                          (parabolic_l, point[2], approx(0.0001999, abs=TOL)),  # approaching zero from minus
                          (parabolic_l, point[3], approx(1.575796285141478, abs=TOL)),  # almost perpendicular
                          (parabolic_l, point[4], approx(1.5657963684483152, abs=TOL)),
                          # same approaching from another side
                          (plane_l, point[5], approx(0, abs=TOL)),  # check with constant (plane) bound
                          ])
def test__get_normal_angle(layer, point, expected):
    assert layer.get_normal_angle(point=point) == expected


v = [Vector(initial_point=Point(x=0, y=0, z=0.1), lum=1, w_length=555, theta=0.01, psi=0),  # inside first
     Vector(initial_point=Point(x=0, y=0, z=25), lum=1, w_length=555, theta=0.01, psi=0),  # inside second
     Vector(initial_point=Point(x=0, y=0, z=-1), lum=1, w_length=555, theta=0.01, psi=0),  # outside any at very left
     Vector(initial_point=Point(x=0, y=0, z=1000), lum=1, w_length=555, theta=0.01, psi=0),  # outside any at very right
     Vector(initial_point=Point(x=0, y=0, z=0), lum=1, w_length=555, theta=0.01 + pi, psi=0),
     # at boundary of first, but directed outside
     ]


@pytest.mark.slow
@pytest.mark.parametrize('vector, expected', [(v[0], 'first lense'),
                                              (v[1], 'second lense'),
                                              (v[4], 'first lense'),
                                              ])
def test__get_containing_component(vector, expected, create_two_lenses_opt_sys):
    opt_sys = create_two_lenses_opt_sys
    assert opt_sys._get_containing_component(vector=vector).name == expected


@pytest.mark.slow
@pytest.mark.parametrize('vector, expected_exception', [(v[2], VectorOutOfComponentException),
                                                        (v[3], VectorOutOfComponentException),
                                                        ])
def test__get_containing_component(vector, expected_exception, create_two_lenses_opt_sys):
    opt_sys = create_two_lenses_opt_sys
    with pytest.raises(expected_exception):
        opt_sys._get_containing_component(vector=vector) == expected_exception


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


v = [Vector(initial_point=Point(x=0, y=0, z=-1), lum=1, w_length=555, theta=0.3, psi=0),
     Vector(initial_point=Point(x=0, y=0, z=-1), lum=1, w_length=555, theta=0.5, psi=0),
     Vector(initial_point=Point(x=0, y=0, z=-1), lum=1, w_length=555, theta=0.9, psi=0),
     Vector(initial_point=Point(x=0, y=10, z=0), lum=1, w_length=555, theta=5.81953769817878, psi=0),
     ]


@pytest.mark.slow
@pytest.mark.parametrize('vector, expected_x_y_z', [
    (v[0], [(approx(0, abs=TOL), approx(0, abs=TOL), approx(-1, abs=TOL)),
            (approx(0, abs=TOL), approx(0.3123, abs=TOL), approx(0.00976, abs=TOL)),
            (approx(0, abs=TOL), approx(3.403, abs=TOL), approx(10, abs=TOL)),
            (approx(0, abs=TOL), approx(6.496, abs=TOL), approx(20, abs=TOL)),
            (approx(0, abs=TOL), approx(7.737, abs=TOL), approx(24.013, abs=TOL)),

            ]
     ),
    (v[1], [(approx(0, abs=TOL), approx(0, abs=TOL), approx(-1, abs=TOL)),
            (approx(0, abs=TOL), approx(0.564, abs=TOL), approx(0.032, abs=TOL)),
            (approx(0, abs=TOL), approx(6.009, abs=TOL), approx(10, abs=TOL))
            ]
     ),
    (v[2], [(approx(0, abs=TOL), approx(0, abs=TOL), approx(-1, abs=TOL)),
            (approx(0, abs=TOL), approx(1.571, abs=TOL), approx(0.247, abs=TOL)),
            (approx(0, abs=TOL), approx(6.364, abs=TOL), approx(4.05, abs=TOL))
            ]
     ),
    (v[3], [(approx(0, abs=TOL), approx(10, abs=TOL), approx(0, abs=TOL)),
            (approx(0, abs=TOL), approx(7.321, abs=TOL), approx(5.359, abs=TOL)),
            (approx(0, abs=TOL), approx(5, abs=TOL), approx(10, abs=TOL)),
            (approx(0, abs=TOL), approx(0, abs=TOL), approx(20, abs=TOL)),
            (approx(0, abs=TOL), approx(-4.142, abs=TOL), approx(28.284, abs=TOL)),
            ]
     ),
]
                         )
def test_trace(vector: Vector, expected_x_y_z: List[Tuple[float]], create_two_lenses_opt_sys):
    def pick_out_coords(vectors: List[Vector]) -> List[Tuple[float]]:
        def _fetch_point(vector: Vector) -> Point:
            return vector.initial_point

        ret = []
        for vect in vectors:
            point = _fetch_point(vect)
            ret.append((point.x, point.y, point.z))
        return ret
    opt_sys = create_two_lenses_opt_sys
    picked = pick_out_coords(opt_sys.trace(vector=vector))
    assert picked == expected_x_y_z

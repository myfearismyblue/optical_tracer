from math import pi
from typing import List, Tuple, Union

import pytest
from pytest import approx

from core import Material, Point, Vector, VectorOutOfComponentException

deg = 2 * pi / 360
Air = Material(name="Air", transmittance=0, refractive_index=1)
TOL = 0.001


@pytest.mark.slow
@pytest.mark.parametrize('vector_angle, normal_angle, prev_index, next_index, expected',
                         [(190 * deg, 145 * deg, 1, 1.5, approx(173.1255 * deg, abs=TOL)),
                          (325 * deg, 10 * deg, 1, 1.5, approx(341.8744 * deg, abs=TOL)),
                          (0 * deg, 0 * deg, 1, 1.5, approx(0 * deg, abs=TOL)),
                          (190 * deg, 85 * deg, 1, 1, approx(190 * deg, abs=TOL)),
                          (180 * deg, 0 * deg, 1, 1, approx(180 * deg, abs=TOL)),
                          (0.3, 3.0792028712392954, 1, 1.5, approx(0.176, abs=TOL)),
                          (0 * deg, 179.99999999 * deg, 1, 1, approx(0 * deg, abs=TOL)),
                          ]
                         )
def test__get_refract_angle(vector_angle, normal_angle, prev_index, next_index, expected, create_two_lenses_opt_sys):
    opt_sys = create_two_lenses_opt_sys
    assert opt_sys._get_refract_angle(vector_angle=vector_angle, normal_angle=normal_angle,
                                      prev_index=prev_index,
                                      next_index=next_index) == expected


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
    """NO_REFRACTION = True here"""
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


v = [Vector(initial_point=Point(x=0, y=0, z=-2), lum=1, w_length=555, theta=0.1, psi=0),
     Vector(initial_point=Point(x=0, y=0, z=40), lum=1, w_length=555, theta=pi+pi/4, psi=0),
     ]


@pytest.mark.slow
@pytest.mark.parametrize('vector, expected',
                         [              #____x___________________y______________________z________________theta______________________psi__________
                             (v[0], [(approx(0, abs=TOL), approx(0, abs=TOL), approx(-2, abs=TOL), approx(.1, abs=TOL), approx(0, abs=TOL)),
                                     (approx(0, abs=TOL), approx(0.2007, abs=TOL), approx(0, abs=TOL), approx(.090883, abs=TOL), approx(0, abs=TOL)),
                                     (approx(0, abs=TOL), approx(1.112, abs=TOL), approx(10, abs=TOL), approx(.083291, abs=TOL), approx(0, abs=TOL)),
                                     (approx(0, abs=TOL), approx(1.9468, abs=TOL), approx(20, abs=TOL), approx(.076871, abs=TOL), approx(0, abs=TOL)),
                                     (approx(0, abs=TOL), approx(2.717, abs=TOL), approx(30, abs=TOL), approx(.07137, abs=TOL), approx(0, abs=TOL)),

                                     ]
                              )
                         ]
                         )
def test_trace(vector: Vector, expected: List[Tuple[float]], create_parallel_slices_opt_sys):
    def pick_out_values(vectors: List[Vector]) -> List[Tuple[float]]:
        """Returns a list of tuples like (*coordinate, *angles) of initial points of each vector"""
        def _fetch_point(vector: Vector) -> Point:
            return vector.initial_point

        def _fetch_angles(vector: Vector) -> Tuple[Union[int, float], Union[int, float]]:
            return vector.theta, vector.psi

        ret = []
        for vect in vectors:
            point = _fetch_point(vect)
            angles = _fetch_angles(vect)
            ret.append((point.x, point.y, point.z, *angles))
        return ret

    opt_sys = create_parallel_slices_opt_sys
    picked = pick_out_values(opt_sys.trace(vector=vector))
    for p, e in zip(picked, expected):
        if p != e:
            raise Exception(f'{p} = {e} False')
    assert picked == expected

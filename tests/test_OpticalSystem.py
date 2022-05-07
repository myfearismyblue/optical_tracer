import pytest
from optical_tracer import Layer, Material, OpticalComponent, OpticalSystem, Point, Side, Vector, VectorOutOfComponentWarning
from math import pi

deg = 2 * pi / 360
Air = Material(name="Air", transmittance=0, refractive_index=1)
opt_c = OpticalComponent(name='first lense')


@pytest.mark.parametrize('vector_angle, normal_angle, refractive_index1, refractive_index2, expected',
                         [(190 * deg, 145 * deg, 1, 1.5, pytest.approx(173.1255 * deg, 0.001)),
                          (325 * deg, 10 * deg, 1, 1.5, pytest.approx(341.8744 * deg, 0.001)),
                          (0 * deg, 0 * deg, 1, 1.5, pytest.approx(0 * deg, 0.001)),
                          (190 * deg, 85 * deg, 1, 1, pytest.approx(190 * deg, 0.001)),
                          (180 * deg, 0 * deg, 1, 1, pytest.approx(180 * deg, 0.001)),
                          ]
                         )
def test__get_refract_angle(vector_angle, normal_angle, refractive_index1, refractive_index2, expected):
    assert opt_c._get_refract_angle(vector_angle=vector_angle, normal_angle=normal_angle,
                                    refractive_index1=refractive_index1,
                                    refractive_index2=refractive_index2) == expected


opt_c = OpticalComponent(name='first lense')
opt_c.material = Material(name='Glass', transmittance=0.9, refractive_index=1.5)
parabolic_l = Layer(name='parabolic',
                    boundary=lambda y: y ** 2,
                    side=Side.RIGHT,
                    )
opt_c.add_layer(layer=parabolic_l)
plane_l = Layer(name='plane',
                boundary=lambda y: 11,
                side=Side.LEFT,
                )
opt_c.add_layer(layer=plane_l)

intersec = [(parabolic_l, Point(x=0, y=0.0001, z=0)),
            (parabolic_l, Point(x=0, y=0, z=0)),
            (parabolic_l, Point(x=0, y=-0.0001, z=0)),
            (parabolic_l, Point(x=0, y=100, z=0)),
            (parabolic_l, Point(x=0, y=-100, z=0)),
            (plane_l, Point(x=0, y=11, z=0)),
            ]


@pytest.mark.parametrize('intersec, expected',
                         [(intersec[0], pytest.approx(3.1413926, 0.001)),  # ~<pi - approaching zero from plus
                          (intersec[1], pytest.approx(0, 0.001)),  # parallel to z-axis
                          (intersec[2], pytest.approx(0.0001999, 0.001)),  # approaching zero from minus
                          (intersec[3], pytest.approx(1.575796285141478, 0.001)),  # almost perpendicular
                          (intersec[4], pytest.approx(1.5657963684483152, 0.001)),  # same approaching from another side
                          (intersec[5], pytest.approx(0, 0.001)),  # check with constant (plane) bound
                          ])
def test__get_normal_angle(intersec, expected):
    assert opt_c._get_normal_angle(intersection=intersec) == expected


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

v = [Vector(initial_point=Point(x=0, y=0, z=0.1), lum=1, w_length=555, theta=0.01, psi=0),  # inside first
     Vector(initial_point=Point(x=0, y=0, z=25), lum=1, w_length=555, theta=0.01, psi=0),   # inside second
     Vector(initial_point=Point(x=0, y=0, z=-1), lum=1, w_length=555, theta=0.01, psi=0),   # outside any at very left
     Vector(initial_point=Point(x=0, y=0, z=1000), lum=1, w_length=555, theta=0.01, psi=0), # outside any at very right
     Vector(initial_point=Point(x=0, y=0, z=0), lum=1, w_length=555, theta=0.01+pi, psi=0), # at boundary of first, but directed outside
     ]

# a = opt_sys.get_containing_component(vector=v[2])
pass


@pytest.mark.parametrize('vector, expected', [(v[0], 'first lense'),
                                              (v[1], 'second lense'),
                                              ])
def test_get_containing_component(vector, expected):
    assert opt_sys.get_containing_component(vector=vector).name == expected


@pytest.mark.parametrize('vector, expected_exception', [(v[2], VectorOutOfComponentWarning),
                                                        (v[3], VectorOutOfComponentWarning),
                                                        (v[4], VectorOutOfComponentWarning),
                                                        ])
def test_get_containing_component(vector, expected_exception):
    with pytest.raises(expected_exception):
        opt_sys.get_containing_component(vector=vector) == expected_exception

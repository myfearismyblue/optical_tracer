import pytest
from optical_tracer import Layer, Material, OpticalComponent, Point, Side, Vector
from math import pi

deg = 2 * pi / 360

opt_c = OpticalComponent()


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
                                    refractive_index1=refractive_index1, refractive_index2=refractive_index2) == expected



opt_c = OpticalComponent()
opt_c.material = Material(name='Glass', transparency=0.9, refractive_index=1.5)
parabolic_l = Layer(name='parabolic',
                    boundary=lambda y: y ** 2 ,
                    side=Side.RIGHT,
                    )
opt_c.add_layer(new_layer=parabolic_l)
plane_l = Layer(name='plane',
                boundary=lambda y: 11,
                side=Side.LEFT,
                )
opt_c.add_layer(new_layer=plane_l)


intersec = [(parabolic_l, Point(x=0, y=0.0001, z=0)),
            (parabolic_l, Point(x=0, y=0, z=0)),
            (parabolic_l, Point(x=0, y=-0.0001, z=0)),
            (parabolic_l, Point(x=0, y=100, z=0)),
            (parabolic_l, Point(x=0, y=-100, z=0)),
            (plane_l, Point(x=0, y=11, z=0)),
            ]
@pytest.mark.parametrize('intersec, expected',
                         [(intersec[0], pytest.approx(3.1413926, 0.001) ),          # ~<pi - approaching zero from plus
                          (intersec[1], pytest.approx(0, 0.001) ),                  # parallel to z-axis
                          (intersec[2], pytest.approx(0.0001999, 0.001) ),          # approaching zero from minus
                          (intersec[3], pytest.approx(1.575796285141478, 0.001)),   # almost perpendicular
                          (intersec[4], pytest.approx(1.5657963684483152, 0.001)),  # same approaching from another side
                          (intersec[5], pytest.approx(0, 0.001)),                   # check with constant (plane) bound
                         ])
def test__get_noraml_angle(intersec, expected):
    assert opt_c._get_noraml_angle(intersection=intersec) == expected




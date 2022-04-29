import pytest
from optical_tracer import OpticalSystem
from math import pi

deg = 2 * pi / 360

opt_s = OpticalSystem()


@pytest.mark.parametrize('vector_angle, normal_angle, refractive_index1, refractive_index2, expected',
                         [(190 * deg, 145 * deg, 1, 1.5, pytest.approx(173.1255 * deg, 0.001)),
                          (325 * deg, 10 * deg, 1, 1.5, pytest.approx(341.8744 * deg, 0.001)),
                          (0 * deg, 0 * deg, 1, 1.5, pytest.approx(0 * deg, 0.001)),
                          (190 * deg, 85 * deg, 1, 1, pytest.approx(190 * deg, 0.001)),
                          (180 * deg, 0 * deg, 1, 1, pytest.approx(180 * deg, 0.001)),
                         ]
                        )
def test__get_refract_angle(vector_angle, normal_angle, refractive_index1, refractive_index2, expected):
    assert opt_s._get_refract_angle(vector_angle=vector_angle, normal_angle=normal_angle,
                                    refractive_index1=refractive_index1, refractive_index2=refractive_index2) == expected

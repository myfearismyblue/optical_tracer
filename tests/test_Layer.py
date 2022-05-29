import pytest
from pytest import approx

from optical_tracer import Point
from tests.test_OpticalSystem import TOL

point = [Point(x=0, y=0.0001, z=0),
         Point(x=0, y=0, z=0),
         Point(x=0, y=-0.0001, z=0),
         Point(x=0, y=100, z=0),
         Point(x=0, y=-100, z=0),
         Point(x=0, y=11, z=0),
         ]


@pytest.mark.slow
@pytest.mark.parametrize('point, expected',
                         [(point[0], approx(3.1413926, abs=TOL)),  # ~<pi - approaching zero from plus
                          (point[1], approx(0, abs=TOL)),  # parallel to z-axis
                          (point[2], approx(0.0001999, abs=TOL)),  # approaching zero from minus
                          (point[3], approx(1.575796285141478, abs=TOL)),  # almost perpendicular
                          (point[4], approx(1.5657963684483152, abs=TOL)),  # same approaching from another side
                          ])
def test__get_normal_angle_parabolic(point, expected, create_parabolic_layer):
    parabolic_layer = create_parabolic_layer
    assert parabolic_layer.get_normal_angle(point=point) == expected


@pytest.mark.slow
@pytest.mark.parametrize('point, expected',
                         [(point[5], approx(0, abs=TOL)),  # check with constant (plane) bound
                          ])
def test__get_normal_angle_parabolic(point, expected, create_plane_layer):
    plane_layer = create_plane_layer
    assert plane_layer.get_normal_angle(point=point) == expected

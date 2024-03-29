from math import sqrt

import pytest
from pytest import approx

from core import get_distance, Point, UnspecifiedFieldException, NanPointException


def create_points():
    ret = []
    ret.append((Point(x=0, y=0, z=0), Point(x=1, y=1, z=1)))
    ret.append((Point(x=-10, y=-10, z=-10), Point(x=10, y=10, z=10)))
    ret.append((Point(x=0, y=0, z=0), Point(x=0, y=0, z=0)))
    ret.append((Point(x=0, y=0, z=float('inf')), (Point(x=0, y=0, z=float('-inf')))))
    ret.append((Point(x=float('inf'), y=float('inf'), z=float('inf')), (Point(x=float('-inf'), y=float('-inf'), z=float('-inf')))))
    ret.append((Point(x=0, y=float('-inf'), z=float('inf')), (Point(x=0, y=float('inf'), z=float('-inf')))))

    return ret


expected = [sqrt(3), 20 * sqrt(3), 0, float('inf'), float('inf'), float('inf')]


@pytest.mark.parametrize('expected, points', [(*zip(map(approx, expected), create_points()))])
def test_get_distance(expected, points):
    assert get_distance(*points) == expected


def create_points_exception():
    ret = []
    ret.append((None, Point(x=1, y=1, z=1)))
    ret.append((None,None,))
    ret.append((Point(x=0, y=float('-inf'), z=float('+inf')), (Point(x=0, y=float('-inf'), z=float('+inf')))))
    ret.append((Point(x=0, y=float('inf'), z=float('inf')), (Point(x=0, y=float('inf'), z=float('+inf')))))
    ret.append((Point(x=0, y=float('-inf'), z=float('inf')), (Point(x=0, y=float('-inf'), z=float('+inf')))))
    return ret

expected_exception = [UnspecifiedFieldException,
                      UnspecifiedFieldException,
                      NanPointException,
                      NanPointException,
                      NanPointException]


@pytest.mark.parametrize('expected_exception, points', [(*zip(expected_exception, create_points_exception()))])
def test_get_distance_exceptions(expected_exception, points):
    with pytest.raises(expected_exception):
        get_distance(*points)

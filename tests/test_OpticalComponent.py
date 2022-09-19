import pytest

from pytest import approx

from core import OpticalComponent, Layer, Side, Point, NoLayersIntersectionException
from tests.test_OpticalSystem import TOL

layers = [(Layer(name='l1', boundary=lambda y: y ** 2, side=Side.RIGHT),Layer(name='l2', boundary=lambda y: 4, side=Side.LEFT)),
          (Layer(name='l2', boundary=lambda y: 4, side=Side.LEFT), Layer(name='l3', boundary=lambda y: y + 4, side=Side.LEFT)),
          (Layer(name='l3', boundary=lambda y: y + 4, side=Side.LEFT), Layer(name='l2', boundary=lambda y: 4, side=Side.LEFT)),
          (Layer(name='l4', boundary=lambda y: y ** 2, side=Side.LEFT), Layer(name='l5', boundary=lambda y: 4, side=Side.RIGHT)),
          ]

exp = [[(Point(x=0, y=-2, z=4), Point(x=0, y=+2, z=4))],
       [(Point(x=0, y=0, z=4), Point(x=0, y=float('+inf'), z=4))],
       [(Point(x=0, y=float('-inf'), z=float('-inf')), Point(x=0, y=0, z=4))],
       [(Point(x=0, y=float('-inf'), z=float('+inf')), Point(x=0, y=-2, z=4)), (Point(x=0, y=2, z=4), Point(x=0, y=float('+inf'), z=float('+inf')))],
       ]

comp = OpticalComponent(name="test")


@pytest.mark.parametrize('current_layer, bounding_layer, expected',
                         [(layers[0][0], layers[0][1], exp[0]),
                          (layers[1][0], layers[1][1], exp[1]),
                          (layers[2][0], layers[2][1], exp[2]),
                          (layers[3][0], layers[3][1], exp[3]),
                          ]
                         )
def test__get_layer_segments(current_layer: Layer, bounding_layer: Layer, expected):
    ret = comp._get_layer_segments(current_layer=current_layer, bounding_layer=bounding_layer)
    print(f'{ret = }')
    for ret_segment, expect_segment in zip(ret, expected):
        for attr in Point.__slots__:
            assert getattr(ret_segment[0], attr) == approx(getattr(expect_segment[0], attr), TOL)
            assert getattr(ret_segment[1], attr) == approx(getattr(expect_segment[1], attr), TOL)


layers_except = [(Layer(name='l6', boundary=lambda y: y ** 2, side=Side.LEFT), Layer(name='l5', boundary=lambda y: 1 + y ** 2, side=Side.RIGHT)),
                 (Layer(name='l6', boundary=lambda y: 2, side=Side.LEFT), Layer(name='l5', boundary=lambda y: 1, side=Side.RIGHT)),

                 ]
exp_except = [NoLayersIntersectionException,
              NoLayersIntersectionException]


@pytest.mark.parametrize('current_layer, bounding_layer, expected_exception',
                         [(layers_except[0][0], layers_except[0][1], exp_except[0]),
                          (layers_except[1][0], layers_except[1][1], exp_except[1]),
                          ]
                         )
def test__get_layer_segments_exceptions(current_layer: Layer, bounding_layer: Layer, expected_exception):
    with pytest.raises(expected_exception):
        comp._get_layer_segments(current_layer=current_layer, bounding_layer=bounding_layer) == expected_exception








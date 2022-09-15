import pytest

from core import Layer, Material, OpticalComponent, OpticalSystem, Point, Side, Vector


@pytest.fixture
def create_point():
    pt = Point(z=0.0, x='-1.2', y=+1)
    return pt

@pytest.fixture
def create_vector():
    return Vector(initial_point=Point(x=999, y=999, z=999), lum=999, w_length=999, theta=999, psi=999)

@pytest.fixture
def create_parabolic_layer():
    parabolic_l = Layer(name='parabolic',
                        boundary=lambda y: y ** 2,
                        side=Side.RIGHT,
                        )
    return parabolic_l

@pytest.fixture
def create_plane_layer():
    plane_l = Layer(name='plane',
                boundary=lambda y: 11,
                side=Side.LEFT,
                )
    return plane_l

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

    def create_fourth_lense():
        fourth_lense = OpticalComponent(name='fourth lense')
        fourth_lense.material = Material(name='Glass', transmittance=0.9, refractive_index=1.5)
        plane_sec = Layer(name='plane_sec', boundary=lambda y: 20, side=Side.RIGHT)
        fourth_lense.add_layer(layer=plane_sec)
        parabolic_sec = Layer(name='parabolic', boundary=lambda y: 30 - y ** 2 / 10, side=Side.LEFT)
        fourth_lense.add_layer(layer=parabolic_sec)
        return fourth_lense

    opt_sys = OpticalSystem()
    first_lense = create_first_lense()
    fourth_lense = create_fourth_lense()
    opt_sys.add_component(component=first_lense)
    opt_sys.add_component(component=fourth_lense)
    return opt_sys

@pytest.fixture
def create_parallel_slices_opt_sys():
    """Creates an Optical System which is composed of three parallel layers and five optical media"""
    def create_first_medium():
        first_left_bound = Layer(boundary=lambda y: 0, side=Side.RIGHT, name='First-left bound')
        first_right_bound = Layer(boundary=lambda y: 10, side=Side.LEFT, name='First-right bound')
        first_material = Material(name='Glass', transmittance=0.9, refractive_index=1.1)
        first_medium = OpticalComponent(name='First')
        first_medium.add_layer(layer=first_left_bound)
        first_medium.add_layer(layer=first_right_bound)
        first_medium.material = first_material
        return first_medium

    def create_second_medium():
        second_left_bound = Layer(boundary=lambda y: 10, side=Side.RIGHT, name='Second-left bound')
        second_right_bound = Layer(boundary=lambda y: 20, side=Side.LEFT, name='Second-right bound')
        second_material = Material(name='Glass', transmittance=0.9, refractive_index=1.2)
        second_medium = OpticalComponent(name='Second')
        second_medium.add_layer(layer=second_left_bound)
        second_medium.add_layer(layer=second_right_bound)
        second_medium.material = second_material
        return second_medium

    def create_third_medium():
        third_left_bound = Layer(boundary=lambda y: 20, side=Side.RIGHT, name='Third-left bound')
        third_right_bound = Layer(boundary=lambda y: 30, side=Side.LEFT, name='Third-right bound')
        third_material = Material(name='Glass', transmittance=0.9, refractive_index=1.3)
        third_medium = OpticalComponent(name='Third')
        third_medium.add_layer(layer=third_left_bound)
        third_medium.add_layer(layer=third_right_bound)
        third_medium.material = third_material
        return third_medium

    def create_fourth_medium():
        fourth_left_bound = Layer(boundary=lambda y: 30, side=Side.RIGHT, name='Fourth-left bound')
        fourth_material = Material(name='Glass', transmittance=0.9, refractive_index=1.4)
        fourth_medium = OpticalComponent(name='Fourth')
        fourth_medium.add_layer(layer=fourth_left_bound)
        fourth_medium.material = fourth_material
        return fourth_medium

    opt_sys = OpticalSystem()
    first_medium, second_medium, third_medium, fourth_medium = (medium for medium in (create_first_medium(),
                                                                                      create_second_medium(),
                                                                                      create_third_medium(),
                                                                                      create_fourth_medium()
                                                                                      )
                                                                )
    [opt_sys.add_component(component=med) for med in (first_medium, second_medium, third_medium, fourth_medium)]
    return opt_sys


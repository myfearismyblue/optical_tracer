import tkinter as tk
from typing import Callable, Tuple, List

from optical_tracer import Layer, OpticalComponent, Material, OpticalSystem, Side, Vector

DEBUG = True
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600
CANVAS_BACKGROUND_COLOR = 'white'
TIME_REFRESH = 100 # ms

# offset of entire optical system relatively to (0, 0) canvas point which is upper-left corner
OPTICAL_SYSTEM_OFFSET = (+CANVAS_WIDTH//4, +3*CANVAS_HEIGHT//4)

# ranges in which components to be drawn relatively to OPTICAL_SYSTEM_OFFSET
BOUNDARY_DRAW_RANGES = (OPTICAL_SYSTEM_OFFSET[1] - CANVAS_HEIGHT, OPTICAL_SYSTEM_OFFSET[1])

def main():
    global root, canvas
    objects = init_objects()

    root = tk.Tk()
    root.geometry(str(CANVAS_WIDTH) + 'x' + str(CANVAS_HEIGHT))
    canvas = tk.Canvas(root, background=CANVAS_BACKGROUND_COLOR)
    if DEBUG:
        mouse_coords_text = canvas.create_text(50, 10, text=f'', font='28')

    canvas.focus_set()

    gr = Grapher(canvas=canvas, opt_system=objects[0])

    canvas.bind('<Motion>', lambda event: key_handler(event, mouse_coords_text, *objects))
    canvas.pack(fill=tk.BOTH, expand=1)
    # tick(*objects)
    root.mainloop()

def convert_tkcoords_to_optical(tk_absciss: int, tk_ordinate: int) -> Tuple[float, float]:
    opt_absciss = tk_absciss - OPTICAL_SYSTEM_OFFSET[0]
    opt_ordinate = OPTICAL_SYSTEM_OFFSET[1] - tk_ordinate
    return opt_absciss, opt_ordinate

def convert_opticalcoords_to_tkcoords(opt_absciss: int, opt_ordinate: int) -> Tuple[float, float]:
    tk_absciss = opt_absciss + OPTICAL_SYSTEM_OFFSET[0]
    tk_ordinate = OPTICAL_SYSTEM_OFFSET[1] - opt_ordinate
    return tk_absciss, tk_ordinate

def tick(*objects):
    root.after(TIME_REFRESH, *objects)


def key_handler(event, mouse_coords, *objects):
    if str(event.type) == 'KeyPress':
        if event.keysym == 'space':
            ...
        elif event.keysym == 'Shift_L':
            ...
    elif str(event.type) == 'KeyRelease':
        if event.keysym == 'Shift_L':
            ...

    elif str(event.type) == 'ButtonPress':
        if event.num == 1:
            # In left click
            if event.state == 131080:
                # Left Alt pressed
                ...
        elif event.num == 3:
            # On right-click
            ...

    elif str(event.type) in ['Motion', '6']:
        opt_abs, opt_ord = convert_tkcoords_to_optical(event.x, event.y)
        canvas.itemconfig(mouse_coords, text=f'y: {opt_ord}, z: {opt_abs}')

    elif str(event.type) == 'MouseWheel':
        if event.delta >= 0:
            ...
        else:
            ...


def init_objects():
    def create_opt_sys():
        """Creates an Optical System which is composed of three parallel layers and five optical media"""

        def create_first_medium():
            first_left_bound = Layer(boundary=lambda y: 0 + y ** 2 / 300, side=Side.RIGHT, name='First-left bound')
            first_right_bound = Layer(boundary=lambda y: 10 + y ** 2 / 300, side=Side.LEFT, name='First-right bound')
            first_material = Material(name='Glass', transmittance=0.9, refractive_index=1.1)
            first_medium = OpticalComponent(name='First')
            first_medium.add_layer(layer=first_left_bound)
            first_medium.add_layer(layer=first_right_bound)
            first_medium.material = first_material
            return first_medium

        def create_second_medium():
            second_left_bound = Layer(boundary=lambda y: 10 + y ** 2 / 300, side=Side.RIGHT, name='Second-left bound')
            second_right_bound = Layer(boundary=lambda y: 20 + y ** 2 / 300, side=Side.LEFT, name='Second-right bound')
            second_material = Material(name='Glass', transmittance=0.9, refractive_index=1.2)
            second_medium = OpticalComponent(name='Second')
            second_medium.add_layer(layer=second_left_bound)
            second_medium.add_layer(layer=second_right_bound)
            second_medium.material = second_material
            return second_medium

        def create_third_medium():
            third_left_bound = Layer(boundary=lambda y: 20 + y ** 2 / 300, side=Side.RIGHT, name='Third-left bound')
            third_right_bound = Layer(boundary=lambda y: 30 + y ** 2 / 300, side=Side.LEFT, name='Third-right bound')
            third_material = Material(name='Glass', transmittance=0.9, refractive_index=1.3)
            third_medium = OpticalComponent(name='Third')
            third_medium.add_layer(layer=third_left_bound)
            third_medium.add_layer(layer=third_right_bound)
            third_medium.material = third_material
            return third_medium

        def create_fourth_medium():
            fourth_left_bound = Layer(boundary=lambda y: 30 + y ** 2 / 300, side=Side.RIGHT, name='Fourth-left bound')
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
    objects = (create_opt_sys(),)
    return objects


class Grapher:
    """Responsible for drawing and refreshing visualities on a canvas"""
    def __init__(self, *, canvas, opt_system):
        self._canvas = canvas
        self._optical_system = opt_system
        self._draw_initial_axes()
        self._draw_components()

    def _draw_initial_axes(self):
        optical_axis_points = 0, \
                              0 + OPTICAL_SYSTEM_OFFSET[1], \
                              CANVAS_WIDTH, \
                              OPTICAL_SYSTEM_OFFSET[1]

        y_axis_points = OPTICAL_SYSTEM_OFFSET[0], \
                        0, \
                        OPTICAL_SYSTEM_OFFSET[0], \
                        CANVAS_HEIGHT

        self._canvas.create_line(*optical_axis_points, arrow=tk.LAST)
        self._canvas.create_line(*y_axis_points, arrow=tk.FIRST)

    def _draw_components(self):
        """Draws components, it's boundaries, material etc."""
        def _fetch_boundaries():
            res = []
            for comp in self._optical_system._components:
                for l in comp._layers:
                   res.append(l.boundary)
            return res
        boundaries = _fetch_boundaries()
        for bound in boundaries:
            self._draw_curve(bound)

    def _draw_curve(self, boundary_func: Callable) -> None:
        assert isinstance(boundary_func, Callable)
        ys = range(BOUNDARY_DRAW_RANGES[0], BOUNDARY_DRAW_RANGES[1])
        zs = (boundary_func(y) for y in ys)
        points_to_draw = (convert_opticalcoords_to_tkcoords(z, y) for z, y in zip(zs, ys))
        self._canvas.create_line(*points_to_draw)

    def _draw_beams(self):
        """Draws beam lines on a canvas"""
        for b in self._optical_system._vectors.values():
            assert isinstance(b, list)
            self._draw_beam(b)

    def _draw_beam(self, beam):
        """Draws a single beam propagating throw optical system"""
        pass



if __name__ == '__main__':
    main()
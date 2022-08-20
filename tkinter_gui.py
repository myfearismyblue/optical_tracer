import tkinter as tk
from math import pi, tan, sin
from typing import Callable, Tuple, List

from optical_tracer import Layer, OpticalComponent, Material, OpticalSystem, Side, Vector, Point

DEBUG = False
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600
CANVAS_BACKGROUND_COLOR = 'white'
TIME_REFRESH = 100  # ms
SCALE = 1  # mm/px
ZOOMIN_TK_EVENT = '<Button-1>'  # you can change this while running on win or macos
ZOOMOUT_TK_EVENT = '<Button-2>'  # you can change this while running on win or macos
WHEEL_ZOOM_RATIO = 1.1  # while zooming with mouse wheel

# offset of entire optical system relatively to (0, 0) canvas point which is upper-left corner
OPTICAL_SYSTEM_OFFSET = (+1 * CANVAS_WIDTH // 3, +1 * CANVAS_HEIGHT // 3)  # in pixels here

# ranges in which components to be drawn relatively to OPTICAL_SYSTEM_OFFSET in pixels here
# coordinates are upside-down because of tkinter vertical axes is reverted
BOUNDARY_DRAW_RANGES = (OPTICAL_SYSTEM_OFFSET[1] - CANVAS_HEIGHT, OPTICAL_SYSTEM_OFFSET[1])


def main():
    global root, canvas
    model_objects = init_model_objects()

    root = tk.Tk()
    root.geometry(str(CANVAS_WIDTH) + 'x' + str(CANVAS_HEIGHT))
    canvas = tk.Canvas(root, background=CANVAS_BACKGROUND_COLOR)
    mouse_coords_text = canvas.create_text(2, 2, text=f'', font='28', justify='left', anchor='nw')
    canvas.itemconfig(mouse_coords_text, tags=('upper_coords', 'testtag'))
    canvas.focus_set()

    view_obj = (Grapher(canvas=canvas, opt_system=model_objects[0]),)

    canvas.bind('<Motion>', lambda event: key_handler(event, *view_obj))
    canvas.bind(ZOOMIN_TK_EVENT, lambda event: key_handler(event, *view_obj))
    canvas.bind(ZOOMOUT_TK_EVENT, lambda event: key_handler(event, *view_obj))
    canvas.pack(fill=tk.BOTH, expand=1)
    # tick(*objects)
    root.mainloop()


def convert_tkcoords_to_optical(tk_absciss: int, tk_ordinate: int, *, scale: float) -> Tuple[float, float]:
    """ Returns real coordinates in mm
    scale - in pixels per mm
    """
    opt_absciss = (tk_absciss - OPTICAL_SYSTEM_OFFSET[0]) / scale
    opt_ordinate = (OPTICAL_SYSTEM_OFFSET[1] - tk_ordinate) / scale
    return opt_absciss, opt_ordinate


def convert_opticalcoords_to_tkcoords(opt_absciss: float, opt_ordinate: float, *,
                                      scale: float,
                                      absciss_offset: int = OPTICAL_SYSTEM_OFFSET[0],
                                      ordinate_offset: int = OPTICAL_SYSTEM_OFFSET[1]
                                      ) -> Tuple[float, float]:
    """ Returns coordinates on canvas
        scale - in pixels per mm
        """
    tk_absciss = opt_absciss * scale + absciss_offset
    tk_ordinate = ordinate_offset - opt_ordinate * scale
    return tk_absciss, tk_ordinate


def tick(*objects):
    root.after(TIME_REFRESH, *objects)


def key_handler(event, *view_obj):
    gr = view_obj[0]
    mouse_coords = gr._canvas.find_withtag('upper_coords')
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
        opt_abs, opt_ord = convert_tkcoords_to_optical(event.x, event.y, scale=SCALE)
        canvas.itemconfig(mouse_coords, text=f'y: {opt_ord}, mm, z: {opt_abs}, mm')

    elif str(event.type) == '4':  # mouse button pressed
        if event.num == 1:
            gr._scale *= WHEEL_ZOOM_RATIO
            gr._offset = [int(_ * gr._scale) for _ in gr._offset]
        else:
            gr._scale /= WHEEL_ZOOM_RATIO
            gr._offset = [int(_ / gr._scale) for _ in gr._offset]
        gr._refresh_canvas()


def init_model_objects():
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

    opt_sys = create_opt_sys()
    in_point = Point(x=0, y=0, z=0)
    vs = (Vector(initial_point=in_point, lum=1, w_length=555, theta=t / 20, psi=0) for t in
          range(0, (int(2 * pi) * 20)))

    for v in vs:
        if v.theta == 0.65:
            opt_sys.trace(vector=v)
    objects = (opt_sys,)
    print('')
    return objects


class Grapher:
    """Responsible for drawing and refreshing visualities on a canvas"""

    def __init__(self, *, canvas, opt_system):
        self._canvas = canvas
        self._optical_system = opt_system
        self._scale = SCALE
        self._offset = (OPTICAL_SYSTEM_OFFSET[0], OPTICAL_SYSTEM_OFFSET[1])
        self._refresh_canvas()

    def _refresh_canvas(self):
        self._canvas.delete("all")
        self._draw_initial_axes()
        self._draw_components()
        self._draw_beams()

    def _draw_initial_axes(self):
        absciss_axis_points = 0, \
                              0 + self._offset[1], \
                              CANVAS_WIDTH, \
                              self._offset[1]

        ordinate_axis_points = self._offset[0], \
                               0, \
                               self._offset[0], \
                               CANVAS_HEIGHT

        self._canvas.create_line(*absciss_axis_points, arrow=tk.LAST)
        self._canvas.create_line(*ordinate_axis_points, arrow=tk.FIRST)

    def _draw_components(self):
        """Draws components, it's boundaries, material etc."""

        def _fetch_boundaries():
            res = []
            for comp in self._optical_system.components:
                for l in comp._layers:
                    res.append(l.boundary)
            return res

        boundaries = _fetch_boundaries()
        for bound in boundaries:
            self._draw_layers_bounds(bound)

    def _draw_layers_bounds(self, boundary_func: Callable) -> None:
        assert isinstance(boundary_func, Callable)
        ys_in_mm = (el * self._scale for el in range(BOUNDARY_DRAW_RANGES[0], BOUNDARY_DRAW_RANGES[1]))
        zs_in_mm = (boundary_func(y) for y in ys_in_mm)
        points_to_draw = list(convert_opticalcoords_to_tkcoords(z, y, scale=self._scale,
                                                                absciss_offset=self._offset[0],
                                                                ordinate_offset=self._offset[1]
                                                                ) for z, y in zip(zs_in_mm, ys_in_mm))
        self._canvas.create_line(*points_to_draw, smooth=True)

    def _draw_beams(self):
        """Draws beam lines on a canvas"""
        for b in self._optical_system._rays.values():
            assert isinstance(b, list)
            self._draw_beam(b)

    def _draw_beam(self, beam):
        """Draws a single beam propagating throw optical system"""
        zs_in_mm, ys_in_mm, ts = ([] for _ in range(3))
        [(zs_in_mm.append(vector.initial_point.z), ys_in_mm.append(vector.initial_point.y), ts.append(vector.theta)) for
         vector in beam]
        last_point_y = beam[-1].initial_point.y
        last_point_z = beam[-1].initial_point.z
        zs_in_mm.append(last_point_z)
        ys_in_mm.append(last_point_y)
        ts.append(ts[-1])
        # points_mm = list(zip(zs_in_mm, ys_in_mm))
        points_to_draw = [convert_opticalcoords_to_tkcoords(z, y, scale=self._scale,
                                                            absciss_offset=self._offset[0],
                                                            ordinate_offset=self._offset[1]) for z, y in
                          zip(zs_in_mm, ys_in_mm)]
        self._canvas.create_line(points_to_draw, arrow='last', fill='blue')

        # if len(points_to_draw)>=2:
        #     self._canvas.create_text(points_to_draw[1], text=f'{ts[1]}', font='28', justify='left', anchor='nw')
        def create_point(x, y, canvas, r=2):
            python_green = "#476042"
            x1, y1 = (x - r), (y - r)
            x2, y2 = (x + r), (y + r)
            canvas.create_oval(x1, y1, x2, y2, fill=python_green)

        [create_point(*point, self._canvas) for point in points_to_draw]
        # for i in range(len(points_to_draw)):
        self._canvas.create_text(*points_to_draw[-1], text=f' {ts[0]:.2f}', font='28', justify='left', anchor='nw')

    @staticmethod
    def _wavelength_to_rgb(wavelength, gamma=0.8):
        ...


if __name__ == '__main__':
    main()

import tkinter as tk
from typing import Callable, Tuple

from optical_tracer import Layer, OpticalComponent, Material, OpticalSystem, Side

DEBUG = True
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600
CANVAS_BACKGROUND_COLOR = 'white'

def main():
    global root, canvas
    objects = init_objects()

    root = tk.Tk()
    root.geometry(str(CANVAS_WIDTH) + 'x' + str(CANVAS_HEIGHT))
    canvas = tk.Canvas(root, background=CANVAS_BACKGROUND_COLOR)
    if DEBUG:
        mouse_coords_text = canvas.create_text(50, 10, text=f'', font='28')

    canvas.focus_set()

    objects = init_objects()
    canvas.bind('<Key>', lambda event: key_handler(event, *objects))
    canvas.pack(fill=tk.BOTH, expand=1)
    # tick(*objects)
    root.mainloop()


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
    objects = (create_opt_sys(),)
    return objects

if __name__ == '__main__':
    main()
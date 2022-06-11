import tkinter as tk
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600
CANVAS_BACKGROUND_COLOR = 'white'

def main():
    global root, canvas

    root = tk.Tk()
    root.geometry(str(CANVAS_WIDTH) + 'x' + str(CANVAS_HEIGHT))
    canvas = tk.Canvas(root, background=CANVAS_BACKGROUND_COLOR)
    if DEBUG:
        mouse_coords_text = canvas.create_text(50, 10, text=f'', font='28')

    canvas.focus_set()

    objects = init_objects()
    canvas.bind('<Key>', lambda event: key_handler(event, *objects))
    canvas.pack(fill=tk.BOTH, expand=1)
    tick(*objects)
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
    raise NotImplementedError
    return objects

if __name__ == '__main__':
    main()
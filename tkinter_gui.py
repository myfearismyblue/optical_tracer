import tkinter as tk
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600
CANVAS_BACKGROUND_COLOR = 'white'

def main():
    global root, canvas

    root = tk.Tk()
    root.geometry(str(CANVAS_WIDTH) + 'x' + str(CANVAS_HEIGHT))
    canvas = tk.Canvas(root, background=CANVAS_BACKGROUND_COLOR)
    canvas.focus_set()

    objects = init_objects()
    canvas.bind('<Key>', lambda event: key_handler(event, *objects))
    canvas.pack(fill=tk.BOTH, expand=1)
    tick(*objects)
    root.mainloop()


def tick(*objects):
    raise NotImplementedError

def key_handler(event, *objects):
    raise NotImplementedError


def init_objects():
    raise NotImplementedError
    return objects

if __name__ == '__main__':
    main()
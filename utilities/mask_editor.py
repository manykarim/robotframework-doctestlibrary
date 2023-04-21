import tkinter as tk
from tkinter import filedialog
import cv2
import json


class ImageEditor:
    def __init__(self, master):
        self.master = master
        self.master.title("Mask Editor")

        # Make resizable canvas and scrollbars
        self.canvas_frame = tk.Frame(self.master)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.hscrollbar = tk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL)
        self.hscrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.vscrollbar = tk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL)
        self.vscrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas = tk.Canvas(self.canvas_frame, width=800, height=600,
                                xscrollcommand=self.hscrollbar.set,
                                yscrollcommand=self.vscrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.hscrollbar.config(command=self.canvas.xview)
        self.vscrollbar.config(command=self.canvas.yview)

        self.rectangles = []

        # create menu bar
        menubar = tk.Menu(self.master)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open", command=self.load_image)
        filemenu.add_command(label="Export", command=self.export_rectangles)
        filemenu.add_command(label="Load mask", command=self.load_rectangles)
        menubar.add_cascade(label="File", menu=filemenu)
        self.master.config(menu=menubar)

        # bind mouse events to canvas
        self.canvas.bind("<Button-1>", self.start_rect)
        self.canvas.bind("<B1-Motion>", self.draw_rect)
        self.canvas.bind("<ButtonRelease-1>", self.end_rect)

        # bind right click to delete rectangle
        self.canvas.bind("<Button-3>", self.delete_rect)

        # bind resize event
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        # bind mousewheel to scrollbars
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)

        # bind horizontal mousewheel to canvas
        self.canvas.bind("<Shift-MouseWheel>", self.on_shift_mousewheel)



    def on_mousewheel(self, event):
        """Scroll canvas on mousewheel event"""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def on_shift_mousewheel(self, event):
        """Scroll canvas horizontally on shift+mousewheel event"""
        self.canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")


    def on_canvas_resize(self, event):
        """Adjust scrollbars when canvas size changes"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def load_image(self):
        filename = filedialog.askopenfilename(title="Select an image",
                                              filetypes=(("PNG files", "*.png"), ("JPEG files", "*.jpg *.jpeg")))
        if filename:
            self.filename = filename
            self.image = cv2.imread(self.filename)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.photo = tk.PhotoImage(
                data=cv2.imencode(".png", self.image)[1].tobytes())
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)            

    def load_rectangles(self):
        filename = filedialog.askopenfilename(title="Select a mask",
                                              filetypes=(("JSON files", "*.json"), ("All files", "*.*")))
        if filename:
            with open(filename, "r") as f:
                rectangles = json.load(f)
                for rect in rectangles:
                    x = rect["x"]
                    y = rect["y"]
                    w = rect["width"]
                    h = rect["height"]
                    self.rectangles.append((x, y, w, h))
                    self.canvas.create_rectangle(
                        x, y, x + w, y + h, outline="red")

    def start_rect(self, event):
        self.rect_start = (event.x, event.y)
        self.rect = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline="red")

    def draw_rect(self, event):
        self.canvas.coords(
            self.rect, self.rect_start[0], self.rect_start[1], event.x, event.y)

    def end_rect(self, event):
        coords = self.canvas.coords(self.rect)
        x = min(coords[0], coords[2])
        y = min(coords[1], coords[3])
        w = abs(coords[0] - coords[2])
        h = abs(coords[1] - coords[3])
        self.rectangles.append((x, y, w, h))

    def delete_rect(self, event):
        """"
        Delete the rectangle that was clicked on
        """
        for rect in self.rectangles:
            x, y, w, h = rect
            if x <= event.x <= x + w and y <= event.y <= y + h:
                self.rectangles.remove(rect)
                break
        self.render_rectangles()

    def render_rectangles(self):
        """
        Render rectangles on canvas
        """
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        for rect in self.rectangles:
            x, y, w, h = rect
            self.canvas.create_rectangle(x, y, x + w, y + h, outline="red")

    def export_rectangles(self):
        """"
        Save rectangles to JSON file
        Create a list of dictionaries, each dictionary is a rectangle
        "page": all,
        "name": "Rectangle 1",
        "type": "coordinates",
        "x": 31,
        "y": 31,
        "height": 14,
        "width": 84
        """
        filename = filedialog.asksaveasfilename(title="Export Rectangles",
                                                filetypes=(
                                                    ("JSON files", "*.json"), ("All files", "*.*")),
                                                defaultextension=".json")
        if filename:
            with open(filename, "w") as f:
                rectangles = []
                for i, rect in enumerate(self.rectangles):
                    x, y, w, h = rect
                    rectangles.append({"page": "all",
                                       "name": f"Rectangle {i}",
                                       "type": "coordinates",
                                       "x": x,
                                       "y": y,
                                       "height": h,
                                       "width": w})
                json.dump(rectangles, f, indent=4)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEditor(root)
    root.mainloop()

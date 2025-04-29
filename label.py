import os
import json
from pathlib import Path
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk


def get_image_label_pairs(base_dir):
    img_root = Path(base_dir) / "Imagenes"
    label_root = Path(base_dir) / "Etiquetas"
    pairs = []
    for split in ["train", "test", "val", "validation/val"]:
        img_dir = img_root / split
        label_dir = label_root / split
        if not img_dir.exists():
            continue
        for img_file in img_dir.glob("*.jpg"):
            label_file = label_dir / img_file.with_suffix(".txt").name
            if label_file.exists():
                pairs.append((img_file, label_file))
    return pairs

def parse_label_file(label_path, img_width, img_height):
    with open(label_path, "r") as f:
        lines = f.readlines()
    boxes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id, x_center, y_center, w, h = map(float, parts)
        x1 = int((x_center - w / 2) * img_width)
        y1 = int((y_center - h / 2) * img_height)
        x2 = int((x_center + w / 2) * img_width)
        y2 = int((y_center + h / 2) * img_height)
        boxes.append((int(cls_id), x1, y1, x2, y2))
    return boxes

class ClassMapInspector:
    def __init__(self, root, pairs):
        self.root = root
        self.pairs = pairs
        self.index = 0
        self.canvas = tk.Canvas(root, width=800, height=600)
        self.canvas.pack()
        self.class_map = {}
        self.seen_classes = set()
        self.load_next_image()

    def load_next_image(self):
        if self.index >= len(self.pairs):
            self.save_class_map()
            messagebox.showinfo("Done", "Finished checking all labels.")
            self.root.quit()
            return

        img_path, label_path = self.pairs[self.index]
        self.curr_img_path = img_path
        self.curr_label_path = label_path

        self.orig_image = Image.open(img_path)
        orig_w, orig_h = self.orig_image.size

        max_w, max_h = 800, 600
        scale = min(max_w / orig_w, max_h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        resized_image = self.orig_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        self.tk_img = ImageTk.PhotoImage(resized_image)
        self.canvas.config(width=new_w, height=new_h)

        self.scale = scale
        self.curr_boxes = parse_label_file(label_path, orig_w, orig_h)

        self.draw_boxes()
        self.query_unknown_classes()

    def query_unknown_classes(self):
        unknown_classes = {cls_id for cls_id, *_ in self.curr_boxes} - self.seen_classes
        if unknown_classes:
            for cls_id in unknown_classes:
                name = simpledialog.askstring("Class Name", f"What is class {cls_id}?")
                if name is None:
                    name = f"class_{cls_id}"
                
                self.class_map[cls_id] = name
                self.seen_classes.add(cls_id)
                
                if 'rio' in self.class_map.values() and 'carretera' in self.class_map.values():
                    self.save_class_map()
                    messagebox.showinfo("Finished", "Both classes 'rio' and 'carretera' found. Exiting...")
                    self.root.quit()
                    return

        self.index += 1
        self.load_next_image()

    def draw_boxes(self):
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        for cls_id, x1, y1, x2, y2 in self.curr_boxes:
            label = self.class_map.get(cls_id, str(cls_id))
            self.canvas.create_rectangle(
                x1 * self.scale, y1 * self.scale, x2 * self.scale, y2 * self.scale,
                outline="red", width=2
            )
            self.canvas.create_text(
                x1 * self.scale + 5, y1 * self.scale + 10,
                anchor="nw", text=label, fill="white", font=("Arial", 12, "bold")
            )

    def save_class_map(self):
        with open("label_map.json", "w") as f:
            json.dump(self.class_map, f, indent=2)

if __name__ == "__main__":
    base_path = "data"
    pairs = get_image_label_pairs(base_path)
    root = tk.Tk()
    app = ClassMapInspector(root, pairs)
    root.mainloop()

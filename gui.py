
import tkinter as tk
from tkinter import Canvas, Button, Label
import torch
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from model import Net

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")

        self.canvas = Canvas(root, width=280, height=280, bg="white")
        self.canvas.pack(pady=10)

        self.predict_button = Button(root, text="Predict", command=self.predict_digit)
        self.predict_button.pack(side=tk.LEFT, padx=10)

        self.clear_button = Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.RIGHT, padx=10)

        self.prediction_label = Label(root, text="", font=("Helvetica", 24))
        self.prediction_label.pack(pady=20)

        self.model = self.load_model()
        self.image = Image.new("L", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

    def load_model(self):
        model = Net()
        model.load_state_dict(torch.load("models/mnist_cnn_subset_1200.pt"))
        model.eval()
        return model

    def paint(self, event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=10)
        self.draw.ellipse([x1, y1, x2, y2], fill="black")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.prediction_label.config(text="")

    def predict_digit(self):
        # Resize and invert the image
        img = self.image.resize((28, 28))
        img = ImageOps.invert(img)

        # Convert to tensor and normalize
        img_array = np.array(img)
        img_tensor = torch.from_numpy(img_array).float().unsqueeze(0).unsqueeze(0)
        img_tensor = img_tensor / 255.0
        img_tensor = (img_tensor - 0.5) / 0.5

        with torch.no_grad():
            output = self.model(img_tensor)
            prediction = output.argmax(dim=1).item()
            self.prediction_label.config(text=f"Prediction: {prediction}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()

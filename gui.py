import tkinter as tk
from tkinter import ttk
import torch
import numpy as np
from model import Net

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")

        self.pixel_size = 10
        self.grid_size = 28
        self.canvas_size = self.grid_size * self.pixel_size

        main_frame = tk.Frame(root)
        main_frame.pack(padx=10, pady=10)

        self.canvas = tk.Canvas(main_frame, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.pack(side=tk.LEFT)

        confidence_frame = tk.Frame(main_frame, padx=10)
        confidence_frame.pack(side=tk.RIGHT)

        self.confidence_labels = []
        self.confidence_bars = []
        for i in range(10):
            label_frame = tk.Frame(confidence_frame)
            label_frame.pack(fill="x")
            label = tk.Label(label_frame, text=f"{i}:", font=("Helvetica", 12), width=3)
            label.pack(side=tk.LEFT)
            bar = ttk.Progressbar(label_frame, length=100, mode='determinate')
            bar.pack(side=tk.LEFT, padx=5)
            percent_label = tk.Label(label_frame, text="0.00%", font=("Helvetica", 12), width=7, anchor="w")
            percent_label.pack(side=tk.LEFT)
            self.confidence_labels.append((label, percent_label))
            self.confidence_bars.append(bar)

        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(pady=10)

        self.prediction_label = tk.Label(root, text="Prediction: ", font=("Helvetica", 24))
        self.prediction_label.pack(pady=10)

        self.model = self.load_model()
        self.grid = np.zeros((self.grid_size, self.grid_size))

        self.canvas.bind("<B1-Motion>", self.paint)
        self.draw_grid()

    def draw_grid(self):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x1 = j * self.pixel_size
                y1 = i * self.pixel_size
                x2 = x1 + self.pixel_size
                y2 = y1 + self.pixel_size
                self.canvas.create_rectangle(x1, y1, x2, y2, fill="white", outline="gray")

    def load_model(self):
        model = Net()
        model.load_state_dict(torch.load("models/mnist_cnn_subset_1200.pt"))
        model.eval()
        return model

    def paint(self, event):
        x, y = event.x, event.y
        grid_x, grid_y = x // self.pixel_size, y // self.pixel_size

        if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
            self.grid[grid_y, grid_x] = 1.0
            x1 = grid_x * self.pixel_size
            y1 = grid_y * self.pixel_size
            x2 = x1 + self.pixel_size
            y2 = y1 + self.pixel_size
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", outline="black")
            self.predict_realtime()

    def clear_canvas(self):
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.canvas.delete("all")
        self.draw_grid()
        self.prediction_label.config(text="Prediction: ")
        for i in range(10):
            self.confidence_labels[i][0].config(bg="SystemButtonFace")
            self.confidence_labels[i][1].config(text="0.00%")
            self.confidence_bars[i]['value'] = 0

    def predict_realtime(self):
        img_tensor = torch.from_numpy(self.grid).float().unsqueeze(0).unsqueeze(0)
        img_tensor = (img_tensor - 0.5) / 0.5

        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = torch.exp(output).squeeze()
            
            prediction = torch.argmax(probabilities).item()

            for i in range(10):
                confidence = probabilities[i].item() * 100
                self.confidence_labels[i][1].config(text=f"{confidence:.2f}%")
                self.confidence_bars[i]['value'] = confidence
                if i == prediction:
                    self.confidence_labels[i][0].config(bg="yellow")
                else:
                    self.confidence_labels[i][0].config(bg="SystemButtonFace")

            self.prediction_label.config(text=f"Prediction: {prediction}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
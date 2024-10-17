from keras.models import load_model
from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image
import numpy as np

# Load the pre-trained model
model = load_model('mnist_augmented.h5')

def predict_digit(img):
    # Resize the image to 28x28 pixels
    img = img.resize((28, 28))
    # Convert to grayscale
    img = img.convert('L')
    img = np.array(img)
    
    # Invert colors: make the drawing white on black background
    img = 255 - img
    
    # Reshape the image for model input and normalize
    img = img.reshape(1, 28, 28, 1)
    img = img / 255.0
    
    # Predict the digit
    res = model.predict([img])[0]
    return np.argmax(res), max(res)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Digit Recognizer")
        self.geometry("500x350")
        self.configure(bg="#f7f7f7")
        
        # Variables to track the drawing position
        self.x = self.y = 0

        # Create the canvas where the user will draw
        self.canvas = tk.Canvas(self, width=300, height=300, bg="white", cursor="cross")
        self.canvas.grid(row=0, column=0, pady=2, padx=2)

        # Label to display the prediction
        self.label = tk.Label(self, text="Draw a digit", font=("Helvetica", 30), bg="#f7f7f7")
        self.label.grid(row=0, column=1, pady=2, padx=2)

        # Button to classify the handwriting
        self.classify_btn = tk.Button(self, text="Recognize", command=self.classify_handwriting, 
                                      font=("Helvetica", 14), bg="#4CAF50", fg="white")
        self.classify_btn.grid(row=1, column=1, pady=10)

        # Button to clear the canvas
        self.clear_btn = tk.Button(self, text="Clear", command=self.clear_all, 
                                   font=("Helvetica", 14), bg="#f44336", fg="white")
        self.clear_btn.grid(row=1, column=0, pady=10)

        # Bind mouse events to draw on the canvas
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        """Clear the canvas."""
        self.canvas.delete("all")
        self.label.configure(text="Draw a digit")

    def classify_handwriting(self):
        """Capture the canvas and predict the digit."""
        HWND = self.canvas.winfo_id()  # Get the handle of the canvas
        rect = win32gui.GetWindowRect(HWND)  # Get the canvas' coordinates
        im = ImageGrab.grab(rect)  # Capture the canvas image

        # Predict the digit and display it
        digit, acc = predict_digit(im)
        self.label.configure(text=f"{digit}, {int(acc * 100)}%")

    def draw_lines(self, event):
        """Draw on the canvas."""
        self.x, self.y = event.x, event.y
        r = 8
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')

# Run the application
app = App()
app.mainloop()

import cv2
import numpy as np
from tkinter import Tk, Label, Button, filedialog, Scale, HORIZONTAL, Canvas, StringVar, Entry, IntVar
from PIL import Image, ImageTk
import threading

class CircleDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Circle Detection App")
        
        # Inisialisasi variabel
        self.image = None
        self.processed_image = None
        self.filename = None

        # Status Loading
        self.status_var = StringVar()
        self.status_var.set("Welcome to Circle Detection App!")

        # Frame atas untuk tombol dan slider
        control_frame = Canvas(root)
        control_frame.pack(side="top", fill="x", pady=5)
        
        # Tombol
        Button(control_frame, text="Load Image", command=self.load_image).pack(side="left", padx=10)
        Button(control_frame, text="Detect Circles", command=self.run_detection).pack(side="left", padx=10)
        Button(control_frame, text="Save Image", command=self.save_image).pack(side="left", padx=10)
        
        # Parameter 1
        Label(control_frame, text="Param1 (Edge Detection)").pack(side="left", padx=5)
        self.param1 = IntVar(value=100)
        self.param1_entry = Entry(control_frame, textvariable=self.param1, width=5)
        self.param1_entry.pack(side="left", padx=5)
        Button(control_frame, text="▲", command=lambda: self.modify_param(self.param1, 1)).pack(side="left", padx=2)
        Button(control_frame, text="▼", command=lambda: self.modify_param(self.param1, -1)).pack(side="left", padx=2)

        # Parameter 2
        Label(control_frame, text="Param2 (Circle Sensitivity)").pack(side="left", padx=5)
        self.param2 = IntVar(value=30)
        self.param2_entry = Entry(control_frame, textvariable=self.param2, width=5)
        self.param2_entry.pack(side="left", padx=5)
        Button(control_frame, text="▲", command=lambda: self.modify_param(self.param2, 1)).pack(side="left", padx=2)
        Button(control_frame, text="▼", command=lambda: self.modify_param(self.param2, -1)).pack(side="left", padx=2)
        
        # Label status
        self.status_label = Label(root, textvariable=self.status_var, anchor="w", fg="blue")
        self.status_label.pack(fill="x", pady=5)

        # Canvas untuk menampilkan gambar
        self.image_canvas = Canvas(root, width=800, height=600, bg="gray")
        self.image_canvas.pack()

    def load_image(self):
        self.filename = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if self.filename:
            self.image = cv2.imread(self.filename)
            self.processed_image = None  # Reset processed image
            self.display_image(self.image)
            self.status_var.set("Image loaded successfully!")

    def run_detection(self):
        if self.image is None:
            self.status_var.set("Please load an image first.")
            return

        # Tampilkan status "Loading"
        self.status_var.set("Detecting circles... Please wait.")
        
        # Jalankan deteksi di thread terpisah
        threading.Thread(target=self.detect_circles).start()

    def detect_circles(self):
        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)

        # Hough Circle Detection
        param1 = self.param1.get()
        param2 = self.param2.get()
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=param1,
            param2=param2,
            minRadius=0,
            maxRadius=0
        )

        # Copy the original image for drawing
        output = self.image.copy()
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # Draw the outer circle
                cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # Draw the center of the circle
                cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)

        # Simpan hasilnya
        self.processed_image = output

        # Update tampilan gambar
        self.display_image(output)

        # Update status
        self.status_var.set("Circle detection completed!")

    def save_image(self):
        if self.processed_image is not None:
            save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                     filetypes=[("PNG files", "*.png"),
                                                                ("JPEG files", "*.jpg"),
                                                                ("All files", "*.*")])
            if save_path:
                cv2.imwrite(save_path, self.processed_image)
                self.status_var.set(f"Image saved at {save_path}!")
        else:
            self.status_var.set("No processed image to save.")

    def display_image(self, image):
        # Convert BGR to RGB for Tkinter display
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb_image)

        # Resize image agar sesuai dengan canvas
        im.thumbnail((800, 600), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=im)

        # Tampilkan gambar di canvas
        self.image_canvas.delete("all")
        self.image_canvas.create_image(400, 300, image=imgtk, anchor="center")
        self.image_canvas.image = imgtk

    def modify_param(self, param_var, increment):
        """Modifikasi nilai parameter dengan tombol up/down."""
        current_value = param_var.get()
        new_value = max(0, current_value + increment)  # Jangan sampai nilai negatif
        param_var.set(new_value)


# Main program
if __name__ == "__main__":
    root = Tk()
    app = CircleDetectionApp(root)
    root.mainloop()

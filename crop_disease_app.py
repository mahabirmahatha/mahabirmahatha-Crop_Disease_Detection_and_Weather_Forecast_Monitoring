# PROJECT 10: CROP DISEASE DETECTION SYSTEM




import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import tensorflow as tf

from PIL import Image, ImageTk, ImageDraw


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = "crop_disease_model.keras"

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "Healthy",
    "Leaf Blight",
    "Rust",
    "Early Blight",
    "Septoria"
]

COLORS = {
    "Healthy": "#22c55e",
    "Leaf Blight": "#facc15",
    "Rust": "#f97316",
    "Early Blight": "#a855f7",
    "Septoria": "#38bdf8"
}


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = None

if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as error:
        print("Model loading error:", error)


# ==========================================
# MAIN APPLICATION
# ==========================================

class CropDiseaseApp:

    def __init__(self, root):

        self.root = root

        self.root.title("Crop Disease Detection")
        self.root.geometry("1100x750")
        self.root.minsize(850, 650)
        self.root.configure(bg="#07152f")

        self.image_path = None
        self.preview_image = None

        self.create_interface()

    # ======================================
    # INTERFACE
    # ======================================

    def create_interface(self):

        header = tk.Frame(
            self.root,
            bg="#07152f",
            height=70
        )

        header.pack(fill="x")

        tk.Label(
            header,
            text="🌿  Crop Disease Detection",
            font=("Arial", 22, "bold"),
            fg="white",
            bg="#07152f"
        ).pack(side="left", padx=25, pady=18)

        tk.Label(
            header,
            text="AI POWERED",
            font=("Arial", 10, "bold"),
            fg="#22c55e",
            bg="#07152f"
        ).pack(side="right", padx=25)

        main = tk.Frame(
            self.root,
            bg="#07152f"
        )

        main.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=10
        )

        # Image panel

        self.image_panel = tk.Frame(
            main,
            bg="#10264a",
            highlightbackground="#2563eb",
            highlightthickness=2
        )

        self.image_panel.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(0, 10)
        )

        tk.Label(
            self.image_panel,
            text="Crop Leaf Image",
            font=("Arial", 17, "bold"),
            fg="white",
            bg="#10264a"
        ).pack(pady=15)

        self.image_display = tk.Label(
            self.image_panel,
            text="🌱\n\nSelect a leaf image to begin",
            font=("Arial", 17),
            fg="#94a3b8",
            bg="#07152f"
        )

        self.image_display.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=10
        )

        tk.Button(
            self.image_panel,
            text="📁  Upload Leaf Image",
            font=("Arial", 12, "bold"),
            bg="#16a34a",
            fg="white",
            activebackground="#15803d",
            relief="flat",
            cursor="hand2",
            command=self.upload_image
        ).pack(
            fill="x",
            padx=20,
            pady=20,
            ipady=10
        )

        # Results panel

        self.result_panel = tk.Frame(
            main,
            bg="#10264a",
            highlightbackground="#2563eb",
            highlightthickness=2,
            width=350
        )

        self.result_panel.pack(
            side="right",
            fill="both",
            padx=(10, 0)
        )

        self.result_panel.pack_propagate(False)

        tk.Label(
            self.result_panel,
            text="🌿  Detection Result",
            font=("Arial", 18, "bold"),
            fg="white",
            bg="#10264a"
        ).pack(pady=20)

        self.status_label = tk.Label(
            self.result_panel,
            text="Waiting for Image",
            font=("Arial", 12, "bold"),
            fg="#38bdf8",
            bg="#10264a"
        )

        self.status_label.pack(pady=10)

        self.disease_label = tk.Label(
            self.result_panel,
            text="No Detection",
            font=("Arial", 22, "bold"),
            fg="white",
            bg="#10264a",
            wraplength=300
        )

        self.disease_label.pack(pady=15)

        self.confidence_label = tk.Label(
            self.result_panel,
            text="Confidence: --",
            font=("Arial", 15),
            fg="#22c55e",
            bg="#10264a"
        )

        self.confidence_label.pack(pady=10)

        # Progress bar

        self.progress = ttk.Progressbar(
            self.result_panel,
            orient="horizontal",
            length=250,
            mode="determinate",
            maximum=100
        )

        self.progress.pack(
            padx=25,
            pady=15,
            fill="x"
        )

        tk.Label(
            self.result_panel,
            text="Crop Health Overview",
            font=("Arial", 15, "bold"),
            fg="white",
            bg="#10264a"
        ).pack(pady=15)

        self.overview_frame = tk.Frame(
            self.result_panel,
            bg="#10264a"
        )

        self.overview_frame.pack(
            fill="x",
            padx=15
        )

        self.overview_labels = {}

        for disease in CLASS_NAMES:

            row = tk.Frame(
                self.overview_frame,
                bg="#10264a"
            )

            row.pack(fill="x", pady=7)

            tk.Label(
                row,
                text=disease,
                font=("Arial", 10),
                fg="white",
                bg="#10264a",
                width=14,
                anchor="w"
            ).pack(side="left")

            label = tk.Label(
                row,
                text="0%",
                font=("Arial", 10, "bold"),
                fg=COLORS[disease],
                bg="#10264a"
            )

            label.pack(side="right")

            self.overview_labels[disease] = label

        # Bottom controls

        controls = tk.Frame(
            self.result_panel,
            bg="#10264a"
        )

        controls.pack(
            side="bottom",
            fill="x",
            padx=15,
            pady=20
        )

        tk.Button(
            controls,
            text="Analyze Image",
            bg="#16a34a",
            fg="white",
            font=("Arial", 11, "bold"),
            relief="flat",
            command=self.detect_disease
        ).pack(
            fill="x",
            pady=5,
            ipady=9
        )

        tk.Button(
            controls,
            text="Reset",
            bg="#2563eb",
            fg="white",
            font=("Arial", 11, "bold"),
            relief="flat",
            command=self.reset
        ).pack(
            fill="x",
            pady=5,
            ipady=9
        )

        # Footer

        footer = tk.Label(
            self.root,
            text="Crop Disease Detection System | TensorFlow + OpenCV",
            font=("Arial", 9),
            fg="#94a3b8",
            bg="#07152f"
        )

        footer.pack(pady=10)

    # ======================================
    # UPLOAD IMAGE
    # ======================================

    def upload_image(self):

        path = filedialog.askopenfilename(
            title="Select Crop Leaf Image",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
                ("All Files", "*.*")
            ]
        )

        if not path:
            return

        try:

            image = Image.open(path).convert("RGB")

            self.image_path = path

            image.thumbnail((600, 500))

            self.preview_image = ImageTk.PhotoImage(image)

            self.image_display.configure(
                image=self.preview_image,
                text=""
            )

            self.status_label.configure(
                text="Image Uploaded",
                fg="#38bdf8"
            )

            self.disease_label.configure(
                text="Ready to Analyze",
                fg="white"
            )

            self.confidence_label.configure(
                text="Confidence: --"
            )

            self.progress["value"] = 0

            for label in self.overview_labels.values():
                label.configure(text="0%")

        except Exception as error:

            messagebox.showerror(
                "Image Error",
                f"Unable to open image:\n{error}"
            )

    # ======================================
    # PREPROCESS IMAGE
    # ======================================

    def preprocess_image(self):

        image = Image.open(
            self.image_path
        ).convert("RGB")

        image = image.resize(IMAGE_SIZE)

        image_array = np.array(
            image,
            dtype=np.float32
        )

        image_array = image_array / 255.0

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        return image_array

    # ======================================
    # DISEASE DETECTION
    # ======================================

    def detect_disease(self):

        if not self.image_path:

            messagebox.showwarning(
                "No Image",
                "Please upload a crop leaf image first."
            )

            return

        if model is None:

            messagebox.showerror(
                "Model Not Found",
                "The trained AI model was not found.\n\n"
                "Place crop_disease_model.keras in the "
                "same folder as this Python file."
            )

            self.status_label.configure(
                text="AI Model Required",
                fg="#facc15"
            )

            return

        try:

            self.status_label.configure(
                text="Analyzing Image...",
                fg="#38bdf8"
            )

            self.root.update_idletasks()

            image_array = self.preprocess_image()

            predictions = model.predict(
                image_array,
                verbose=0
            )[0]

            if len(predictions) != len(CLASS_NAMES):

                raise ValueError(
                    "The model output does not match "
                    "the configured disease classes."
                )

            # Convert predictions to probabilities
            # only if the model outputs logits.

            if (
                np.any(predictions < 0)
                or not np.isclose(
                    np.sum(predictions),
                    1.0,
                    atol=0.02
                )
            ):

                exp_predictions = np.exp(
                    predictions - np.max(predictions)
                )

                predictions = (
                    exp_predictions /
                    np.sum(exp_predictions)
                )

            class_index = int(
                np.argmax(predictions)
            )

            disease = CLASS_NAMES[class_index]

            confidence = float(
                predictions[class_index] * 100
            )

            self.show_results(
                disease,
                confidence,
                predictions
            )

        except Exception as error:

            messagebox.showerror(
                "Detection Error",
                str(error)
            )

            self.status_label.configure(
                text="Detection Failed",
                fg="#ef4444"
            )

    # ======================================
    # DISPLAY RESULTS
    # ======================================

    def show_results(
        self,
        disease,
        confidence,
        predictions
    ):

        self.status_label.configure(
            text="✓ Detection Complete",
            fg="#22c55e"
        )

        self.disease_label.configure(
            text=disease,
            fg=COLORS[disease]
        )

        self.confidence_label.configure(
            text=f"Confidence: {confidence:.2f}%"
        )

        self.progress["value"] = confidence

        for index, name in enumerate(CLASS_NAMES):

            percentage = predictions[index] * 100

            self.overview_labels[name].configure(
                text=f"{percentage:.1f}%"
            )

    # ======================================
    # RESET APPLICATION
    # ======================================

    def reset(self):

        self.image_path = None
        self.preview_image = None

        self.image_display.configure(
            image="",
            text="🌱\n\nSelect a leaf image to begin"
        )

        self.status_label.configure(
            text="Waiting for Image",
            fg="#38bdf8"
        )

        self.disease_label.configure(
            text="No Detection",
            fg="white"
        )

        self.confidence_label.configure(
            text="Confidence: --"
        )

        self.progress["value"] = 0

        for label in self.overview_labels.values():

            label.configure(
                text="0%"
            )


# ==========================================
# START APPLICATION
# ==========================================

if __name__ == "__main__":

    root = tk.Tk()

    app = CropDiseaseApp(root)

    root.mainloop()

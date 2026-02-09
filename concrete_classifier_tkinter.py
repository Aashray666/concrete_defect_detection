# concrete_classifier_tkinter.py

import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# ----- PyTorch imports (3-class model) -----
import torch
import torch.nn as nn
from torchvision import models, transforms as tv_transforms

# ----- TensorFlow / Keras imports (2-class model) -----
import numpy as np
import tensorflow as tf
from tensorflow import keras

# ==================== CONFIGURATION ====================

IMG_SIZE = 224

# 3-class PyTorch model: honeycombing / crack / spalling [attached_file:1]
PYTORCH_CLASS_NAMES = ["honeycombing", "crack", "spalling"]
PYTORCH_MODEL_PATH = "./models/best_model_3class_hcs.pth"  # <- update if needed

# 2-class Keras model: spalling / void (binary classifier) [attached_file:2]
KERAS_CLASS_NAMES = ["spalling", "void"]  # order must match your training label encoding
KERAS_MODEL_PATH = "./models/spalling_void_model.keras"    # .keras or .h5, update as needed

# PyTorch validation transform (matches training notebook) [attached_file:1]
pytorch_transform = tv_transforms.Compose([
    tv_transforms.Resize((IMG_SIZE, IMG_SIZE)),
    tv_transforms.ToTensor(),
    tv_transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ==================== KERAS PREPROCESS (MOBILENETV2 STYLE) ====================

def mnv2_preprocess(x: np.ndarray) -> np.ndarray:
    """
    Minimal MobileNetV2-style preprocessing:
    - Input: RGB image array in [0, 255], dtype float32 or convertible to it
    - Output: scaled to [-1, 1], as expected by MobileNetV2. [attached_file:2]
    """
    x = x.astype("float32")
    return (x / 127.5) - 1.0

# ==================== MODEL LOADERS ====================

def load_pytorch_model(model_path: str):
    """
    Load the 3-class ResNet50 PyTorch model. [attached_file:1]
    Handles both raw state_dict and checkpoint dicts.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"PyTorch model not found at: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, len(PYTORCH_CLASS_NAMES)),
    )

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, device


def load_keras_model_local(model_path: str):
    """
    Load the 2-class Keras model (MobileNetV2-based). [attached_file:2]
    Works with both .keras and .h5 formats saved via model.save().
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Keras model not found at: {model_path}")

    model = keras.models.load_model(model_path)
    return model

# ==================== GUI APPLICATION ====================

class ConcreteDefectClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🏗️ Concrete Defect Classifier (PyTorch + Keras)")
        self.root.geometry("1000x750")
        self.root.configure(bg="#f0f0f0")

        # Runtime state
        self.current_image = None
        self.current_backend = None  # "pytorch" or "keras"
        self.current_class_names = []
        self.model_loaded = False
        self.error_message = ""

        # Model handles
        self.pytorch_model = None
        self.pytorch_device = None
        self.keras_model = None

        self.create_widgets()

        # Default model: 3-class PyTorch
        self.model_var.set("pytorch")
        self.switch_model()

    # ---------- UI CREATION ----------

    def create_widgets(self):
        # Title Frame
        title_frame = tk.Frame(self.root, bg="#2c3e50", height=80)
        title_frame.pack(fill=tk.X, pady=(0, 10))

        title_label = tk.Label(
            title_frame,
            text="🏗️ Concrete Defect Classification System",
            font=("Arial", 20, "bold"),
            bg="#2c3e50",
            fg="white",
        )
        title_label.pack(pady=10)

        # Model selection frame
        model_frame = tk.Frame(self.root, bg="#f0f0f0")
        model_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            model_frame,
            text="Select Model:",
            font=("Arial", 12, "bold"),
            bg="#f0f0f0",
        ).pack(side=tk.LEFT, padx=(20, 10))

        self.model_var = tk.StringVar(value="pytorch")

        tk.Radiobutton(
            model_frame,
            text="3-Class (PyTorch: honeycombing/crack/spalling)",
            variable=self.model_var,
            value="pytorch",
            command=self.switch_model,
            bg="#f0f0f0",
            anchor="w",
        ).pack(side=tk.LEFT, padx=10)

        tk.Radiobutton(
            model_frame,
            text="2-Class (Keras: spalling/void)",
            variable=self.model_var,
            value="keras",
            command=self.switch_model,
            bg="#f0f0f0",
            anchor="w",
        ).pack(side=tk.LEFT, padx=10)

        # Main container
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Left panel - Image display
        left_frame = tk.Frame(main_frame, bg="white", relief=tk.RAISED, borderwidth=2)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        tk.Label(
            left_frame,
            text="📷 Image Preview",
            font=("Arial", 14, "bold"),
            bg="white",
        ).pack(pady=10)

        self.image_label = tk.Label(left_frame, bg="white", text="No image loaded")
        self.image_label.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

        # Right panel - Results
        right_frame = tk.Frame(main_frame, bg="white", relief=tk.RAISED, borderwidth=2)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(
            right_frame,
            text="📊 Classification Results",
            font=("Arial", 14, "bold"),
            bg="white",
        ).pack(pady=10)

        self.result_text = tk.Text(
            right_frame,
            height=25,
            width=45,
            font=("Courier", 11),
            bg="#f9f9f9",
            relief=tk.FLAT,
            padx=10,
            pady=10,
        )
        self.result_text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        # Button frame
        button_frame = tk.Frame(self.root, bg="#f0f0f0")
        button_frame.pack(pady=10)

        self.upload_btn = tk.Button(
            button_frame,
            text="📁 Upload Image",
            command=self.upload_image,
            font=("Arial", 12, "bold"),
            bg="#3498db",
            fg="white",
            padx=20,
            pady=10,
            cursor="hand2",
        )
        self.upload_btn.pack(side=tk.LEFT, padx=10)

        self.classify_btn = tk.Button(
            button_frame,
            text="🔍 Classify Defect",
            command=self.classify_image,
            font=("Arial", 12, "bold"),
            bg="#27ae60",
            fg="white",
            padx=20,
            pady=10,
            cursor="hand2",
            state=tk.DISABLED,
        )
        self.classify_btn.pack(side=tk.LEFT, padx=10)

        self.clear_btn = tk.Button(
            button_frame,
            text="🗑️ Clear",
            command=self.clear_results,
            font=("Arial", 12, "bold"),
            bg="#e74c3c",
            fg="white",
            padx=20,
            pady=10,
            cursor="hand2",
        )
        self.clear_btn.pack(side=tk.LEFT, padx=10)

        # Status bar
        self.status_label = tk.Label(
            self.root,
            text="🔄 Initializing...",
            font=("Arial", 10),
            bg="#34495e",
            fg="white",
            anchor=tk.W,
            padx=10,
        )
        self.status_label.pack(fill=tk.X, side=tk.BOTTOM)

    # ---------- MODEL SWITCHING ----------

    def switch_model(self):
        """Load the selected backend model (PyTorch 3-class or Keras 2-class)."""
        backend = self.model_var.get()
        self.current_backend = backend
        self.model_loaded = False
        self.classify_btn.config(state=tk.DISABLED)
        self.result_text.delete(1.0, tk.END)

        try:
            if backend == "pytorch":
                if self.pytorch_model is None:
                    self.status_label.config(text="🔄 Loading 3-Class PyTorch model...")
                    self.root.update()
                    self.pytorch_model, self.pytorch_device = load_pytorch_model(
                        PYTORCH_MODEL_PATH
                    )
                self.current_class_names = PYTORCH_CLASS_NAMES
                self.model_loaded = True
                self.status_label.config(
                    text=f"✅ 3-Class PyTorch model loaded from {PYTORCH_MODEL_PATH}"
                )

            elif backend == "keras":
                if self.keras_model is None:
                    self.status_label.config(text="🔄 Loading 2-Class Keras model...")
                    self.root.update()
                    self.keras_model = load_keras_model_local(KERAS_MODEL_PATH)
                self.current_class_names = KERAS_CLASS_NAMES
                self.model_loaded = True
                self.status_label.config(
                    text=f"✅ 2-Class Keras model loaded from {KERAS_MODEL_PATH}"
                )

            if self.current_image is not None:
                self.classify_btn.config(state=tk.NORMAL)

        except Exception as e:
            self.model_loaded = False
            self.error_message = str(e)
            self.status_label.config(text=f"❌ Error loading model: {self.error_message}")
            self.result_text.insert(
                tk.END,
                f"❌ Model load error for backend '{backend}':\n\n{self.error_message}",
            )

    # ---------- IMAGE HANDLING ----------

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Concrete Image",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
                ("All Files", "*.*"),
            ],
        )
        if not file_path:
            return

        try:
            self.current_image = Image.open(file_path).convert("RGB")

            display_img = self.current_image.copy()
            display_img.thumbnail((400, 400), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(display_img)

            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo

            if self.model_loaded:
                self.classify_btn.config(state=tk.NORMAL)

            self.status_label.config(
                text=f"📁 Image loaded: {os.path.basename(file_path)}"
            )
        except Exception as e:
            self.status_label.config(text=f"❌ Error loading image: {str(e)}")

    # ---------- CLASSIFICATION ----------

    def classify_image(self):
        if not self.model_loaded:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(
                tk.END,
                "❌ Model not loaded!\n\nPlease select a model and ensure the path is correct.",
            )
            return

        if self.current_image is None:
            self.status_label.config(text="⚠️ Please upload an image first")
            return

        backend = self.current_backend

        try:
            self.status_label.config(text=f"🔄 Classifying with {backend} model...")
            self.root.update()

            if backend == "pytorch":
                self._classify_with_pytorch()
            elif backend == "keras":
                self._classify_with_keras()

        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(
                tk.END, f"❌ Error during classification:\n\n{str(e)}"
            )
            self.status_label.config(text="❌ Classification failed")

    def _classify_with_pytorch(self):
        """Use the 3-class PyTorch ResNet50 model. [attached_file:1]"""
        device = self.pytorch_device
        model = self.pytorch_model

        img_tensor = pytorch_transform(self.current_image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            confidence_score = float(probs[pred_idx] * 100.0)

        predicted_class = self.current_class_names[pred_idx]
        self._display_results(predicted_class, confidence_score, probs)

        self.status_label.config(
            text=f"✅ PyTorch classification complete: {predicted_class.upper()}"
        )

    def _classify_with_keras(self):
        """Use the 2-class Keras MobileNetV2-based model. [attached_file:2]"""
        model = self.keras_model

        img = self.current_image.resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img, dtype=np.float32)
        arr = mnv2_preprocess(arr)
        arr = np.expand_dims(arr, axis=0)

        preds = model.predict(arr, verbose=0)

        # Support both sigmoid (1 unit) and softmax (2 units)
        if preds.shape[-1] == 1:
            p_spalling = float(preds[0, 0])
            p_void = 1.0 - p_spalling
            probs = np.array([p_spalling, p_void])
        else:
            probs = preds[0]
            probs = probs / (np.sum(probs) + 1e-8)

        pred_idx = int(np.argmax(probs))
        confidence_score = float(probs[pred_idx] * 100.0)
        predicted_class = self.current_class_names[pred_idx]

        self._display_results(predicted_class, confidence_score, probs)

        self.status_label.config(
            text=f"✅ Keras classification complete: {predicted_class.upper()}"
        )

    # ---------- RESULT RENDERING ----------

    def _display_results(self, predicted_class, confidence_score, probs_array):
        self.result_text.delete(1.0, tk.END)

        emoji_map = {
            "honeycombing": "🏗️",
            "crack": "⚡",
            "spalling": "💥",
            "void": "⭕",
        }

        result = f"""
╔═══════════════════════════════════╗
║     CLASSIFICATION RESULT         ║
╚═══════════════════════════════════╝

🎯 PREDICTED DEFECT:
   {emoji_map.get(predicted_class, '🔧')} {predicted_class.upper()}

📊 CONFIDENCE: {confidence_score:.2f}%

{'='*38}

📈 ALL CLASS PROBABILITIES:

"""

        for i, cls in enumerate(self.current_class_names):
            prob = float(probs_array[i] * 100.0)
            bar = "█" * int(prob / 5)
            result += f"   {cls:15s} {prob:5.2f}% {bar}\n"

        result += f"\n{'='*38}\n\n"

        if confidence_score > 90:
            result += "✅ HIGH CONFIDENCE\n   Reliable prediction"
        elif confidence_score > 70:
            result += "⚠️  MODERATE CONFIDENCE\n   Consider verification"
        else:
            result += "❌ LOW CONFIDENCE\n   Manual inspection needed"

        self.result_text.insert(tk.END, result)

    # ---------- CLEAR ----------

    def clear_results(self):
        self.current_image = None
        self.image_label.configure(image="", text="No image loaded")
        self.result_text.delete(1.0, tk.END)
        self.classify_btn.config(state=tk.DISABLED)
        self.status_label.config(text="🔄 Ready for new image")

# ==================== RUN APPLICATION ====================

if __name__ == "__main__":
    root = tk.Tk()
    app = ConcreteDefectClassifierApp(root)
    root.mainloop()




# # concrete_classifier_tkinter.py

# import tkinter as tk
# from tkinter import filedialog, ttk
# from PIL import Image, ImageTk
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# import os

# # ==================== CONFIGURATION ====================

# CLASS_NAMES = ['honeycombing', 'crack', 'spalling']
# NUM_CLASSES = 3
# IMG_SIZE = 224

# # Your local model path (relative)
# MODEL_PATH = "./models/best_model_3class_hcs.pth"  # Keep as-is if this file exists

# # Transforms (must match validation/test transforms used in training)
# test_transforms = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# # ==================== MODEL LOADING ====================

# def load_model(model_path: str):
#     """
#     Load the trained ResNet50 model robustly.

#     Handles:
#     - checkpoint dicts with keys like 'model_state_dict', 'state_dict', etc.
#     - plain state_dicts
#     - CPU/GPU differences (map_location)
#     """
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model file not found at: {model_path}")

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # Define architecture EXACTLY as in training (ResNet50 + Dropout + Linear(3))
#     model = models.resnet50(pretrained=False)
#     num_ftrs = model.fc.in_features
#     model.fc = nn.Sequential(
#         nn.Dropout(0.5),
#         nn.Linear(num_ftrs, NUM_CLASSES)
#     )

#     # Load checkpoint
#     checkpoint = torch.load(model_path, map_location=device)

#     # Figure out the correct state_dict
#     if isinstance(checkpoint, dict):
#         # Typical training script: torch.save({'epoch':..., 'model_state_dict':..., ...})
#         if "model_state_dict" in checkpoint:
#             state_dict = checkpoint["model_state_dict"]
#         elif "state_dict" in checkpoint:
#             state_dict = checkpoint["state_dict"]
#         else:
#             # Assume the whole dict is already a state_dict
#             state_dict = checkpoint
#     else:
#         # Direct state_dict object
#         state_dict = checkpoint

#     # Load weights
#     model.load_state_dict(state_dict)
#     model = model.to(device)
#     model.eval()

#     return model, device

# # ==================== GUI APPLICATION ====================

# class ConcreteDefectClassifierApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("🏗️ Concrete Defect Classifier")
#         self.root.geometry("900x700")
#         self.root.configure(bg="#f0f0f0")

#         # Load model
#         try:
#             self.model, self.device = load_model(MODEL_PATH)
#             self.model_loaded = True
#             self.error_message = ""
#         except Exception as e:
#             self.model_loaded = False
#             self.model = None
#             self.device = torch.device("cpu")
#             self.error_message = str(e)

#         self.current_image = None
#         self.create_widgets()

#     def create_widgets(self):
#         """Create GUI components"""

#         # Title Frame
#         title_frame = tk.Frame(self.root, bg="#2c3e50", height=80)
#         title_frame.pack(fill=tk.X, pady=(0, 20))

#         title_label = tk.Label(
#             title_frame,
#             text="🏗️ Concrete Defect Classification System",
#             font=("Arial", 20, "bold"),
#             bg="#2c3e50",
#             fg="white"
#         )
#         title_label.pack(pady=20)

#         # Main container
#         main_frame = tk.Frame(self.root, bg="#f0f0f0")
#         main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

#         # Left panel - Image display
#         left_frame = tk.Frame(main_frame, bg="white", relief=tk.RAISED, borderwidth=2)
#         left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

#         tk.Label(
#             left_frame,
#             text="📷 Image Preview",
#             font=("Arial", 14, "bold"),
#             bg="white"
#         ).pack(pady=10)

#         self.image_label = tk.Label(left_frame, bg="white", text="No image loaded")
#         self.image_label.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

#         # Right panel - Results
#         right_frame = tk.Frame(main_frame, bg="white", relief=tk.RAISED, borderwidth=2)
#         right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

#         tk.Label(
#             right_frame,
#             text="📊 Classification Results",
#             font=("Arial", 14, "bold"),
#             bg="white"
#         ).pack(pady=10)

#         self.result_text = tk.Text(
#             right_frame,
#             height=20,
#             width=40,
#             font=("Courier", 11),
#             bg="#f9f9f9",
#             relief=tk.FLAT,
#             padx=10,
#             pady=10
#         )
#         self.result_text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

#         # Button frame
#         button_frame = tk.Frame(self.root, bg="#f0f0f0")
#         button_frame.pack(pady=20)

#         # Upload button
#         self.upload_btn = tk.Button(
#             button_frame,
#             text="📁 Upload Image",
#             command=self.upload_image,
#             font=("Arial", 12, "bold"),
#             bg="#3498db",
#             fg="white",
#             padx=20,
#             pady=10,
#             cursor="hand2"
#         )
#         self.upload_btn.pack(side=tk.LEFT, padx=10)

#         # Classify button
#         self.classify_btn = tk.Button(
#             button_frame,
#             text="🔍 Classify Defect",
#             command=self.classify_image,
#             font=("Arial", 12, "bold"),
#             bg="#27ae60",
#             fg="white",
#             padx=20,
#             pady=10,
#             cursor="hand2",
#             state=tk.NORMAL if self.model_loaded else tk.DISABLED
#         )
#         self.classify_btn.pack(side=tk.LEFT, padx=10)

#         # Clear button
#         self.clear_btn = tk.Button(
#             button_frame,
#             text="🗑️ Clear",
#             command=self.clear_results,
#             font=("Arial", 12, "bold"),
#             bg="#e74c3c",
#             fg="white",
#             padx=20,
#             pady=10,
#             cursor="hand2"
#         )
#         self.clear_btn.pack(side=tk.LEFT, padx=10)

#         # Status bar
#         status_text = (
#             f"✅ Model loaded: {MODEL_PATH}"
#             if self.model_loaded
#             else f"❌ Error loading model: {self.error_message}"
#         )

#         self.status_label = tk.Label(
#             self.root,
#             text=status_text,
#             font=("Arial", 10),
#             bg="#34495e",
#             fg="white",
#             anchor=tk.W,
#             padx=10
#         )
#         self.status_label.pack(fill=tk.X, side=tk.BOTTOM)

#     def upload_image(self):
#         """Handle image upload"""
#         file_path = filedialog.askopenfilename(
#             title="Select Concrete Image",
#             filetypes=[
#                 ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
#                 ("All Files", "*.*")
#             ]
#         )

#         if file_path:
#             try:
#                 self.current_image = Image.open(file_path).convert("RGB")

#                 display_img = self.current_image.copy()
#                 display_img.thumbnail((400, 400), Image.Resampling.LANCZOS)
#                 photo = ImageTk.PhotoImage(display_img)

#                 self.image_label.configure(image=photo, text="")
#                 self.image_label.image = photo

#                 if self.model_loaded:
#                     self.classify_btn.config(state=tk.NORMAL)

#                 self.status_label.config(
#                     text=f"📁 Image loaded: {os.path.basename(file_path)}"
#                 )

#             except Exception as e:
#                 self.status_label.config(text=f"❌ Error loading image: {str(e)}")

#     def classify_image(self):
#         """Classify the uploaded image"""
#         if not self.model_loaded:
#             self.result_text.delete(1.0, tk.END)
#             self.result_text.insert(
#                 tk.END,
#                 "❌ Model not loaded!\n\nPlease check the model file and restart the app."
#             )
#             return

#         if self.current_image is None:
#             self.status_label.config(text="⚠️ Please upload an image first")
#             return

#         try:
#             self.status_label.config(text="🔄 Classifying...")
#             self.root.update()

#             # Preprocess
#             img_tensor = test_transforms(self.current_image).unsqueeze(0).to(self.device)

#             # Predict
#             with torch.no_grad():
#                 outputs = self.model(img_tensor)
#                 probabilities = torch.nn.functional.softmax(outputs, dim=1)
#                 confidence, predicted = torch.max(probabilities, 1)

#             predicted_class = CLASS_NAMES[predicted.item()]
#             confidence_score = confidence.item() * 100.0

#             # Display results
#             self.result_text.delete(1.0, tk.END)

#             emoji_map = {
#                 "honeycombing": "🏗️",
#                 "crack": "⚡",
#                 "spalling": "💥"
#             }

#             result = f"""
# ╔═══════════════════════════════════╗
# ║     CLASSIFICATION RESULT         ║
# ╚═══════════════════════════════════╝

# 🎯 PREDICTED DEFECT:
#    {emoji_map.get(predicted_class, '🔧')} {predicted_class.upper()}

# 📊 CONFIDENCE: {confidence_score:.2f}%

# {'='*38}

# 📈 ALL CLASS PROBABILITIES:

# """
#             for i, cls in enumerate(CLASS_NAMES):
#                 prob = probabilities[0][i].item() * 100.0
#                 bar = "█" * int(prob / 5)
#                 result += f"   {cls:15s} {prob:5.2f}% {bar}\n"

#             result += f"\n{'='*38}\n\n"

#             if confidence_score > 90:
#                 result += "✅ HIGH CONFIDENCE\n   Reliable prediction"
#             elif confidence_score > 70:
#                 result += "⚠️  MODERATE CONFIDENCE\n   Consider verification"
#             else:
#                 result += "❌ LOW CONFIDENCE\n   Manual inspection needed"

#             self.result_text.insert(tk.END, result)
#             self.status_label.config(
#                 text=f"✅ Classification complete: {predicted_class.upper()}"
#             )

#         except Exception as e:
#             self.result_text.delete(1.0, tk.END)
#             self.result_text.insert(
#                 tk.END,
#                 f"❌ Error during classification:\n\n{str(e)}"
#             )
#             self.status_label.config(text="❌ Classification failed")

#     def clear_results(self):
#         """Clear all results"""
#         self.current_image = None
#         self.image_label.configure(image="", text="No image loaded")
#         self.result_text.delete(1.0, tk.END)
#         self.classify_btn.config(state=tk.DISABLED)
#         self.status_label.config(text="🔄 Ready for new image")

# # ==================== RUN APPLICATION ====================

# if __name__ == "__main__":
#     root = tk.Tk()
#     app = ConcreteDefectClassifierApp(root)
#     root.mainloop()

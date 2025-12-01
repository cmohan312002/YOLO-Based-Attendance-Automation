import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
from deepface import DeepFace
import numpy as np
import os
import pandas as pd
from datetime import datetime

# --------------------------- Configuration ---------------------------
KNOWN_FACES_DIR = "known_faces"
YOLO_MODEL_PATH = "yolov8s-face-lindevs.pt"  # use the one you actually have
ATTENDANCE_FILE = "attendance.csv"
EMBEDDING_MODEL = "Facenet"
DISTANCE_THRESHOLD = 1.0   # tune this based on your testing

# Globals for embeddings
known_embeddings = []
known_names = []

# --------------------------- Helper Functions ---------------------------

def init_attendance_file():
    """Create CSV file if it doesn't exist or is empty."""
    if not os.path.exists(ATTENDANCE_FILE) or os.stat(ATTENDANCE_FILE).st_size == 0:
        pd.DataFrame(columns=["Name", "Time"]).to_csv(ATTENDANCE_FILE, index=False)


def load_known_faces():
    """Load embeddings for all known faces from KNOWN_FACES_DIR."""
    global known_embeddings, known_names

    if not os.path.isdir(KNOWN_FACES_DIR):
        print(f"[!] Folder '{KNOWN_FACES_DIR}' not found. No known faces loaded.")
        return

    files = os.listdir(KNOWN_FACES_DIR)
    if not files:
        print(f"[!] Folder '{KNOWN_FACES_DIR}' is empty. No known faces loaded.")
        return

    for img_file in files:
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(KNOWN_FACES_DIR, img_file)
        try:
            reps = DeepFace.represent(
                img_path=path,
                model_name=EMBEDDING_MODEL,
                enforce_detection=False
            )
            if not reps:
                print(f"[!] No embedding returned for {path}")
                continue

            embedding = np.array(reps[0]["embedding"], dtype="float32")
            known_embeddings.append(embedding)
            known_names.append(os.path.splitext(img_file)[0])
            print(f"[+] Loaded embedding for {img_file}")

        except Exception as e:
            print(f"[!] Error processing {path}: {e}")


def recognize_face(face_img):
    """Return best matching name for the given face image or None."""
    if not known_embeddings:
        # No known faces to compare
        return None

    try:
        reps = DeepFace.represent(
            img_path=face_img,  # can be numpy array
            model_name=EMBEDDING_MODEL,
            enforce_detection=False
        )
        if not reps:
            return None

        face_embedding = np.array(reps[0]["embedding"], dtype="float32")
    except Exception as e:
        print(f"[!] DeepFace error on captured face: {e}")
        return None

    best_match = None
    best_dist = float("inf")

    for idx, known_emb in enumerate(known_embeddings):
        known_emb_array = np.array(known_emb, dtype="float32")
        dist = np.linalg.norm(face_embedding - known_emb_array)

        if dist < best_dist:
            best_dist = dist
            best_match = known_names[idx]

    # Debug print to tune threshold
    print(f"Best match: {best_match}, distance: {best_dist:.4f}")

    if best_dist < DISTANCE_THRESHOLD:
        return best_match
    return None


def mark_attendance(name):
    """Add name to attendance CSV if not already present."""
    if not name:
        return

    df = pd.read_csv(ATTENDANCE_FILE)

    if name not in df["Name"].values:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df.loc[len(df)] = [name, now]
        df.to_csv(ATTENDANCE_FILE, index=False)
        update_attendance_list()
        print(f"[✔] Attendance marked for {name}")
    else:
        print(f"[i] {name} already marked present.")


def update_attendance_list():
    """Refresh the Treeview widget with latest attendance."""
    if not os.path.exists(ATTENDANCE_FILE):
        return

    df = pd.read_csv(ATTENDANCE_FILE)
    attendance_list.delete(*attendance_list.get_children())

    for _, row in df.iterrows():
        attendance_list.insert("", "end", values=(row["Name"], row["Time"]))


def upload_photo():
    """Handle class photo upload, face detection and recognition."""
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )
    if not file_path:
        return

    if not known_embeddings:
        messagebox.showwarning(
            "No Known Faces",
            "No known faces loaded! Please add images to the 'known_faces' folder first."
        )

    # Read image
    img = cv2.imread(file_path)
    if img is None:
        messagebox.showerror("Error", "Failed to read image file.")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        results = model(img_rgb)
    except Exception as e:
        messagebox.showerror("YOLO Error", f"Error while running detection: {e}")
        return

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r, "boxes") else []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            # Clamp coordinates to image bounds
            h, w = img_rgb.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face_img = img_rgb[y1:y2, x1:x2]
            if face_img.size == 0:
                continue

            name = recognize_face(face_img)

            if name:
                mark_attendance(name)
                color = (0, 255, 0)  # Green for known
                text = name
            else:
                color = (255, 0, 0)  # Red for unknown
                text = "Unknown"

            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img_rgb,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

    # Display image in GUI
    img_pil = Image.fromarray(img_rgb)
    img_pil = img_pil.resize((600, 400))
    imgtk = ImageTk.PhotoImage(img_pil)
    photo_label.imgtk = imgtk
    photo_label.configure(image=imgtk)


# --------------------------- Init Models & Data ---------------------------

print("[*] Initializing YOLO model...")
model = YOLO(YOLO_MODEL_PATH)

print("[*] Initializing attendance file...")
init_attendance_file()

print("[*] Loading known faces...")
load_known_faces()

# --------------------------- GUI ---------------------------

root = tk.Tk()
root.title("Class Photo Attendance System")
root.geometry("800x700")

upload_btn = tk.Button(
    root,
    text="Upload Class Photo",
    command=upload_photo,
    font=("Arial", 14)
)
upload_btn.pack(pady=10)

photo_label = tk.Label(root)
photo_label.pack(pady=10)

attendance_frame = tk.Frame(root)
attendance_frame.pack(fill="both", expand=True, padx=10, pady=10)

columns = ("Name", "Time")
attendance_list = ttk.Treeview(attendance_frame, columns=columns, show="headings")

for col in columns:
    attendance_list.heading(col, text=col)
    attendance_list.column(col, width=200, anchor="center")

# Add a scrollbar for the Treeview
scrollbar = ttk.Scrollbar(attendance_frame, orient="vertical", command=attendance_list.yview)
attendance_list.configure(yscrollcommand=scrollbar.set)
attendance_list.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

update_attendance_list()

root.mainloop()

import tkinter as tk
from tkinter import filedialog, ttk
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
YOLO_MODEL_PATH = "yolov8n-face.pt"
ATTENDANCE_FILE = "attendance.csv"

# Load YOLO face model
model = YOLO("yolov8s-face-lindevs.pt")  # no "models/" folder needed if in root


# Load reference embeddings
known_embeddings = []
known_names = []

for img_file in os.listdir(KNOWN_FACES_DIR):
    path = os.path.join(KNOWN_FACES_DIR, img_file)
    embedding = DeepFace.represent(path, model_name="Facenet")[0]["embedding"]
    known_embeddings.append(np.array(embedding))  # convert to NumPy array here
    known_names.append(os.path.splitext(img_file)[0])


# Create CSV if missing or empty
if not os.path.exists(ATTENDANCE_FILE) or os.stat(ATTENDANCE_FILE).st_size == 0:
    pd.DataFrame(columns=["Name", "Time"]).to_csv(ATTENDANCE_FILE, index=False)

# --------------------------- Functions ---------------------------
def recognize_face(face_img):
    # Get embedding and convert to NumPy array
    face_embedding = np.array(
    DeepFace.represent(face_img, model_name="Facenet", enforce_detection=False)[0]["embedding"]
)

    best_match = None
    best_dist = 999
    for idx, known_emb in enumerate(known_embeddings):
        known_emb_array = np.array(known_emb)  # convert known embedding to NumPy
        dist = np.linalg.norm(face_embedding - known_emb_array)
        if dist < best_dist:
            best_dist = dist
            best_match = known_names[idx]
    if best_dist < 10:  # threshold
        return best_match
    return None

def mark_attendance(name):
    df = pd.read_csv(ATTENDANCE_FILE)
    if name not in df["Name"].values:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df.loc[len(df)] = [name, now]
        df.to_csv(ATTENDANCE_FILE, index=False)
        update_attendance_list()
        print(f"[✔] Attendance marked for {name}")

def update_attendance_list():
    df = pd.read_csv(ATTENDANCE_FILE)
    attendance_list.delete(*attendance_list.get_children())
    for _, row in df.iterrows():
        attendance_list.insert("", "end", values=(row["Name"], row["Time"]))

def upload_photo():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if not file_path:
        return

    # Read image
    img = cv2.imread(file_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img_rgb)

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r, 'boxes') else []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face_img = img_rgb[y1:y2, x1:x2]
            if face_img.size == 0:
                continue
            name = recognize_face(face_img)
            if name:
                mark_attendance(name)
                color = (0, 255, 0)
                text = name
            else:
                color = (255, 0, 0)
                text = "Unknown"
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_rgb, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Display image in GUI
    img_pil = Image.fromarray(img_rgb)
    img_pil = img_pil.resize((600, 400))
    imgtk = ImageTk.PhotoImage(img_pil)
    photo_label.imgtk = imgtk
    photo_label.configure(image=imgtk)

# --------------------------- GUI ---------------------------
root = tk.Tk()
root.title("Class Photo Attendance System")
root.geometry("800x700")

upload_btn = tk.Button(root, text="Upload Class Photo", command=upload_photo, font=("Arial", 14))
upload_btn.pack(pady=10)

photo_label = tk.Label(root)
photo_label.pack(pady=10)

attendance_frame = tk.Frame(root)
attendance_frame.pack(fill="both", expand=True, padx=10, pady=10)

columns = ("Name", "Time")
attendance_list = ttk.Treeview(attendance_frame, columns=columns, show="headings")
for col in columns:
    attendance_list.heading(col, text=col)
attendance_list.pack(fill="both", expand=True)

update_attendance_list()

root.mainloop()

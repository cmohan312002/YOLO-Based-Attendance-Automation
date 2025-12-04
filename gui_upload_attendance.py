import tkinter as tk
from tkinter import filedialog, ttk, messagebox, simpledialog
from tkinter import scrolledtext
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
from deepface import DeepFace
import numpy as np
import os
import pandas as pd
from datetime import datetime


KNOWN_FACES_DIR = "known_faces"
YOLO_MODEL_PATH = "yolov8s-face-lindevs.pt"  
ATTENDANCE_FILE = "attendance.csv"
EMBEDDING_MODEL = "Facenet"
DISTANCE_THRESHOLD = 15 

model = None
known_embeddings = []
known_names = []

root = None
upload_btn = None
register_btn = None
clear_btn = None
status_label = None
progress_bar = None
log_text = None
attendance_list = None
photo_label = None




def log(msg: str):
    """Log message to the text area and console."""
    print(msg)
    if log_text is not None:
        log_text.insert(tk.END, msg + "\n")
        log_text.see(tk.END)




def init_attendance_file():
    """Create CSV file if it doesn't exist or is empty, with Name/Date/Time columns."""
    expected_cols = ["Name", "Date", "Time"]

    if not os.path.exists(ATTENDANCE_FILE) or os.stat(ATTENDANCE_FILE).st_size == 0:
        df = pd.DataFrame(columns=expected_cols)
        df.to_csv(ATTENDANCE_FILE, index=False)
        log("[*] Created new attendance file.")
    else:
        df = pd.read_csv(ATTENDANCE_FILE)

        
        if list(df.columns) != expected_cols:
            df = pd.DataFrame(columns=expected_cols)
            df.to_csv(ATTENDANCE_FILE, index=False)
            log("[!] Existing attendance file had different columns. Reinitialized with Name/Date/Time.")

def mark_attendance(name: str):
    """Add name to attendance CSV if not already present today."""
    if not name:
        return

    df = pd.read_csv(ATTENDANCE_FILE)
    today = datetime.now().strftime("%Y-%m-%d")

    already_today = ((df["Name"] == name) & (df["Date"] == today)).any()
    if already_today:
        log(f"[i] {name} is already marked present today ({today}).")
        return

    now_time = datetime.now().strftime("%H:%M:%S")
    df.loc[len(df)] = [name, today, now_time]
    df.to_csv(ATTENDANCE_FILE, index=False)
    update_attendance_list()
    log(f"[✔] Attendance marked for {name} on {today} at {now_time}.")


def update_attendance_list():
    """Refresh the Treeview widget with latest attendance."""
    if not os.path.exists(ATTENDANCE_FILE):
        return

    df = pd.read_csv(ATTENDANCE_FILE)
    attendance_list.delete(*attendance_list.get_children())

    for _, row in df.iterrows():
        attendance_list.insert(
            "",
            "end",
            values=(row["Name"], row["Date"], row["Time"])
        )


def clear_all_attendance():
    """Clear all attendance records after confirmation."""
    if not os.path.exists(ATTENDANCE_FILE):
        return

    ans = messagebox.askyesno(
        "Confirm",
        "Are you sure you want to clear ALL attendance records?"
    )
    if not ans:
        return

    pd.DataFrame(columns=["Name", "Date", "Time"]).to_csv(ATTENDANCE_FILE, index=False)
    update_attendance_list()
    log("[!] All attendance records cleared.")


# --------------------------- Known Faces & Recognition ---------------------------

def load_known_faces():
    """Load embeddings for all known faces from KNOWN_FACES_DIR."""
    global known_embeddings, known_names

    known_embeddings.clear()
    known_names.clear()

    if not os.path.isdir(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
        log(f"[!] Folder '{KNOWN_FACES_DIR}' not found. Created empty folder.")
        return

    files = [
        f for f in os.listdir(KNOWN_FACES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    total = len(files)
    if total == 0:
        log(f"[!] Folder '{KNOWN_FACES_DIR}' is empty. No known faces loaded.")
        return

    status_label.config(text="Status: Loading known faces...")
    progress_bar["maximum"] = total
    progress_bar["value"] = 0
    root.update_idletasks()

    for idx, img_file in enumerate(files, start=1):
        path = os.path.join(KNOWN_FACES_DIR, img_file)
        try:
            reps = DeepFace.represent(
                img_path=path,
                model_name=EMBEDDING_MODEL,
                enforce_detection=False
            )
            if not reps:
                log(f"[!] No embedding returned for {img_file}")
                continue

            embedding = np.array(reps[0]["embedding"], dtype="float32")
            known_embeddings.append(embedding)
            name = os.path.splitext(img_file)[0]
            known_names.append(name)
            log(f"[+] Loaded embedding for {name}")

        except Exception as e:
            log(f"[!] Error processing {img_file}: {e}")

        progress_bar["value"] = idx
        root.update_idletasks()

    status_label.config(text="Status: Ready")
    log(f"[*] Loaded {len(known_embeddings)} known faces.")


def add_single_known_face(name: str, img_path: str):
    """Add one new known face from the given image path."""
    global known_embeddings, known_names

    try:
        reps = DeepFace.represent(
            img_path=img_path,
            model_name=EMBEDDING_MODEL,
            enforce_detection=False
        )
        if not reps:
            log(f"[!] No embedding returned for new face: {name}")
            return

        embedding = np.array(reps[0]["embedding"], dtype="float32")
        known_embeddings.append(embedding)
        known_names.append(name)
        log(f"[+] Registered new student: {name}")

    except Exception as e:
        log(f"[!] Error registering {name}: {e}")


def recognize_face(face_img):
    """Return best matching name for the given face image or None."""
    if not known_embeddings:
        return None

    try:
        reps = DeepFace.represent(
            face_img,  # numpy array
            model_name=EMBEDDING_MODEL,
            enforce_detection=False
        )
        if not reps:
            return None

        face_embedding = np.array(reps[0]["embedding"], dtype="float32")
    except Exception as e:
        log(f"[!] DeepFace error on captured face: {e}")
        return None

    best_match = None
    best_dist = float("inf")

    for idx, known_emb in enumerate(known_embeddings):
        known_emb_array = np.array(known_emb, dtype="float32")
        dist = np.linalg.norm(face_embedding - known_emb_array)

        if dist < best_dist:
            best_dist = dist
            best_match = known_names[idx]

    log(f"[debug] Best match: {best_match}, distance: {best_dist:.4f}")

    if best_dist < DISTANCE_THRESHOLD:
        return best_match
    return None


# --------------------------- GUI Actions ---------------------------

def upload_photo():
    """Handle class photo upload, face detection and recognition."""
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )
    if not file_path:
        return

    if model is None:
        messagebox.showerror("Error", "YOLO model is not initialized yet.")
        return

    img = cv2.imread(file_path)
    if img is None:
        messagebox.showerror("Error", "Failed to read image file.")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        results = model(img_rgb)
    except Exception as e:
        messagebox.showerror("YOLO Error", f"Error while running detection: {e}")
        log(f"[!] YOLO error: {e}")
        return

    h, w = img_rgb.shape[:2]

    if not known_embeddings:
        messagebox.showwarning(
            "No Known Faces",
            "No known faces loaded. Unknown faces will be shown but not marked."
        )

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r, "boxes") else []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            # Clamp coordinates to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

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
            cv2.putText(
                img_rgb,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

    img_pil = Image.fromarray(img_rgb)
    img_pil = img_pil.resize((600, 400))
    imgtk = ImageTk.PhotoImage(img_pil)
    photo_label.imgtk = imgtk
    photo_label.configure(image=imgtk)


def register_new_student():
    """Register a new student's face via GUI."""
    name = simpledialog.askstring("Student Name", "Enter student name:")
    if not name:
        return

    file_path = filedialog.askopenfilename(
        title="Select student face image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        return

    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    ext = os.path.splitext(file_path)[1].lower()
    save_path = os.path.join(KNOWN_FACES_DIR, name + ext)

    try:
        img = Image.open(file_path)
        img.save(save_path)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save image: {e}")
        log(f"[!] Error saving new student image: {e}")
        return

    add_single_known_face(name, save_path)
    messagebox.showinfo("Success", f"Registered new student: {name}")



def init_system():
    global model

    # Disable buttons during init
    upload_btn.config(state=tk.DISABLED)
    register_btn.config(state=tk.DISABLED)
    clear_btn.config(state=tk.DISABLED)

    status_label.config(text="Status: Initializing YOLO model...")
    root.update_idletasks()
    log("[*] Initializing YOLO model...")

    try:
        model = YOLO(YOLO_MODEL_PATH)
    except Exception as e:
        messagebox.showerror("YOLO Error", f"Failed to load YOLO model: {e}")
        log(f"[!] Failed to load YOLO model: {e}")
        status_label.config(text="Status: Error loading YOLO")
        return

    log("[*] YOLO model loaded.")

    log("[*] Initializing attendance file...")
    init_attendance_file()
    update_attendance_list()

    log("[*] Loading known faces...")
    load_known_faces()

    upload_btn.config(state=tk.NORMAL)
    register_btn.config(state=tk.NORMAL)
    clear_btn.config(state=tk.NORMAL)

    status_label.config(text="Status: Ready")
    log("[*] System initialized. You can now upload class photos.")



def build_gui():
    global root, upload_btn, register_btn, clear_btn
    global status_label, progress_bar, log_text, attendance_list, photo_label

    root = tk.Tk()
    root.title("Class Photo Attendance System")
    root.geometry("900x800")

    top_frame = tk.Frame(root)
    top_frame.pack(pady=10)

    upload_btn = tk.Button(
        top_frame,
        text="Upload Class Photo",
        command=upload_photo,
        font=("Arial", 12),
        state=tk.DISABLED  
    )
    upload_btn.grid(row=0, column=0, padx=5)

    register_btn = tk.Button(
        top_frame,
        text="Register New Student",
        command=register_new_student,
        font=("Arial", 12),
        state=tk.DISABLED
    )
    register_btn.grid(row=0, column=1, padx=5)

    clear_btn = tk.Button(
        top_frame,
        text="Clear All Attendance",
        command=clear_all_attendance,
        font=("Arial", 12),
        state=tk.DISABLED
    )
    clear_btn.grid(row=0, column=2, padx=5)


    photo_label = tk.Label(root)
    photo_label.pack(pady=10)


    attendance_frame = tk.LabelFrame(root, text="Attendance")
    attendance_frame.pack(fill="both", expand=True, padx=10, pady=10)

    columns = ("Name", "Date", "Time")
    attendance_list = ttk.Treeview(attendance_frame, columns=columns, show="headings")

    for col in columns:
        attendance_list.heading(col, text=col)
        attendance_list.column(col, width=200, anchor="center")

    scrollbar = ttk.Scrollbar(attendance_frame, orient="vertical", command=attendance_list.yview)
    attendance_list.configure(yscrollcommand=scrollbar.set)
    attendance_list.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")


    status_frame = tk.Frame(root)
    status_frame.pack(fill="x", padx=10, pady=(0, 5))

    status_label = tk.Label(status_frame, text="Status: Initializing...", anchor="w")
    status_label.pack(side="left", fill="x", expand=True)

    progress_bar = ttk.Progressbar(status_frame, orient="horizontal", mode="determinate")
    progress_bar.pack(side="right", fill="x", expand=False, padx=(10, 0))


    log_frame = tk.LabelFrame(root, text="Log")
    log_frame.pack(fill="both", expand=True, padx=10, pady=5)

    global log_text
    log_text = scrolledtext.ScrolledText(log_frame, height=8)
    log_text.pack(fill="both", expand=True)

    root.after(100, init_system)

    return root



if __name__ == "__main__":
    app = build_gui()
    app.mainloop()

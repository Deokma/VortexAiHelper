"""
██╗   ██╗ ██████╗ ██████╗ ████████╗███████╗██╗  ██╗     █████╗ ██╗
██║   ██║██╔═══██╗██╔══██╗╚══██╔══╝██╔════╝╚██╗██╔╝    ██╔══██╗██║
██║   ██║██║   ██║██████╔╝   ██║   █████╗   ╚███╔╝     ███████║██║
╚██╗ ██╔╝██║   ██║██╔══██╗   ██║   ██╔══╝   ██╔██╗     ██╔══██║██║
 ╚████╔╝ ╚██████╔╝██║  ██║   ██║   ███████╗██╔╝ ██╗    ██║  ██║██║
  ╚═══╝   ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝    ╚═╝  ╚═╝╚═╝

VortexAI Helper — smart assistant for downloading mods
Learns from your actions and automates repetitive tasks.

Install dependencies:
    pip install pyautogui pynput opencv-python pillow scikit-learn numpy

Run:
    python vortex_ai_helper.py
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import json
import os
import time
import sys

# ─── Dependency Check ────────────────────────────────────────────────────────

MISSING = []
try:
    import pyautogui
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.05
except ImportError:
    MISSING.append("pyautogui")

try:
    from pynput import mouse, keyboard
except ImportError:
    MISSING.append("pynput")

try:
    import cv2
    import numpy as np
except ImportError:
    MISSING.append("opencv-python")

try:
    from PIL import Image, ImageTk, ImageGrab
except ImportError:
    MISSING.append("pillow")

try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    import pickle
except ImportError:
    MISSING.append("scikit-learn")

# ─── Constants ───────────────────────────────────────────────────────────────

DATA_FILE    = "vortex_ai_brain.json"
MODEL_FILE   = "vortex_ai_model.pkl"
SHOTS_DIR    = "vortex_screenshots"
TEMPLATE_DIR = "vortex_templates"

STEP_NAMES = {
    0: "Click «Download Manually» in Vortex",
    1: "Click browser button (step 1)",
    2: "Click browser button (step 2)",
}

# ─── Utilities ───────────────────────────────────────────────────────────────

def ensure_dirs():
    for d in [SHOTS_DIR, TEMPLATE_DIR]:
        os.makedirs(d, exist_ok=True)

def screenshot(name=None):
    img = ImageGrab.grab()
    if name:
        path = os.path.join(SHOTS_DIR, name)
        img.save(path)
    return img

def img_to_cv(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def match_template(screen_cv, template_cv, threshold=0.75):
    """Returns (x, y) center of matched template, or None."""
    result = cv2.matchTemplate(screen_cv, template_cv, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val >= threshold:
        h, w = template_cv.shape[:2]
        cx = max_loc[0] + w // 2
        cy = max_loc[1] + h // 2
        return (cx, cy), max_val
    return None, max_val

def load_brain():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"steps": {}, "sessions": 0}

def save_brain(brain):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(brain, f, ensure_ascii=False, indent=2)

def load_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f)
    return None, None

def save_model(clf, scaler):
    with open(MODEL_FILE, "wb") as f:
        pickle.dump((clf, scaler), f)

def crop_around(img, x, y, pad=40):
    """Crops the area around a click point for use as a template."""
    left  = max(0, x - pad)
    top   = max(0, y - pad)
    right = min(img.width,  x + pad)
    bot   = min(img.height, y + pad)
    return img.crop((left, top, right, bot))

# ─── Brain: Neural Network ───────────────────────────────────────────────────

class VortexBrain:
    """
    Simple neural network (MLPClassifier from sklearn):
    Input:  pixel vector of a screenshot around the cursor
    Output: step class (0, 1, 2, ...)
    """

    def __init__(self):
        self.clf     = None
        self.scaler  = None
        self.trained = False
        self._load()

    def _load(self):
        clf, scaler = load_model()
        if clf is not None:
            self.clf     = clf
            self.scaler  = scaler
            self.trained = True

    def _img_features(self, pil_crop):
        """Converts an image into a feature vector."""
        small = pil_crop.resize((20, 20)).convert("L")
        return list(small.getdata())

    def train(self, samples):
        """
        samples: list of (pil_crop_image, step_label)
        """
        X, y = [], []
        for img, label in samples:
            X.append(self._img_features(img))
            y.append(label)

        if len(set(y)) < 2:
            # Not enough classes — fall back to template matching
            self.trained = False
            return False

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.clf = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=42
        )
        self.clf.fit(X_scaled, y)
        self.trained = True
        save_model(self.clf, self.scaler)
        return True

    def predict(self, pil_crop):
        if not self.trained:
            return None
        feat = self._img_features(pil_crop)
        feat_scaled = self.scaler.transform([feat])
        return int(self.clf.predict(feat_scaled)[0])


# ─── Stop-Key Listener ───────────────────────────────────────────────────────

STOP_KEY = keyboard.Key.f8   # Press F8 to stop the loop

class KeyboardStopListener:
    """
    Listens for the stop key (F8) in the background via pynput.
    When pressed, calls on_stop().
    """

    def __init__(self, on_stop):
        self.on_stop   = on_stop
        self._active   = False
        self._listener = None

    def start(self):
        self._active = True
        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.start()

    def stop(self):
        self._active = False
        if self._listener:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None

    def _on_press(self, key):
        if self._active and key == STOP_KEY:
            self._active = False
            self.on_stop()

    # Compatibility stub — set_expected is no longer needed
    def set_expected(self, x, y):
        pass


# ─── Action Recorder ─────────────────────────────────────────────────────────

class ActionRecorder:
    def __init__(self, on_click_cb):
        self.on_click_cb = on_click_cb
        self._listener   = None
        self.recording   = False

    def start(self):
        self.recording = True
        self._listener = mouse.Listener(on_click=self._on_click)
        self._listener.start()

    def stop(self):
        self.recording = False
        if self._listener:
            self._listener.stop()
            self._listener = None

    def _on_click(self, x, y, button, pressed):
        if not self.recording:
            return False
        if pressed and button == mouse.Button.left:
            self.on_click_cb(x, y)


# ─── GUI ─────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🎮 VortexAI Helper")
        self.resizable(False, False)
        self.configure(bg="#1e1e2e")

        ensure_dirs()
        self.brain   = VortexBrain()
        self.data    = load_brain()
        self.net_clf = None

        # Training state
        self.teach_mode    = False
        self.teach_step    = 0
        self.teach_samples = []  # (img, label)
        self.templates     = {}  # step -> cv2 image
        self.recorder      = ActionRecorder(self._on_recorded_click)

        # Playback state
        self.play_thread = None
        self.playing     = False
        self.loop_mode   = False   # True = run continuously
        self.loop_count  = 0       # completed loop count
        self.takeover    = KeyboardStopListener(self._on_stop_key)

        self._load_templates()
        self._build_ui()
        self._refresh_status()

    # ── Templates ────────────────────────────────────────────────────────────

    def _load_templates(self):
        for step_id in STEP_NAMES:
            path = os.path.join(TEMPLATE_DIR, f"step_{step_id}.png")
            if os.path.exists(path):
                self.templates[step_id] = cv2.imread(path)

    def _save_template(self, step_id, pil_img):
        path = os.path.join(TEMPLATE_DIR, f"step_{step_id}.png")
        cv2.imwrite(path, img_to_cv(pil_img))
        self.templates[step_id] = img_to_cv(pil_img)

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self):
        PAD = dict(padx=12, pady=6)

        # Header
        tk.Label(
            self, text="🎮  VortexAI Helper",
            font=("Segoe UI", 16, "bold"),
            bg="#1e1e2e", fg="#cba6f7"
        ).pack(pady=(16, 2))

        tk.Label(
            self, text="Smart assistant for downloading mods via Vortex",
            font=("Segoe UI", 9),
            bg="#1e1e2e", fg="#6c7086"
        ).pack(pady=(0, 12))

        # Status
        self.status_frame = tk.Frame(self, bg="#313244", bd=0)
        self.status_frame.pack(fill="x", padx=16, pady=4)

        self.lbl_status = tk.Label(
            self.status_frame,
            text="", font=("Segoe UI", 10),
            bg="#313244", fg="#a6e3a1",
            justify="left", wraplength=360
        )
        self.lbl_status.pack(**PAD)

        # ── Training Mode ──
        sep = tk.Frame(self, height=1, bg="#45475a")
        sep.pack(fill="x", padx=16, pady=8)

        tk.Label(
            self, text="🧠  Train Neural Network",
            font=("Segoe UI", 12, "bold"),
            bg="#1e1e2e", fg="#89b4fa"
        ).pack()

        tk.Label(
            self, text=(
                "Click «Start Training», then download a mod as you normally would.\n"
                "The AI will remember which buttons you click and where they are."
            ),
            font=("Segoe UI", 9),
            bg="#1e1e2e", fg="#6c7086",
            wraplength=360, justify="center"
        ).pack(pady=4)

        self.teach_frame = tk.Frame(self, bg="#1e1e2e")
        self.teach_frame.pack(pady=6)

        self.btn_teach = self._btn(
            self.teach_frame, "▶  Start Training", "#a6e3a1", "#1e1e2e",
            self._toggle_teach
        )
        self.btn_teach.pack(side="left", padx=4)

        self.btn_retrain = self._btn(
            self.teach_frame, "🔄  Retrain Model", "#89dceb", "#1e1e2e",
            self._retrain
        )
        self.btn_retrain.pack(side="left", padx=4)

        # Training step indicator
        self.lbl_step = tk.Label(
            self, text="",
            font=("Segoe UI", 10, "italic"),
            bg="#1e1e2e", fg="#f9e2af"
        )
        self.lbl_step.pack(pady=2)

        # ── Training progress ──
        self.progress_var = tk.IntVar()
        self.progress = ttk.Progressbar(
            self, variable=self.progress_var,
            maximum=len(STEP_NAMES), length=360
        )
        self.progress.pack(padx=16, pady=4)

        # ── Automation Mode ──
        sep2 = tk.Frame(self, height=1, bg="#45475a")
        sep2.pack(fill="x", padx=16, pady=8)

        tk.Label(
            self, text="🤖  Automation",
            font=("Segoe UI", 12, "bold"),
            bg="#1e1e2e", fg="#fab387"
        ).pack()

        tk.Label(
            self, text=(
                "Once the neural network is trained, click «Auto-Download».\n"
                "The AI will find and click all the required buttons by itself."
            ),
            font=("Segoe UI", 9),
            bg="#1e1e2e", fg="#6c7086",
            wraplength=360, justify="center"
        ).pack(pady=4)

        auto_btns = tk.Frame(self, bg="#1e1e2e")
        auto_btns.pack(pady=4)

        self.btn_run = self._btn(
            auto_btns, "⚡  Auto (1 mod)", "#fab387", "#1e1e2e",
            lambda: self._run_auto(loop=False)
        )
        self.btn_run.pack(side="left", padx=4)

        self.btn_loop = self._btn(
            auto_btns, "🔁  Auto-Loop", "#cba6f7", "#1e1e2e",
            lambda: self._run_auto(loop=True)
        )
        self.btn_loop.pack(side="left", padx=4)

        # Loop counter + hint
        self.lbl_loop_info = tk.Label(
            self,
            text="Press F8 at any time to stop the loop",
            font=("Segoe UI", 8, "italic"),
            bg="#1e1e2e", fg="#6c7086"
        )
        self.lbl_loop_info.pack()

        self.lbl_counter = tk.Label(
            self, text="",
            font=("Segoe UI", 11, "bold"),
            bg="#1e1e2e", fg="#cba6f7"
        )
        self.lbl_counter.pack(pady=2)

        # ── Log ──
        sep3 = tk.Frame(self, height=1, bg="#45475a")
        sep3.pack(fill="x", padx=16, pady=8)

        self.log_text = tk.Text(
            self, height=7, width=52,
            bg="#181825", fg="#cdd6f4",
            font=("Consolas", 9),
            relief="flat", bd=0,
            state="disabled"
        )
        self.log_text.pack(padx=16, pady=(0, 8))

        # ── Bottom bar ──
        bottom = tk.Frame(self, bg="#313244")
        bottom.pack(fill="x", padx=0, pady=0)

        self.lbl_sessions = tk.Label(
            bottom,
            text=f"Training sessions: {self.data.get('sessions', 0)}",
            font=("Segoe UI", 8),
            bg="#313244", fg="#6c7086"
        )
        self.lbl_sessions.pack(side="left", padx=12, pady=4)

        tk.Label(
            bottom,
            text="F8 = stop loop",
            font=("Segoe UI", 8),
            bg="#313244", fg="#6c7086"
        ).pack(side="right", padx=12, pady=4)

        # ESC listener to stop everything
        self.bind_all("<Escape>", lambda e: self._stop_all())

    def _btn(self, parent, text, bg, fg, command):
        return tk.Button(
            parent, text=text,
            font=("Segoe UI", 10, "bold"),
            bg=bg, fg=fg,
            activebackground=fg, activeforeground=bg,
            relief="flat", cursor="hand2",
            padx=12, pady=6,
            command=command
        )

    # ── Logging ──────────────────────────────────────────────────────────────

    def _log(self, msg, color="#cdd6f4"):
        ts = time.strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{ts}] {msg}\n")
        self.log_text.configure(state="disabled")
        self.log_text.see("end")

    def _refresh_status(self):
        steps_learned = len(self.templates)
        total = len(STEP_NAMES)
        model_ok = self.brain.trained

        if model_ok and steps_learned == total:
            txt   = f"✅ Neural network trained ({steps_learned}/{total} steps). Ready to go!"
            color = "#a6e3a1"
        elif steps_learned > 0:
            txt   = f"⚠️ Partially trained ({steps_learned}/{total} steps). Continue training."
            color = "#f9e2af"
        else:
            txt   = "❌ Neural network not trained. Start with training mode."
            color = "#f38ba8"

        self.lbl_status.configure(text=txt, fg=color)
        self.progress_var.set(steps_learned)

    # ── Training ─────────────────────────────────────────────────────────────

    def _toggle_teach(self):
        if not self.teach_mode:
            self._start_teach()
        else:
            self._stop_teach()

    def _start_teach(self):
        self.teach_mode    = True
        self.teach_step    = 0
        self.teach_samples = []
        self.btn_teach.configure(text="⏹  Stop Training", bg="#f38ba8")
        self._show_teach_step()
        self.recorder.start()
        self._log("Training mode started. Download a mod now!")

    def _stop_teach(self):
        self.teach_mode = False
        self.recorder.stop()
        self.btn_teach.configure(text="▶  Start Training", bg="#a6e3a1", fg="#1e1e2e")
        self.lbl_step.configure(text="")
        self._log("Training stopped.")
        self._refresh_status()

    def _show_teach_step(self):
        if self.teach_step < len(STEP_NAMES):
            name = STEP_NAMES[self.teach_step]
            self.lbl_step.configure(
                text=f"👉 Step {self.teach_step + 1}/{len(STEP_NAMES)}: {name}"
            )

    def _on_recorded_click(self, x, y):
        """Called when the user clicks the mouse during training."""
        if not self.teach_mode:
            return

        step = self.teach_step
        self._log(f"Step {step}: click ({x}, {y})")

        # Take a screenshot and crop the area around the click
        time.sleep(0.15)
        scr  = screenshot(f"step_{step}_{int(time.time())}.png")
        crop = crop_around(scr, x, y, pad=50)

        # Save template
        self._save_template(step, crop)

        # Add sample for the neural network
        self.teach_samples.append((crop, step))

        # Persist to brain data
        if str(step) not in self.data["steps"]:
            self.data["steps"][str(step)] = []
        self.data["steps"][str(step)].append({"x": x, "y": y, "ts": time.time()})
        save_brain(self.data)

        self._log(f"✅ Template saved for step {step + 1}")

        # Advance to next step
        self.teach_step += 1
        if self.teach_step >= len(STEP_NAMES):
            self._finish_teach_session()
        else:
            self.after(0, self._show_teach_step)

    def _finish_teach_session(self):
        """Finishes one training session."""
        self.recorder.stop()
        self.teach_mode = False
        self.data["sessions"] = self.data.get("sessions", 0) + 1
        save_brain(self.data)

        self._log("🎉 Session recorded! Training neural network...")
        self.btn_teach.configure(text="▶  Start Training", bg="#a6e3a1", fg="#1e1e2e")
        self.lbl_step.configure(text="")

        # Train neural network in the background
        threading.Thread(target=self._do_train, daemon=True).start()

    def _retrain(self):
        """Retrain on all saved samples."""
        self._log("Retraining neural network...")
        threading.Thread(target=self._do_train, daemon=True).start()

    def _do_train(self):
        """Loads all templates and trains the model."""
        samples = []
        for step_id in STEP_NAMES:
            tpl_path = os.path.join(TEMPLATE_DIR, f"step_{step_id}.png")
            if os.path.exists(tpl_path):
                img = Image.open(tpl_path)
                samples.append((img, step_id))

        # Append samples from the current session
        samples += self.teach_samples

        if len(samples) >= 2:
            ok = self.brain.train(samples)
            if ok:
                self._log("🧠 Neural network trained successfully!")
            else:
                self._log("ℹ️ Neural network: too few classes, using templates.")
        else:
            self._log("⚠️ Not enough data. Please run training.")

        self.after(0, self._refresh_status)
        self.after(0, lambda: self.lbl_sessions.configure(
            text=f"Training sessions: {self.data.get('sessions', 0)}"
        ))

    # ── Auto-Execute ─────────────────────────────────────────────────────────

    def _on_stop_key(self):
        """Called when the user presses F8."""
        self._log("⌨️  F8 pressed — stopping loop.")
        self.after(0, self._stop_all)

    def _run_auto(self, loop=False):
        if not self.templates:
            messagebox.showwarning(
                "Not Trained",
                "Train the neural network first!\nClick «Start Training» and download a mod manually."
            )
            return

        if self.playing:
            self._stop_all()
            return

        self.playing    = True
        self.loop_mode  = loop
        self.loop_count = 0

        if loop:
            self.btn_loop.configure(text="⏹  Stop Loop", bg="#f38ba8", fg="#1e1e2e")
            self.btn_run.configure(state="disabled")
            self._log("🔁 Auto-loop started. Press F8 to stop.")
        else:
            self.btn_run.configure(text="⏹  Stop", bg="#f38ba8", fg="#1e1e2e")
            self.btn_loop.configure(state="disabled")

        self.takeover.start()
        self.play_thread = threading.Thread(
            target=self._auto_loop, daemon=True
        )
        self.play_thread.start()

    def _stop_all(self):
        self.playing    = False
        self.loop_mode  = False
        self.teach_mode = False
        self.recorder.stop()
        self.takeover.stop()
        self.btn_run.configure(
            text="⚡  Auto (1 mod)", bg="#fab387", fg="#1e1e2e", state="normal"
        )
        self.btn_loop.configure(
            text="🔁  Auto-Loop", bg="#cba6f7", fg="#1e1e2e", state="normal"
        )
        if self.loop_count:
            self._log(f"⏹ Stopped. Total loops completed: {self.loop_count}")
        else:
            self._log("⏹ Stopped.")
        self.lbl_counter.configure(text="")

    def _auto_loop(self):
        """
        Main automation loop.
        In loop_mode it runs indefinitely while playing=True.
        Stops when the user moves the mouse or presses ESC.
        """
        while self.playing:
            self.loop_count += 1
            prefix = f"[Loop #{self.loop_count}] " if self.loop_mode else ""
            self._log(f"{'🔁' if self.loop_mode else '🤖'} {prefix}Starting. Open Vortex...")

            if self.loop_mode:
                self.after(0, lambda n=self.loop_count: self.lbl_counter.configure(
                    text=f"🔁 Loop #{n}  |  Completed: {n - 1}"
                ))

            success = True
            for step_id in sorted(STEP_NAMES.keys()):
                if not self.playing:
                    success = False
                    break

                name = STEP_NAMES[step_id]
                self._log(f"  ⏳ Step {step_id + 1}: «{name}»")

                found = self._wait_and_click(step_id, timeout=90)
                if not found:
                    self._log(f"  ❌ Step {step_id + 1} not found — skipping loop.")
                    success = False
                    break

                self._log(f"  ✅ Step {step_id + 1} done!")
                time.sleep(1.2)

            if success and self.playing:
                self._log(f"🎉 {prefix}Mod downloaded!")

            if not self.loop_mode:
                # Single mode — exit after one run
                break

            if self.playing:
                # Short pause between loops
                self._log(f"  ⏸ Waiting 3 sec before next loop...")
                for _ in range(30):
                    if not self.playing:
                        break
                    time.sleep(0.1)

        self.after(0, self._stop_all)

    def _wait_and_click(self, step_id, timeout=90):
        """
        Waits for step_id template to appear on screen and clicks it.
        Returns True on success.
        """
        if step_id not in self.templates:
            return False

        template = self.templates[step_id]
        deadline = time.time() + timeout

        while time.time() < deadline:
            if not self.playing:
                return False

            # Take screenshot and search for template
            scr_pil = screenshot()
            scr_cv  = img_to_cv(scr_pil)

            pos, score = match_template(scr_cv, template, threshold=0.65)

            if pos:
                cx, cy = pos
                self._log(f"    🎯 Found (score={score:.2f}) → ({cx}, {cy})")

                self.takeover.set_expected(cx, cy)

                pyautogui.moveTo(cx, cy, duration=0.4)
                time.sleep(0.2)
                pyautogui.click(cx, cy)
                return True

            # Neural network as an additional hint
            if self.brain.trained:
                cx_scr, cy_scr = scr_pil.width // 2, scr_pil.height // 2
                crop = crop_around(scr_pil, cx_scr, cy_scr, pad=100)
                predicted_step = self.brain.predict(crop)
                if predicted_step == step_id:
                    pos2, score2 = match_template(scr_cv, template, threshold=0.50)
                    if pos2:
                        cx2, cy2 = pos2
                        self._log(f"    🧠 Neural network confirmed (score={score2:.2f})")
                        self.takeover.set_expected(cx2, cy2)
                        pyautogui.moveTo(cx2, cy2, duration=0.4)
                        time.sleep(0.2)
                        pyautogui.click(cx2, cy2)
                        return True

            time.sleep(0.5)

        return False


# ─── Entry Point ─────────────────────────────────────────────────────────────

def check_deps():
    if MISSING:
        print("=" * 60)
        print("❌ Missing dependencies:")
        for m in MISSING:
            print(f"   • {m}")
        print()
        print("Install them with:")
        print(f"   pip install {' '.join(MISSING)}")
        print("=" * 60)
        input("Press Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    check_deps()
    app = App()
    app.mainloop()
# 🎮 VortexAI Helper

> Smart AI-powered assistant for automating mod downloads via the Vortex Mod Manager.
> Learns from your actions — so you only have to click once.

---

## 📖 What is this?

Downloading mods through Vortex often involves the same repetitive sequence of clicks: hit **Download Manually** in Vortex, then confirm a couple of prompts in the browser. VortexAI Helper records that sequence once, trains a small neural network on it, and then replicates it automatically — forever, if you want.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 Neural network | Trains an MLPClassifier on screenshots of your clicks |
| 🖼️ Template matching | OpenCV-based visual matching to locate buttons on screen |
| 🔁 Auto-loop | Repeats the download sequence indefinitely until you press F8 |
| ⚡ Single-run mode | Executes one mod download and stops |
| 📸 Screenshot logger | Saves every training screenshot for review |
| 💾 Persistent memory | Brain and model are saved to disk between sessions |

---

## 🖥️ Requirements

- Python **3.8+**
- Windows (uses `ImageGrab` for screenshots; macOS/Linux support may vary)

### Dependencies

```bash
pip install pyautogui pynput opencv-python pillow scikit-learn numpy
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install pyautogui pynput opencv-python pillow scikit-learn numpy

# 2. Run the app
python vortex_ai_helper.py
```

---

## 🧠 How to Train

1. Open **Vortex** and navigate to the mod you want to download.
2. Launch VortexAI Helper and click **▶ Start Training**.
3. The app will prompt you step by step — just download the mod as you normally would.
4. After all 3 steps are recorded, the neural network trains automatically.
5. Repeat with a second mod to improve accuracy (optional but recommended).

> The more sessions you record, the more robust the model becomes.

---

## 🤖 How to Automate

Once trained:

- **⚡ Auto (1 mod)** — runs the sequence once and stops.
- **🔁 Auto-Loop** — keeps repeating until you press **F8** or **ESC**.

The AI will:
1. Watch the screen continuously.
2. Detect each button using template matching + the neural network.
3. Move the mouse and click automatically.
4. Wait up to 90 seconds per step before giving up.

---

## ⌨️ Hotkeys

| Key | Action |
|---|---|
| `F8` | Stop the auto-loop |
| `ESC` | Stop everything immediately |

---

## 📁 File Structure

```
vortex_ai_helper.py     ← Main script
vortex_ai_brain.json    ← Saved click data (auto-created)
vortex_ai_model.pkl     ← Trained neural network (auto-created)
vortex_screenshots/     ← Training screenshots (auto-created)
vortex_templates/       ← Button template images (auto-created)
```

> All data files are created automatically on first run. You can delete them to reset the AI completely.

---

## ⚙️ Configuration

To change the steps the AI learns (e.g. add a 4th browser confirmation), edit the `STEP_NAMES` dictionary at the top of the script:

```python
STEP_NAMES = {
    0: "Click «Download Manually» in Vortex",
    1: "Click browser button (step 1)",
    2: "Click browser button (step 2)",
    # 3: "Click browser button (step 3)",  ← uncomment to add
}
```

To change the stop key from F8 to something else:

```python
STOP_KEY = keyboard.Key.f8   # ← change this
```

---

## 🛡️ Safety

- **Failsafe**: move your mouse to the top-left corner of the screen to trigger PyAutoGUI's built-in emergency stop.
- **F8 / ESC**: always available to halt the loop cleanly.
- The app never modifies game files — it only clicks buttons on your behalf.

---

## 🐛 Troubleshooting

| Problem | Solution |
|---|---|
| Buttons not found | Re-run training; make sure the UI scale/resolution matches training |
| Low match score | Add more training sessions with `▶ Start Training` |
| Loop skips steps | Increase wait time (`timeout=90` in `_wait_and_click`) |
| App won't start | Run `pip install ...` for any missing packages listed on launch |
| Model feels wrong | Click **🔄 Retrain Model** to rebuild from all saved templates |

---

## 📄 License

MIT — free to use, modify, and distribute.

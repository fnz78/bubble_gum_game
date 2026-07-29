# 🎈 Bubble Gum Blow Challenge 💨

Bubble Gum Blow Challenge is an interactive computer vision game built using **Python, OpenCV, and MediaPipe's Face Landmarker Tasks API**. 

Players use their webcam to blow virtual bubble gum. By opening and closing their mouth in front of the camera, players simulate the blowing action, causing a virtual bubble gum bubble to grow on their mouth. The goal is to grow the bubble as large as possible without popping it, or to blow it until it reaches its maximum limit and pops!

---

## 🌟 Features

* **Real-time Face & Mouth Tracking:** Leverages MediaPipe's new Tasks API (v0.10+) to track face landmarks and mouth movement accurately.
* **Intelligent Blow Detection:** Computes mouth openness ratio and tracks mouth movement. True blowing is detected when the player closes their mouth quickly after opening it (simulating an exhaling push).
* **Robust 2-Player Assignment:** Compete side-by-side with a friend. The system automatically detects multiple faces and sorts them left-to-right, ensuring smooth assignment to Player 1 (left) and Player 2 (right) regardless of screen split boundaries.
* **Dynamic Difficulty Settings:** Switch between **Easy**, **Medium**, and **Hard** difficulty levels. This adjusts the rate at which the bubble shrinks when you stop blowing. Toggle it via the **`D`** key or click the **DIFF** button directly on the control panel.
* **Premium Dark & Glow Aesthetics:**
  * Vectorized, high-performance dark vignette background.
  * Semi-transparent frosted-glass UI panels (glassmorphism look).
  * Glowing neon progress bars (pill-style) and colors dynamically assigned per player.
  * Beautiful gradient bubbles with inner rings and specular highlights.
  * High-fidelity particle explosion effect when the bubble pops.
* **Persistent Model Storage:** Downloads the pre-trained model file (~25 MB) automatically on the first run and caches it in a persistent folder so it doesn't need to re-download.

---

## 🎮 Game Controls

### Desktop version (OpenCV)
| Key | Action |
| :--- | :--- |
| **`1`** | Switch to **1-Player Mode** |
| **`2`** | Switch to **2-Player Mode** |
| **`D`** | Cycle **Difficulty** (Easy ➔ Medium ➔ Hard) |
| **`S`** | **Start** / **Pause** / **Resume** the game |
| **`R`** | **Reset** the game / return to menu |
| **`Q`** / **`ESC`** | **Quit** the game |

*Note: You can also hover and left-click the buttons on the right-hand **Controls Panel** using your mouse.*

### Web version (Streamlit)
* **Start WebRTC stream**: Click the **Start** button in the WebRTC player component to request webcam access.
* **Settings & Play Controls**: Use the interactive dropdowns and buttons on the left **Sidebar** to change player mode, difficulty, start/pause, or reset the game.

---

## 🛠️ Tech Stack

* **Language:** Python 3.8+
* **Framework (Web):** Streamlit + `streamlit-webrtc`
* **Computer Vision:** OpenCV (`opencv-python`)
* **Machine Learning & Tracking:** Google MediaPipe Tasks (`mediapipe`)
* **Numerical Operations:** NumPy (`numpy`)

---

## 🚀 Setup & Installation

### 1. Prerequisites
Make sure you have Python 3.8 or higher installed on your machine and a working webcam.

### 2. Create and Activate a Virtual Environment

Open your terminal, navigate to the `bubble_gum_game` directory, and run:

**On Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install the required libraries using pip:
```bash
pip install -r requirements.txt
```

### 4. Run the Game

#### Desktop version:
Launch the native OpenCV app:
```bash
python src/main.py
```

#### Streamlit web version:
Launch the Streamlit browser-based app locally:
```bash
streamlit run streamlit_app.py
```

*Note: On the very first run, the game will automatically download the `face_landmarker.task` model file (~25 MB) from Google's servers. Please ensure you are connected to the internet.*

---

## 🌐 Deploying to Streamlit Cloud

You can easily deploy the web version of the game for free on Streamlit Community Cloud:

1. **Push your code to GitHub** (follow the steps below).
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **New app**, select your repository and branch, set the main file path to `streamlit_app.py`, and click **Deploy**.
4. To ensure OpenCV runs correctly on Streamlit's Linux servers, Streamlit will automatically read your `requirements.txt` and install all dependencies. 
   *(Note: Free Google STUN servers are configured in `streamlit_app.py` to handle remote WebRTC NAT traversal automatically).*

---

## 📦 Packaging as a Desktop App

You can package the desktop application into a standalone executable (`.exe` on Windows, executable binary on macOS/Linux) so that other users can play it without installing Python:

1. Install PyInstaller in your virtual environment:
   ```bash
   pip install pyinstaller
   ```
2. Build the standalone executable:
   ```bash
   pyinstaller --onefile --noconsole --name "BubbleGumChallenge" src/main.py
   ```
3. Once completed, your executable will be located in the `dist/` directory.

---

## 🐙 How to Push to GitHub

To push this repository to your own GitHub account:

1. **Initialize Git Repository** (if not already initialized):
   ```bash
   git init
   ```
2. **Add Files to Staging** (our `.gitignore` ensures node modules/virtual environments/tasks files are ignored):
   ```bash
   git add .
   ```
3. **Commit the Changes**:
   ```bash
   git commit -m "Initial commit: Bubble Gum Blow Challenge with Streamlit support"
   ```
4. **Create a New Repo on GitHub** (without initializing it with a README, gitignore, or license).
5. **Rename your main branch and add the remote URL**:
   ```bash
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   ```
6. **Push your code**:
   ```bash
   git push -u origin main
   ```

---

## 📂 Project Structure

```text
bubble_gum_game/
├── requirements.txt      # Python dependencies (opencv, mediapipe, numpy, streamlit, webrtc, av)
├── README.md             # Project documentation (this file)
├── streamlit_app.py      # Streamlit web entry point
├── venv/                 # Virtual environment (ignored by git)
└── src/
    ├── __init__.py       # Package initialization
    ├── main.py           # Desktop entry point, core game loop, camera feed, and controls
    ├── ui_drawing.py     # Clean UI drawing functions, design tokens, HUDs, menus, and overlays
    ├── face_tracker.py   # Wrapper around MediaPipe Face Landmarker for landmark extraction
    ├── game_logic.py     # Game states (countdown, menu, playing), player tracker, and blowing logic
    ├── bubble.py         # Bubble physics, animation interpolation, scaling, and pop particles
    └── utils.py          # Color schemes, advanced glow/gradient rendering, and font styling
```

---

## 📖 How to Play

1. **Start the Game:** Select player mode (`1` or `2`) and press **`S`** (or click Start). A countdown will begin.
2. **Align Face:** Look directly at the webcam. Ensure your face is fully visible.
3. **Blow a Bubble:**
   * Open your mouth wide.
   * Swiftly close it to "blow" and push air into the virtual bubble.
   * Repeat this open-and-close motion. If you stop blowing, the bubble will slowly shrink!
4. **Pop to Win:** Grow the bubble until it reaches maximum size (`MAX_BUBBLE_PX = 130`). The first player to pop their bubble wins!

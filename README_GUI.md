# Protenix GUI

A modern, cross-platform graphical user interface (GUI) for **Protenix**, built with PyQt6. This GUI provides an intuitive way to manage single and batch protein structure prediction tasks, configure run parameters, visualize 3D results with pLDDT scoring, and safely manage prediction history.

## Quick Start (Recommended)

The easiest way to start the Protenix GUI is using the provided launcher scripts. These scripts will automatically check your Python environment, install any missing dependencies, and start the GUI.

**For Linux / macOS:**
```bash
# Make the script executable (only needed once)
chmod +x run_protenix_gui.sh

# Run the launcher
./run_protenix_gui.sh
```

**For Windows:**
Double-click the `run_protenix_gui.bat` file, or run it from the command prompt:
```cmd
run_protenix_gui.bat
```

---

## Manual Installation

If you prefer to install dependencies manually, you can follow these steps:

### 1. Install Python Packages
Run the following command to install the required Python dependencies:

```bash
pip install PyQt6 PyQt6-WebEngine send2trash
```

**Dependency details:**
*   `PyQt6`: The core UI framework.
*   `PyQt6-WebEngine`: Required for the built-in 3D structure viewer (3Dmol.js). *If this fails to install, the GUI will still run, but you won't be able to use the "Preview" feature.*
*   `send2trash`: Used to safely move deleted tasks/samples to the system trash bin instead of permanently deleting them.

### 2. System Dependencies (Linux/Ubuntu Only)
If you are running on a Linux desktop (e.g., Ubuntu), PyQt6 requires the `xcb` platform plugin. If you encounter an error like `qt.qpa.plugin: Could not find the Qt platform plugin "xcb"`, you need to install the missing system libraries:

```bash
sudo apt-get update
sudo apt-get install -y libxcb-cursor0 libxcb-xinerama0 libxcb-xkb1 libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-shape0
```

If not working, try:
pip install PyQt6 PyQt6-Qt6 PyQt6-sip --only-binary :all: -i https://pypi.tuna.tsinghua.edu.cn/simple

### 3. External Tools (Optional but Recommended)
*   **PyMOL**: The GUI includes a "View in PyMOL" button. To use this, ensure PyMOL is installed on your system. If you use a custom PyMOL path, you may need to ensure it's accessible via your system's PATH.
*   **Protenix Backend**: The GUI acts as a frontend. It expects `protenix pred` or a custom `inference.py` script to be available to execute the prediction tasks.

## How to Run

Once dependencies are installed, simply execute the main Python script:

```bash
python Protenix_GUI.py
```

## Key Features
*   **Single Prediction:** Easy form-based entry for sequences, modifications, and covalent bonds.
*   **Batch Prediction:** Excel-compatible table view for creating multiple jobs at once, with CSV template support.
*   **3D Viewer:** Built-in WebGL viewer using `3Dmol.js` to visualize `.cif` results, colored by pLDDT confidence scores.
*   **Native Charts:** Per-atom pLDDT line charts built directly into the UI.
*   **Safe Data Management:** Batch select and delete tasks or individual samples. Files are moved to the system Trash bin by default.
*   **Smart CLI Integration:** Automatically translates your UI settings into the correct command-line arguments for the Protenix backend.

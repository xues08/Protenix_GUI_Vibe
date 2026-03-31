#!/bin/bash

echo "================================================"
echo "    Protenix GUI Launcher (Linux / macOS)       "
echo "================================================"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH."
    exit 1
fi

echo "Checking and installing required Python packages..."
# Install requirements
python3 -m pip install PyQt6 PyQt6-WebEngine send2trash

# Check if we are on Linux and need to install system dependencies for PyQt6
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Linux detected. Checking for necessary Qt platform plugins (xcb)..."
    # A simple check: if we can import PyQt6.QtWidgets without error, xcb is likely fine or we are headless.
    # But it's safer to just provide the apt-get command as a hint if it fails.
    if ! python3 -c "from PyQt6.QtWidgets import QApplication; import sys; app = QApplication(sys.argv)" &> /dev/null; then
        echo ""
        echo "WARNING: It seems the Qt platform plugin 'xcb' might be missing."
        echo "If the GUI fails to start, please run the following command to install system dependencies:"
        echo "  sudo apt-get update && sudo apt-get install -y libxcb-cursor0 libxcb-xinerama0 libxcb-xkb1 libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-shape0"
        echo ""
    fi
fi

echo "Starting Protenix GUI..."
python3 Protenix_GUI.py

#!/usr/bin/env python3.11
#
# Sets up a Python virtual environment for the project based on the host OS
# (Windows or macOS), and installs all dependencies from requirements.txt.
# Automatically chooses the correct venv directory and Python binary per platform.
#
import subprocess
import sys
import venv
from pathlib import Path

BASE = Path(__file__).parent.resolve()

if sys.platform == "win32":
    VENV_DIR = BASE / "venv-win"
elif sys.platform == "darwin":
    VENV_DIR = BASE / "venv-mac"
else:
    raise Exception(f"Unsupported platform: {sys.platform}")

REQ_FILE = BASE / "requirements.txt"

def create_venv():
    if VENV_DIR.exists():
        print("Virtual environment already exists:", VENV_DIR)
        return

    print("Creating virtual environment:", VENV_DIR)
    builder = venv.EnvBuilder(with_pip=True)
    builder.create(VENV_DIR)
    print("Done.")

def get_python_bin():
    """Return path to python inside venv."""
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"

def install_requirements():
    if not REQ_FILE.exists():
        print("No requirements.txt found — skipping install.")
        return

    python_bin = get_python_bin()
    print("Installing requirements from requirements.txt …")
    subprocess.check_call([str(python_bin), "-m", "pip", "install", "-r", str(REQ_FILE)])
    print("Python requirements installed.")
    print("All requirements installed.")

def main():
    create_venv()
    install_requirements()
    print("Environment setup complete.")

if __name__ == "__main__":
    main()

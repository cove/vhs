#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

def run(cmd):
    subprocess.check_call(cmd)

def main():
    root = Path(__file__).parent.resolve()
    venv_dir = root / "venv"
    req_file = root / "requirements.txt"

    # 1. Create venv
    if not venv_dir.exists():
        print("Creating virtual environment...")
        run([sys.executable, "-m", "venv", str(venv_dir)])
    else:
        print("Virtual environment already exists.")

    # 2. Path to pip/python inside venv (Windows only)
    pip = venv_dir / "Scripts" / "pip.exe"
    python = venv_dir / "Scripts" / "python.exe"

    # 3. Upgrade pip
    print("Upgrading pip...")
    run([str(pip), "install", "--upgrade", "pip"])

    # 4. Install packages from requirements.txt
    if req_file.exists():
        print("Installing packages from requirements.txt...")
        run([str(pip), "install", "-r", str(req_file)])
    else:
        print("requirements.txt not found; skipping package installation.")

    # 5. Show packages
    print("Installed packages:")
    run([str(pip), "list"])

    print("\nDone! Activate your virtual environment using:")
    print(rf"{venv_dir}\Scripts\activate")

if __name__ == "__main__":
    main()

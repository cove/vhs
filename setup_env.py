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

    pip = venv_dir / "Scripts" / "pip.exe"
    python = venv_dir / "Scripts" / "python.exe"


    if req_file.exists():
        print("Installing packages from requirements.txt...")
        run([str(pip), "install", "-r", str(req_file)])
    else:
        print("requirements.txt not found; skipping package installation.")

    print("\nDone! Activate your virtual environment using:")
    print(rf"{venv_dir}\Scripts\activate")

if __name__ == "__main__":
    main()

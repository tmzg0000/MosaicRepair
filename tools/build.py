"""Build on the target OS; copy the resulting directory as a portable application."""
from pathlib import Path
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
command = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir", "--windowed",
           "--name", "MosaicRepair", "--copy-metadata", "PySide6-Essentials", "--copy-metadata", "shiboken6",
           str(root / "main.py")]
subprocess.run(command, cwd=root, check=True)
output = root / "dist" / "MosaicRepair"
shutil.copy2(root / "README.md", output / "README.md")
if (root / "samples").exists():
    shutil.copytree(root / "samples", output / "samples", dirs_exist_ok=True)
# Keep dependency license notices with the distributed portable application.
notice = output / "third_party_licenses"
notice.mkdir(exist_ok=True)
from importlib.metadata import distribution
for package in ("numpy", "Pillow", "opencv-python-headless", "PySide6-Essentials", "shiboken6"):
    dist = distribution(package)
    for file in dist.files or ():
        if any(word in str(file).lower() for word in ("license", "copying", "notice")) and dist.locate_file(file).is_file():
            destination = notice / package / str(file).replace("..", "_")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dist.locate_file(file), destination)
print(f"Portable application: {output}")

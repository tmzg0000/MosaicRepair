"""Build on the target OS; copy the resulting directory as a portable application."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

from verify_windows_icon import read_ico, verify_executable_icon

root = Path(__file__).resolve().parents[1]
icon = root / "mosaic_repair" / "assets" / "app-icon.ico"
read_ico(icon)  # Fail before packaging if an old/incomplete source archive was used.
command = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir", "--windowed",
           "--name", "MosaicRepair", "--copy-metadata", "PySide6-Essentials", "--copy-metadata", "shiboken6",
           "--icon", str(icon),
           "--add-data", f"{root / 'mosaic_repair' / 'assets' / 'app-icon.png'}:mosaic_repair/assets",
           str(root / "main.py")]
subprocess.run(command, cwd=root, check=True)
output = root / "dist" / "MosaicRepair"
if sys.platform == "win32":
    executable = output / "MosaicRepair.exe"
    sizes = verify_executable_icon(executable, icon)
    report = {"exe_icon_verified": True, "icon_sizes": sizes,
              "icon_sha256": hashlib.sha256(icon.read_bytes()).hexdigest(),
              "exe_sha256": hashlib.sha256(executable.read_bytes()).hexdigest()}
    (output / "build-info.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("EXE icon verified: " + ", ".join(map(str, sizes)) + " px", flush=True)
    # Windows may retain the previous PyInstaller icon at the same EXE path.
    import ctypes
    ctypes.windll.shell32.SHChangeNotify(0x08000000, 0, None, None)
shutil.copy2(icon, output / "MosaicRepair.ico")
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

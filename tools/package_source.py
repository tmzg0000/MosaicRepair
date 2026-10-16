"""Create a clean, portable source archive without local dependencies or private inputs."""
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
dist = root / "dist"
dist.mkdir(exist_ok=True)
target = dist / "MosaicRepair-source.zip"
files = [root/name for name in ("README.md", "PLAN.md", "TEST_REPORT.md", "main.py", "bootstrap.py", "requirements.txt",
                                "requirements-dev.txt", "start_windows.bat", "build_windows.bat", ".gitignore", ".gitattributes")]
for directory in ("mosaic_repair", "tests", "tools", "samples", "docs", ".github"):
    files.extend(p for p in (root/directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(set(files)):
        if path.is_file():
            archive.write(path, "MosaicRepair/" + path.relative_to(root).as_posix())
print(target)

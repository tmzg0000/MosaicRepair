"""Verify the actual PE icon resources, independently of Explorer's icon cache."""
from pathlib import Path
import argparse
import struct


def read_ico(path):
    data = Path(path).read_bytes()
    if len(data) < 6:
        raise ValueError("Truncated ICO header")
    reserved, kind, count = struct.unpack_from("<HHH", data)
    directory_end = 6 + count * 16
    if reserved != 0 or kind != 1 or count == 0 or directory_end > len(data):
        raise ValueError("Invalid ICO directory")
    frames = []
    for index in range(count):
        entry = struct.unpack_from("<BBBBHHII", data, 6 + index * 16)
        size, offset = entry[-2:]
        if size == 0 or offset < directory_end or offset + size > len(data):
            raise ValueError("Truncated ICO image")
        frames.append((entry[:7], data[offset:offset + size]))
    return frames


def verify_group(group, load_image, frames):
    """Compare every size and every image byte, not just the resource count."""
    if len(group) < 6:
        raise ValueError("EXE icon group is truncated")
    reserved, kind, count = struct.unpack_from("<HHH", group)
    if (reserved, kind, count) != (0, 1, len(frames)) or len(group) != 6 + count * 14:
        raise ValueError("EXE icon directory does not match app-icon.ico")
    for index, (metadata, image) in enumerate(frames):
        entry = struct.unpack_from("<BBBBHHIH", group, 6 + index * 14)
        if entry[:7] != metadata or load_image(entry[-1]) != image:
            size = metadata[0] or 256
            raise ValueError(f"EXE icon at {size}px does not match app-icon.ico")


def verify_executable_icon(executable, icon):
    # Use the same Windows resource reader that PyInstaller installs for builds.
    from PyInstaller.compat import win32api

    frames = read_ico(icon)
    handle = win32api.LoadLibraryEx(str(Path(executable).resolve()), 0, 2)
    try:
        names = win32api.EnumResourceNames(handle, 14)  # RT_GROUP_ICON
        if len(names) != 1:
            raise ValueError("EXE must contain exactly the application's icon group")
        group = win32api.LoadResource(handle, 14, names[0])
        verify_group(group, lambda image_id: win32api.LoadResource(handle, 3, image_id), frames)
    finally:
        win32api.FreeLibrary(handle)
    return [metadata[0] or 256 for metadata, _ in frames]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("--icon", type=Path,
                        default=Path(__file__).resolve().parents[1] / "mosaic_repair/assets/app-icon.ico")
    args = parser.parse_args()
    sizes = verify_executable_icon(args.executable, args.icon)
    print("EXE icon verified against app-icon.ico: " + ", ".join(map(str, sizes)) + " px")

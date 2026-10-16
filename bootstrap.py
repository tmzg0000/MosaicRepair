"""Visible Windows startup; standard library only, with native errors in startup.log."""
from pathlib import Path
import os
import queue
import struct
import subprocess
import sys
import threading
import time
import traceback


class StartupError(RuntimeError):
    pass


class Reporter:
    def __init__(self, path):
        self.file = Path(path).open("a", encoding="utf-8", buffering=1)

    def write(self, text):
        text = str(text)
        print(text, flush=True)
        self.file.write(text + "\n")

    def close(self):
        self.file.close()


def require_supported_python(version=None, bits=None):
    version = tuple(sys.version_info[:2]) if version is None else tuple(version)
    bits = struct.calcsize("P") * 8 if bits is None else bits
    if not (3, 11) <= version <= (3, 13) or bits != 64:
        raise StartupError("Install 64-bit Python 3.12 (supported: 3.11-3.13). "
                           "Detected Python %s.%s, %s-bit." % (*version, bits))


def run_logged(command, report, cwd, env=None, timeout=None, heartbeat=False):
    """Stream stderr and stdout, with activity messages for silent preparation."""
    report.write("> " + subprocess.list2cmdline([str(v) for v in command]))
    child_env = os.environ.copy()
    child_env.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONFAULTHANDLER": "1"})
    if env:
        child_env.update(env)
    process = subprocess.Popen([str(v) for v in command], cwd=cwd, env=child_env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               encoding="utf-8", errors="replace", bufsize=1)
    output = queue.Queue()
    def read_output():
        try:
            for line in process.stdout:
                output.put(line.rstrip("\r\n"))
        finally:
            output.put(None)
    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    started = time.monotonic()
    try:
        while True:
            if timeout is not None and time.monotonic() - started > timeout:
                process.kill()
                raise StartupError("This step timed out after %s seconds. Check startup.log." % timeout)
            try:
                line = output.get(timeout=2)
            except queue.Empty:
                if heartbeat:
                    report.write("  Still working... %s seconds" % int(time.monotonic()-started))
                continue
            if line is None:
                break
            report.write(line)
        return process.wait()
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        reader.join(timeout=2)
        process.stdout.close()


def environment_python(folder):
    return Path(folder) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


PYTHON_PROBE = ("import sys,struct; print(sys.executable,sys.version); "
                "sys.exit(0 if (3,11)<=sys.version_info[:2]<=(3,13) and struct.calcsize('P')==8 else 1)")


def prepare_environment(root, report):
    report.write("[2/4] Preparing the project Python environment...")
    default = root / ".venv"
    fallback = root / ".venv-win"
    for folder in (default, fallback):
        python = environment_python(folder)
        if python.is_file():
            try:
                if run_logged([python, "-c", PYTHON_PROBE], report, root, timeout=30) == 0:
                    return python
            except (OSError, StartupError) as exc:
                report.write("Existing environment is unusable: " + str(exc))
    # Preserve an unusable/copied .venv rather than deleting it.
    folder = fallback if default.exists() else default
    if folder.exists():
        raise StartupError("The existing %s environment cannot run. Rename that folder and retry; "
                           "your images and project files do not need to be changed." % folder.name)
    base = getattr(sys, "_base_executable", sys.executable)
    code = run_logged([base, "-u", "-m", "venv", folder], report, root, timeout=180, heartbeat=True)
    if code != 0:
        raise StartupError("Could not create the environment. See the command output above.")
    return environment_python(folder)


def dependency_probe(requirements):
    pins = {}
    for line in Path(requirements).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, sep, version = line.partition("==")
            if not sep:
                raise StartupError("Unsupported requirement entry: " + line)
            pins[name] = version
    return ("from importlib.metadata import version; "
            "pins=%r; "
            "assert all(version(k)==v for k,v in pins.items()), 'Dependency versions changed'; "
            "import cv2,numpy,PIL; from PySide6.QtWidgets import QApplication; "
            "print('Dependencies OK')" % pins)


def install_dependencies(python, root, report, filename="requirements.txt"):
    code = run_logged([python, "-u", "-m", "pip", "install", "--disable-pip-version-check",
                       "--only-binary=:all:", "--timeout", "20", "--retries", "1",
                       "-r", root / filename], report, root, timeout=1800, heartbeat=True)
    if code != 0:
        raise StartupError("Dependency installation failed. Check the network/pip error above. "
                           "The startup window will stay open; startup.log contains the full output.")


def launch(root, report, args):
    require_supported_python()
    report.write("Python: " + sys.executable)
    report.write("Version: " + sys.version.replace("\n", " "))
    report.write("Project folder: " + str(root))
    python = prepare_environment(root, report)
    report.write("[3/4] Checking Qt, OpenCV, NumPy and Pillow...")
    probe = dependency_probe(root / "requirements.txt")
    if run_logged([python, "-c", probe], report, root, timeout=60) != 0:
        report.write("Installing required packages. The Qt download can take several minutes.")
        install_dependencies(python, root, report)
        if run_logged([python, "-c", probe], report, root, timeout=60) != 0:
            raise StartupError("Packages were installed, but imports failed. Send startup.log; "
                               "do not keep reinstalling them.")
    if args == ["--prepare-only"]:
        report.write("Environment is ready.")
        return 0
    if args == ["--build"]:
        report.write("[4/4] Installing build tools, testing and packaging...")
        install_dependencies(python, root, report, "requirements-dev.txt")
        has_tests = (root / "tests").is_dir()
        if not has_tests:
            report.write("Local tests are excluded from Git; running the application startup check instead.")
        check = [python,"-m","pytest","-q"] if has_tests else [python,root / "main.py","--smoke-test"]
        if run_logged(check, report, root,
                      env={"QT_QPA_PLATFORM": "offscreen"}, timeout=600, heartbeat=True) != 0:
            raise StartupError("Validation failed; packaging stopped. See startup.log.")
        if run_logged([python, root / "tools/build.py"], report, root, timeout=1800, heartbeat=True) != 0:
            raise StartupError("Packaging failed. See startup.log.")
        report.write("Ready: " + str(root / "dist/MosaicRepair/MosaicRepair.exe"))
        return 0
    report.write("[4/4] Opening the application window...")
    report.write("Keep this console open while using the app. Errors are recorded in startup.log.")
    started = time.monotonic()
    code = run_logged([python, "-u", root / "main.py", *args], report, root)
    report.write("Application exited with code %s after %.1f seconds." % (code, time.monotonic()-started))
    if code != 0:
        raise StartupError("The application could not start or closed with an error. Send startup.log.")
    return 0


def main(args=None):
    root = Path(__file__).resolve().parent
    args = sys.argv[1:] if args is None else list(args)
    try:
        report = Reporter(root / "startup.log")
    except OSError as exc:
        print("Cannot write startup.log. Extract the ZIP to a writable local folder: " + str(exc), flush=True)
        return 1
    try:
        return launch(root, report, args)
    except Exception as exc:
        report.write("\nERROR: " + str(exc))
        report.write(traceback.format_exc())
        return 1
    finally:
        report.close()


if __name__ == "__main__":
    raise SystemExit(main())

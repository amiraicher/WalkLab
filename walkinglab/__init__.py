"""
WalkingLab Package.

Performs platform-specific DLL path initialization on Windows and disables
OpenVINO telemetry file writes in restricted or CI environments.
"""
import os
import sys

os.environ.setdefault("CI", "true")

_dll_dir_handles = []

if sys.platform == "win32":
    _path_entries = os.environ.get("PATH", "").split(os.pathsep)
    for _dll_dir in (sys.prefix, os.path.join(sys.prefix, "Scripts")):
        if os.path.isdir(_dll_dir):
            if hasattr(os, "add_dll_directory"):
                _dll_dir_handles.append(os.add_dll_directory(_dll_dir))
            if _dll_dir not in _path_entries:
                os.environ["PATH"] = _dll_dir + os.pathsep + os.environ.get("PATH", "")

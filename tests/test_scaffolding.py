import pytest
import importlib
import walkinglab  # Ensure Windows DLL search paths are initialized

def test_ui_app_import():
    """Verify that the main UI app can be imported and has the run_app function."""
    try:
        from walkinglab.ui import app
        assert hasattr(app, "run_app"), "walkinglab.ui.app module should have a run_app function"
        assert callable(app.run_app), "run_app should be a callable function"
    except ImportError as e:
        pytest.fail(f"Failed to import walkinglab.ui.app: {e}")

@pytest.mark.parametrize("module_name", [
    "walkinglab.ui.annotation",
    "walkinglab.ui.pose",
    "walkinglab.ui.settings",
    "walkinglab.ui.metrics",
])
def test_ui_modules_exist(module_name):
    """Verify that UI modules exist and have a render function."""
    try:
        mod = importlib.import_module(module_name)
        assert hasattr(mod, "render"), f"{module_name} should have a render function"
        assert callable(mod.render), "render should be a callable function"
    except ImportError as e:
        pytest.fail(f"Failed to import {module_name}: {e}")

@pytest.mark.parametrize("module_name", [
    "openvino",
    "torch",
    "walkinglab.backend.inference",
    "walkinglab.backend.filtering",
    "walkinglab.backend.analysis",
    "walkinglab.ui.app",
])
def test_backend_modules_exist(module_name):
    """Verify that core runtime dependencies and backend/UI modules can be imported."""
    try:
        importlib.import_module(module_name)
    except ImportError as e:
        pytest.fail(f"Failed to import {module_name}: {e}")


def test_clean_process_import_order():
    """Verify that openvino and the main.py import chain load cleanly in a fresh interpreter."""
    import subprocess
    import sys

    code = (
        "from walkinglab.ui.app import run_app; "
        "import openvino as ov; "
        "import torch; "
        "import walkinglab.backend.inference as inference; "
        "core = ov.Core(); "
        "assert core is not None"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"Clean process import failed with code {result.returncode}:\n"
        f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )



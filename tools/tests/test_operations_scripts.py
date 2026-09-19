from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def test_operations_shell_scripts_parse() -> None:
    for script in sorted((ROOT / "operations").glob("*.sh")):
        result = subprocess.run(
            ["bash", "-n", str(script)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{script}: {result.stderr}"


def test_profile_marker_precedes_dependent_service_restarts() -> None:
    script = (ROOT / "operations/qwen-profile-switch.sh").read_text(encoding="utf-8")
    marker = script.index('install -m 0644 -o root -g root "${marker_tmp}" "${PROFILE_MARKER}"')
    exporter_restart = script.index('systemctl restart "${EXPORTER}"')
    gateway_restart = script.index('systemctl restart "${GATEWAY}"')
    assert marker < exporter_restart < gateway_restart
    assert "restore_profile_enablement" in script
    assert "trap 'on_signal 130' INT" in script
    assert "trap 'on_signal 143' TERM" in script

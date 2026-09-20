from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def test_operations_shell_scripts_parse() -> None:
    for script in sorted((ROOT / "operations").rglob("*.sh")):
        result = subprocess.run(
            ["bash", "-n", str(script)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{script}: {result.stderr}"


def test_rtx5090_profile_is_explicit_and_loopback_only() -> None:
    profile = ROOT / "operations/hardware/rtx5090"
    installer = (profile / "install-native-q4.sh").read_text(encoding="utf-8")
    service = (profile / "systemd/ai-rtx5090-qwen3.8-q4-native.service").read_text(encoding="utf-8")
    monitor = (profile / "systemd/ai-rtx5090-monitor.service").read_text(encoding="utf-8")

    assert "AI_SERVER_RTX5090_CONFIRM=install" in installer
    assert "systemctl disable" not in installer
    assert "127.0.0.1" in service
    assert "GGML_CUDA_ENABLE_UNIFIED_MEMORY" not in service
    assert "127.0.0.1" in monitor


def test_profile_marker_precedes_dependent_service_restarts() -> None:
    script = (ROOT / "operations/qwen-profile-switch.sh").read_text(encoding="utf-8")
    marker = script.index('install -m 0644 -o root -g root "${marker_tmp}" "${PROFILE_MARKER}"')
    exporter_restart = script.index('systemctl restart "${EXPORTER}"')
    gateway_restart = script.index('systemctl restart "${GATEWAY}"')
    assert marker < exporter_restart < gateway_restart
    assert "restore_profile_enablement" in script
    assert "trap 'on_signal 130' INT" in script
    assert "trap 'on_signal 143' TERM" in script

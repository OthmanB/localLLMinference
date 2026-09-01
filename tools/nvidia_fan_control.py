#!/usr/bin/env python3
"""Set a fixed speed on every controllable fan of one NVIDIA GPU."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import sys
from ctypes import byref, c_uint, c_void_p


NVML_SUCCESS = 0


def configure_functions(nvml: ctypes.CDLL) -> None:
    nvml.nvmlInit_v2.restype = c_uint
    nvml.nvmlShutdown.restype = c_uint
    nvml.nvmlDeviceGetCount_v2.argtypes = [ctypes.POINTER(c_uint)]
    nvml.nvmlDeviceGetCount_v2.restype = c_uint
    nvml.nvmlDeviceGetHandleByIndex_v2.argtypes = [c_uint, ctypes.POINTER(c_void_p)]
    nvml.nvmlDeviceGetHandleByIndex_v2.restype = c_uint
    nvml.nvmlDeviceGetNumFans.argtypes = [c_void_p, ctypes.POINTER(c_uint)]
    nvml.nvmlDeviceGetNumFans.restype = c_uint
    nvml.nvmlDeviceSetFanSpeed_v2.argtypes = [c_void_p, c_uint, c_uint]
    nvml.nvmlDeviceSetFanSpeed_v2.restype = c_uint


def check(result: int, operation: str) -> None:
    if result != NVML_SUCCESS:
        raise RuntimeError(f"{operation} failed with NVML status {result}")


def set_fan_speed(gpu_index: int, speed: int) -> int:
    library = ctypes.util.find_library("nvidia-ml") or "libnvidia-ml.so.1"
    nvml = ctypes.CDLL(library)
    configure_functions(nvml)
    initialized = False
    try:
        check(nvml.nvmlInit_v2(), "NVML initialization")
        initialized = True
        count = c_uint()
        check(nvml.nvmlDeviceGetCount_v2(byref(count)), "GPU enumeration")
        if gpu_index >= count.value:
            raise RuntimeError(f"GPU index {gpu_index} is unavailable; found {count.value} GPU(s)")

        handle = c_void_p()
        check(nvml.nvmlDeviceGetHandleByIndex_v2(gpu_index, byref(handle)), "GPU handle lookup")
        fan_count = c_uint()
        check(nvml.nvmlDeviceGetNumFans(handle, byref(fan_count)), "fan enumeration")
        if fan_count.value == 0:
            raise RuntimeError(f"GPU {gpu_index} exposes no controllable fans")

        for fan_index in range(fan_count.value):
            check(
                nvml.nvmlDeviceSetFanSpeed_v2(handle, fan_index, speed),
                f"setting GPU {gpu_index} fan {fan_index} to {speed}%",
            )
        print(f"Set GPU {gpu_index} fan(s) to {speed}% ({fan_count.value} fan(s))")
        return 0
    finally:
        if initialized:
            nvml.nvmlShutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to control (default: 0)")
    parser.add_argument("--speed", type=int, default=70, help="Fixed fan speed percentage (default: 70)")
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("--gpu must be non-negative")
    if not 0 <= args.speed <= 100:
        parser.error("--speed must be between 0 and 100")
    try:
        return set_fan_speed(args.gpu, args.speed)
    except (OSError, RuntimeError) as error:
        print(f"nvidia fan control unavailable: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

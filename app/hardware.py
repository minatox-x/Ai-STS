"""
Detects CPU/RAM/GPU/VRAM/disk without requiring the user to enter anything.

GPU detection is best-effort: NVIDIA via GPUtil/pynvml, otherwise falls back to
WMI (Windows) for a display name only (no reliable VRAM figure for AMD/Intel
integrated GPUs from Python alone) -- we say so rather than guessing.
"""
from __future__ import annotations

import platform
import shutil
from dataclasses import dataclass, asdict
from typing import Optional

import psutil


@dataclass
class GPUInfo:
    name: Optional[str] = None
    vram_gb: Optional[float] = None
    vendor: Optional[str] = None  # "nvidia" | "amd" | "intel" | "unknown"
    vram_confidence: str = "unknown"  # "measured" | "estimated" | "unknown"


@dataclass
class HardwareProfile:
    os_name: str
    cpu_name: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    cpu_arch: str
    ram_total_gb: float
    ram_available_gb: float
    disk_free_gb: float
    gpu: GPUInfo
    performance_tier: str  # "entry" | "mid" | "high"

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _detect_cpu_name() -> str:
    try:
        import cpuinfo  # py-cpuinfo
        info = cpuinfo.get_cpu_info()
        name = info.get("brand_raw")
        if name:
            return name
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


def _detect_gpu() -> GPUInfo:
    # Try NVIDIA first -- this is the only path that gives a trustworthy VRAM number.
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            g = gpus[0]
            return GPUInfo(
                name=g.name,
                vram_gb=round(g.memoryTotal / 1024, 2),
                vendor="nvidia",
                vram_confidence="measured",
            )
    except Exception:
        pass

    # Windows fallback: WMI gives a name and an "AdapterRAM" figure that is often
    # wrong/capped for modern cards (32-bit field), so treat it as estimated only.
    if platform.system() == "Windows":
        try:
            import wmi
            w = wmi.WMI()
            for gpu in w.Win32_VideoController():
                vram_gb = None
                try:
                    if gpu.AdapterRAM and gpu.AdapterRAM > 0:
                        vram_gb = round(gpu.AdapterRAM / (1024**3), 2)
                except Exception:
                    pass
                vendor = "unknown"
                name = (gpu.Name or "").lower()
                if "nvidia" in name:
                    vendor = "nvidia"
                elif "amd" in name or "radeon" in name:
                    vendor = "amd"
                elif "intel" in name:
                    vendor = "intel"
                return GPUInfo(
                    name=gpu.Name,
                    vram_gb=vram_gb,
                    vendor=vendor,
                    vram_confidence="estimated" if vram_gb else "unknown",
                )
        except Exception:
            pass

    return GPUInfo(name=None, vram_gb=None, vendor="unknown", vram_confidence="unknown")


def _performance_tier(ram_gb: float, cores: int, gpu: GPUInfo) -> str:
    has_real_gpu = gpu.vendor == "nvidia" and (gpu.vram_gb or 0) >= 4
    if ram_gb >= 16 and cores >= 8 and has_real_gpu:
        return "high"
    if ram_gb >= 8 and cores >= 4:
        return "mid"
    return "entry"


def detect_hardware() -> HardwareProfile:
    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(".")

    gpu = _detect_gpu()
    ram_total_gb = round(vm.total / (1024**3), 2)
    cores_logical = psutil.cpu_count(logical=True) or 1
    cores_physical = psutil.cpu_count(logical=False) or cores_logical

    return HardwareProfile(
        os_name=f"{platform.system()} {platform.release()}",
        cpu_name=_detect_cpu_name(),
        cpu_cores_physical=cores_physical,
        cpu_cores_logical=cores_logical,
        cpu_arch=platform.machine(),
        ram_total_gb=ram_total_gb,
        ram_available_gb=round(vm.available / (1024**3), 2),
        disk_free_gb=round(disk.free / (1024**3), 2),
        gpu=gpu,
        performance_tier=_performance_tier(ram_total_gb, cores_physical, gpu),
    )

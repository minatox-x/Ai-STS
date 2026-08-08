from app.hardware import detect_hardware, _performance_tier, GPUInfo


def test_detect_hardware_runs_without_error():
    hw = detect_hardware()
    assert hw.ram_total_gb > 0
    assert hw.cpu_cores_logical >= 1
    assert hw.performance_tier in ("entry", "mid", "high")


def test_performance_tier_entry_level():
    tier = _performance_tier(4.0, 2, GPUInfo())
    assert tier == "entry"


def test_performance_tier_high_needs_real_gpu():
    weak_gpu = GPUInfo(vendor="intel", vram_gb=None)
    tier = _performance_tier(32.0, 16, weak_gpu)
    assert tier != "high"  # no dedicated NVIDIA GPU with enough VRAM -> not "high"

    strong_gpu = GPUInfo(vendor="nvidia", vram_gb=8)
    tier = _performance_tier(32.0, 16, strong_gpu)
    assert tier == "high"

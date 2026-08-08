from app.hardware import HardwareProfile, GPUInfo
from app.model_registry import ModelRegistry


def _hw(ram_available_gb, disk_free_gb=50, gpu=None):
    return HardwareProfile(
        os_name="Windows 11",
        cpu_name="Test CPU",
        cpu_cores_physical=4,
        cpu_cores_logical=8,
        cpu_arch="x86_64",
        ram_total_gb=ram_available_gb + 2,
        ram_available_gb=ram_available_gb,
        disk_free_gb=disk_free_gb,
        gpu=gpu or GPUInfo(),
        performance_tier="mid",
    )


def test_catalog_loads():
    reg = ModelRegistry()
    assert len(reg.list("asr")) > 0
    assert len(reg.list("llm")) > 0
    assert len(reg.list("tts")) > 0


def test_low_ram_never_recommends_gpu_only_higher_quality_as_recommended():
    reg = ModelRegistry()
    hw = _hw(ram_available_gb=3.0)  # entry-level machine, no GPU
    recs = reg.recommend("tts", hw)
    # Qwen3-TTS needs a GPU to be practical -- must not be the "recommended" pick
    # on a GPU-less low-RAM machine.
    assert recs["recommended"] is None or recs["recommended"].id != "qwen3-tts-0.6b"


def test_safer_is_never_larger_than_recommended_ram():
    reg = ModelRegistry()
    hw = _hw(ram_available_gb=6.0)
    recs = reg.recommend("llm", hw)
    if recs["safer"] and recs["recommended"]:
        assert recs["safer"].ram_recommended_gb <= recs["recommended"].ram_recommended_gb

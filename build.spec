# PyInstaller spec for Friend.
# Build with:   pyinstaller build.spec
# Produces:     dist/Friend/Friend.exe  (onedir build -- faster startup than onefile)
#
# Models are intentionally NOT bundled here -- they are downloaded post-install
# into %APPDATA%\Friend\Models, keeping this build small per the spec's
# "lightweight installer" requirement.

# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_data_files

# 1. Base datas
datas = [
    ("app/models.json", "app"),
]

# 2. Collect runtime data and assets
datas += collect_data_files("piper")

# 3. Collect all datas, binaries, and hiddenimports for heavy/C-extension packages
fw_datas, fw_binaries, fw_hidden = collect_all("faster_whisper")
ct2_datas, ct2_binaries, ct2_hidden = collect_all("ctranslate2")
onnx_datas, onnx_binaries, onnx_hidden = collect_all("onnxruntime")

datas += fw_datas + ct2_datas + onnx_datas
binaries = fw_binaries + ct2_binaries + onnx_binaries

# 4. Define all hidden imports
hiddenimports = [
    "app.providers.asr_faster_whisper",
    "app.providers.asr_qwen3",
    "app.providers.llm_llamacpp",
    "app.providers.tts_piper",
    "app.providers.tts_qwen3",
    "app.providers.gemini_provider",
    "av",
    "huggingface_hub",
    "tokenizers",
] + fw_hidden + ct2_hidden + onnx_hidden

a = Analysis(
    ["app/main.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    excludes=["torch", "transformers"],  # heavy GPU-tier deps: only pulled in if user opts into that tier
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Friend",
    debug=False,
    strip=False,
    upx=True,
    console=False,
    icon=None,  # add app/ui/assets/icon.ico here once you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    name="Friend",
)

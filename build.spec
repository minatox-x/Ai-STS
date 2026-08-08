# PyInstaller spec for Friend.
# Build with:  pyinstaller build.spec
# Produces:    dist/Friend/Friend.exe  (onedir build -- faster startup than onefile)
#
# Models are intentionally NOT bundled here -- they are downloaded post-install
# into %APPDATA%\Friend\Models, keeping this build small per the spec's
# "lightweight installer" requirement.

# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files

datas = [
    ("app/models.json", "app"),
]
datas += collect_data_files("piper")  # bundles Piper's small runtime data files if present

a = Analysis(
    ["app/main.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "app.providers.asr_faster_whisper",
        "app.providers.asr_qwen3",
        "app.providers.llm_llamacpp",
        "app.providers.tts_piper",
        "app.providers.tts_qwen3",
        "app.providers.gemini_provider",
    ],
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

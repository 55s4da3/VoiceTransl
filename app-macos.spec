# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files
from pathlib import Path

qt_material_datas = collect_data_files('qt_material')


def tool_data(name):
    for candidate in (Path('dist') / name, Path(name)):
        if candidate.is_dir() and any(candidate.iterdir()):
            return [(str(candidate), name)]
    return []


app_datas = [
    ('icon.png', '.'), ('avatar.png', '.'), ('llama', 'llama'),
    ('crispasr', 'crispasr'), ('plugins', 'plugins'),
    ('ffmpeg', 'ffmpeg'), ('translation_guidelines', 'translation_guidelines'),
] + tool_data('separate') + tool_data('translate') + qt_material_datas


a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=app_datas,
    hiddenimports=['tiktoken_ext.openai_public', 'tiktoken_ext'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch','onnx','onnxruntime','librosa','soundfile'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VoiceTransl',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='VoiceTransl',
)

app = BUNDLE(
    coll,
    name='VoiceTransl.app',
    icon='icon.icns',
    bundle_identifier='com.voicetransl.app',
)

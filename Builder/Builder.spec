# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Builder application.

Build with:
    ./venv/bin/pyinstaller Builder.spec
"""

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect all submodules for packages that have dynamic imports
hiddenimports = [
    'Builder',
    'Builder.common',
    'Builder.common.image',
    'Builder.common.utils',
    'Builder.MeshBuilder',
    'Builder.MeshBuilder.session',
    'Builder.MeshBuilder.models',
    'Builder.StimBuilder',
    'Builder.StimBuilder.session',
    'Builder.StimBuilder.models',
    'Builder.ui',
    'Builder.ui.server',
    'flask',
    'jinja2',
    'jinja2.ext',
    'werkzeug',
    'PIL',
    'PIL.Image',
    'numpy',
    'scipy',
    'scipy.ndimage',
    'cv2',
    'cairosvg',
]

# Data files to include
datas = [
    ('Builder/ui/templates', 'Builder/ui/templates'),
    ('Builder/ui/static', 'Builder/ui/static'),
]

a = Analysis(
    ['Builder/launcher.py'],
    pathex=['.'],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'PyQt5', 'PyQt6'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Builder',
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
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Builder',
)

app = BUNDLE(
    coll,
    name='Builder.app',
    icon=None,
    bundle_identifier='com.heartconduction.builder',
    info_plist={
        'CFBundleName': 'Builder',
        'CFBundleDisplayName': 'Builder',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSHighResolutionCapable': True,
        'LSBackgroundOnly': False,
    },
)

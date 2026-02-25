"""
py2app setup script for Builder application.

Build with:
    ./venv/bin/python setup.py py2app
"""

from setuptools import setup
import os

APP = ['Builder/launcher.py']
APP_NAME = 'Builder'

# Collect all data files from Builder package
DATA_FILES = []

# Add templates
template_dir = 'Builder/ui/templates'
if os.path.exists(template_dir):
    templates = [os.path.join(template_dir, f) for f in os.listdir(template_dir) if f.endswith('.html')]
    if templates:
        DATA_FILES.append((template_dir, templates))

# Add static CSS
css_dir = 'Builder/ui/static/css'
if os.path.exists(css_dir):
    css_files = [os.path.join(css_dir, f) for f in os.listdir(css_dir) if f.endswith('.css')]
    if css_files:
        DATA_FILES.append((css_dir, css_files))

# Add static JS
js_dir = 'Builder/ui/static/js'
if os.path.exists(js_dir):
    js_files = [os.path.join(js_dir, f) for f in os.listdir(js_dir) if f.endswith('.js')]
    if js_files:
        DATA_FILES.append((js_dir, js_files))

OPTIONS = {
    'argv_emulation': False,
    'iconfile': None,  # Add icon file path here if you have one
    'plist': {
        'CFBundleName': APP_NAME,
        'CFBundleDisplayName': 'Builder',
        'CFBundleIdentifier': 'com.heartconduction.builder',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSHighResolutionCapable': True,
        'LSBackgroundOnly': False,
    },
    'packages': [
        'Builder',
        'Builder.common',
        'Builder.MeshBuilder',
        'Builder.StimBuilder',
        'Builder.ui',
        'flask',
        'jinja2',
        'werkzeug',
        'PIL',
        'numpy',
        'scipy',
        'cv2',
    ],
    'includes': [
        'Builder.ui.server',
        'cairosvg',
    ],
    'excludes': [
        'tkinter',
        'matplotlib',
        'PyQt5',
        'PyQt6',
    ],
    'resources': [
        'Builder/ui/templates',
        'Builder/ui/static',
    ],
}

setup(
    app=APP,
    name=APP_NAME,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)

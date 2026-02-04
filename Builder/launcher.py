#!/usr/bin/env python3
"""
Builder App Launcher

This script launches the Flask server and opens the web browser.
"""

import os
import sys
import threading
import webbrowser
import time

# Ensure we can import our package
if getattr(sys, 'frozen', False):
    # Running as packaged app (PyInstaller)
    # _MEIPASS contains the path to bundled resources
    bundle_dir = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    sys.path.insert(0, bundle_dir)
    os.chdir(bundle_dir)

    # Set template and static folders for Flask
    template_folder = os.path.join(bundle_dir, 'Builder', 'ui', 'templates')
    static_folder = os.path.join(bundle_dir, 'Builder', 'ui', 'static')
else:
    # Running as script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    sys.path.insert(0, parent_dir)
    os.chdir(parent_dir)
    template_folder = None
    static_folder = None

from Builder.ui.server import app

# Override Flask template/static folders if running frozen
if getattr(sys, 'frozen', False) and template_folder:
    app.template_folder = template_folder
    app.static_folder = static_folder

PORT = 5001


def open_browser():
    """Open browser after short delay to let server start."""
    time.sleep(1.5)
    webbrowser.open(f'http://localhost:{PORT}')


def main():
    """Main entry point."""
    # Start browser opener in background
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()

    # Run Flask server
    print(f"Starting Builder at http://localhost:{PORT}")
    app.run(
        host='127.0.0.1',
        port=PORT,
        debug=False,
        use_reloader=False,
        threaded=True
    )


if __name__ == '__main__':
    main()

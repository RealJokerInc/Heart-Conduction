"""
Interactive REPL mode for Builder.
"""

import sys
from typing import Optional

from .commands.common import (
    prompt_choice,
    prompt_string,
    print_header,
    print_error,
    print_success,
)


class InteractiveSession:
    """Interactive REPL session managing both builders."""

    def __init__(self):
        self.mesh_session = None
        self.stim_session = None
        self.running = True

    def run(self) -> int:
        """Run the interactive REPL."""
        print_header("Builder Interactive Mode")
        print("Type 'help' for available commands, 'quit' to exit.\n")

        while self.running:
            try:
                cmd = input("builder> ").strip().lower()
                self._handle_command(cmd)
            except KeyboardInterrupt:
                print("\n")
                continue
            except EOFError:
                print("\n")
                self.running = False

        return 0

    def _handle_command(self, cmd: str) -> None:
        """Handle a single command."""
        if not cmd:
            return

        parts = cmd.split()
        command = parts[0]
        args = parts[1:] if len(parts) > 1 else []

        commands = {
            'help': self._cmd_help,
            'quit': self._cmd_quit,
            'exit': self._cmd_quit,
            'mesh': self._cmd_mesh,
            'stim': self._cmd_stim,
            'status': self._cmd_status,
            'load': self._cmd_load,
            'detect': self._cmd_detect,
            'configure': self._cmd_configure,
            'summary': self._cmd_summary,
        }

        if command in commands:
            commands[command](args)
        else:
            print(f"Unknown command: {command}")
            print("Type 'help' for available commands.")

    def _cmd_help(self, args) -> None:
        """Show help."""
        print("""
Available commands:
  mesh              Switch to MeshBuilder mode
  stim              Switch to StimBuilder mode
  load <path>       Load an image file
  detect            Detect colors in loaded image
  configure         Configure groups/regions interactively
  summary           Show current session summary
  status            Show current mode and session state
  help              Show this help
  quit, exit        Exit interactive mode
""")

    def _cmd_quit(self, args) -> None:
        """Quit interactive mode."""
        self.running = False
        print("Goodbye!")

    def _cmd_mesh(self, args) -> None:
        """Switch to mesh mode."""
        from Builder.MeshBuilder import MeshBuilderSession
        if self.mesh_session is None:
            self.mesh_session = MeshBuilderSession()
        print_success("Switched to MeshBuilder mode")

    def _cmd_stim(self, args) -> None:
        """Switch to stim mode."""
        from Builder.StimBuilder import StimBuilderSession
        if self.stim_session is None:
            self.stim_session = StimBuilderSession()
        print_success("Switched to StimBuilder mode")

    def _cmd_status(self, args) -> None:
        """Show current status."""
        print("\nSession Status:")
        print(f"  MeshBuilder: {'active' if self.mesh_session else 'not started'}")
        print(f"  StimBuilder: {'active' if self.stim_session else 'not started'}")

        if self.mesh_session and self.mesh_session.image_path:
            print(f"  Mesh image: {self.mesh_session.image_path.name}")
        if self.stim_session and self.stim_session.image_path:
            print(f"  Stim image: {self.stim_session.image_path.name}")

    def _cmd_load(self, args) -> None:
        """Load an image."""
        if not args:
            path = prompt_string("Image path")
        else:
            path = args[0]

        # Determine which session to load into
        if self.mesh_session is None and self.stim_session is None:
            mode = prompt_choice("Load into:", ["MeshBuilder", "StimBuilder"])
            if mode == 0:
                self._cmd_mesh([])
            else:
                self._cmd_stim([])

        try:
            if self.mesh_session:
                self.mesh_session.load_image(path)
                print_success(f"Loaded into MeshBuilder: {self.mesh_session.image_size}")
            elif self.stim_session:
                self.stim_session.load_image(path)
                print_success(f"Loaded into StimBuilder: {self.stim_session.image_size}")
        except Exception as e:
            print_error(f"Failed to load: {e}")

    def _cmd_detect(self, args) -> None:
        """Detect colors."""
        if self.mesh_session and self.mesh_session.image_array is not None:
            groups = self.mesh_session.detect_colors()
            print_success(f"MeshBuilder: Found {len(groups)} color groups")
        elif self.stim_session and self.stim_session.image_array is not None:
            regions = self.stim_session.detect_colors()
            print_success(f"StimBuilder: Found {len(regions)} color regions")
        else:
            print_error("No image loaded. Use 'load <path>' first.")

    def _cmd_configure(self, args) -> None:
        """Configure groups/regions interactively."""
        if self.mesh_session and self.mesh_session.color_groups:
            from .commands.mesh import _interactive_configure
            _interactive_configure(self.mesh_session)
        elif self.stim_session and self.stim_session.stim_regions:
            from .commands.stim import _interactive_configure
            _interactive_configure(self.stim_session)
        else:
            print_error("No colors detected. Use 'detect' first.")

    def _cmd_summary(self, args) -> None:
        """Show session summary."""
        if self.mesh_session:
            print(self.mesh_session.summary())
        if self.stim_session:
            print(self.stim_session.summary())
        if not self.mesh_session and not self.stim_session:
            print("No active sessions.")


def run_interactive() -> int:
    """Run the interactive REPL."""
    session = InteractiveSession()
    return session.run()

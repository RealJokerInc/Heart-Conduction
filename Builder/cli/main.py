"""
CLI entry point and argument parsing.

Usage:
    builder mesh <image>    Start MeshBuilder workflow
    builder stim <image>    Start StimBuilder workflow
    builder interactive     Launch interactive REPL
"""

import argparse
import sys


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog='builder',
        description='Build tissue meshes and stimulation maps from images.'
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Mesh command
    mesh_parser = subparsers.add_parser('mesh', help='Build tissue mesh from image')
    mesh_parser.add_argument('image', help='Path to input image')
    mesh_parser.add_argument('--output', '-o', help='Output file path')
    mesh_parser.add_argument('--interactive', '-i', action='store_true',
                            help='Run in interactive mode')

    # Stim command
    stim_parser = subparsers.add_parser('stim', help='Build stimulation map from image')
    stim_parser.add_argument('image', help='Path to input image')
    stim_parser.add_argument('--output', '-o', help='Output file path')
    stim_parser.add_argument('--interactive', '-i', action='store_true',
                            help='Run in interactive mode')

    # Interactive REPL
    subparsers.add_parser('interactive', help='Launch interactive REPL')

    return parser


def main(argv=None):
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == 'mesh':
        from .commands.mesh import run_mesh_workflow
        return run_mesh_workflow(args)

    elif args.command == 'stim':
        from .commands.stim import run_stim_workflow
        return run_stim_workflow(args)

    elif args.command == 'interactive':
        from .interactive import run_interactive
        return run_interactive()

    return 0


if __name__ == '__main__':
    sys.exit(main())

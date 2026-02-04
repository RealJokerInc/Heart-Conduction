"""
MeshBuilder CLI commands.
"""

import sys
from pathlib import Path
from typing import Tuple

from .common import (
    prompt_string,
    prompt_float,
    prompt_choice,
    prompt_confirm,
    print_header,
    print_table,
    print_success,
    print_error,
)


def run_mesh_workflow(args) -> int:
    """Run the MeshBuilder workflow."""
    from Builder.MeshBuilder import MeshBuilderSession
    from Builder.common.utils import color_to_hex

    print_header("MeshBuilder")

    # Create session and load image
    session = MeshBuilderSession()

    try:
        print(f"Loading image: {args.image}")
        session.load_image(args.image)
        print_success(f"Loaded {session.image_size[0]}x{session.image_size[1]} px")
    except Exception as e:
        print_error(f"Failed to load image: {e}")
        return 1

    # Detect colors
    print("\nDetecting colors...")
    groups = session.detect_colors()
    print_success(f"Found {len(groups)} color groups")

    # Show detected groups
    _display_groups(session)

    # Interactive mode
    if args.interactive or not args.output:
        _interactive_configure(session)

    # Show summary
    print("\n" + session.summary())

    return 0


def _display_groups(session) -> None:
    """Display detected color groups."""
    from Builder.common.utils import color_to_hex

    print("\nDetected color groups:")
    headers = ["#", "Color", "Pixels", "Type", "Status"]
    rows = []

    for i, group in enumerate(session.get_color_groups(), 1):
        hex_color = color_to_hex(group.color)
        group_type = "background" if group.is_background else "tissue"
        status = "configured" if group.is_configured else "pending"
        rows.append([i, hex_color, group.pixel_count, group_type, status])

    print_table(headers, rows)


def _interactive_configure(session) -> None:
    """Interactive configuration of tissue groups."""
    from Builder.common.utils import color_to_hex

    # Configure dimensions first
    print_header("Tissue Dimensions")
    width = prompt_float("Tissue width (cm)", default=1.0)
    height = prompt_float("Tissue height (cm)", default=1.0)
    dx = prompt_float("Spatial resolution dx (cm)", default=0.01)
    session.set_dimensions(width, height, dx)
    print_success(f"Mesh resolution: {session.get_mesh_resolution()}")

    # Configure each tissue group
    tissue_groups = session.tissue_groups
    if not tissue_groups:
        print("\nNo tissue groups to configure.")
        return

    print_header("Configure Tissue Groups")

    for i, group in enumerate(tissue_groups, 1):
        hex_color = color_to_hex(group.color)
        print(f"\n--- Group {i}/{len(tissue_groups)}: {hex_color} ({group.pixel_count} px) ---")

        # Option to skip or mark as background
        action = prompt_choice(
            "Action:",
            ["Configure as tissue", "Mark as background", "Skip"],
            default=1
        )

        if action == 1:  # Mark as background
            session.mark_as_background(group.color)
            print_success("Marked as background")
            continue
        elif action == 2:  # Skip
            continue

        # Configure tissue properties
        label = prompt_string("Label (e.g., 'ventricle', 'atria')")
        cell_type = prompt_string("Cell type (e.g., 'ventricular', 'atrial')")

        print("\nConductivity tensor (cm²/ms):")
        D_xx = prompt_float("  D_xx (longitudinal)", default=0.001)
        D_yy = prompt_float("  D_yy (transverse)", default=0.0003)
        D_xy = prompt_float("  D_xy (off-diagonal)", default=0.0)

        session.configure_group(
            color=group.color,
            label=label,
            cell_type=cell_type,
            D_xx=D_xx,
            D_yy=D_yy,
            D_xy=D_xy
        )
        print_success(f"Configured '{label}'")

    # Summary
    if session.all_groups_configured:
        print_success("\nAll tissue groups configured!")
    else:
        unconfigured = len(session.unconfigured_groups)
        print(f"\n{unconfigured} group(s) still need configuration.")

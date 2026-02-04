"""
StimBuilder CLI commands.
"""

import sys
from pathlib import Path

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


def run_stim_workflow(args) -> int:
    """Run the StimBuilder workflow."""
    from Builder.StimBuilder import StimBuilderSession
    from Builder.StimBuilder.models import StimType
    from Builder.common.utils import color_to_hex

    print_header("StimBuilder")

    # Create session and load image
    session = StimBuilderSession()

    try:
        print(f"Loading image: {args.image}")
        session.load_image(args.image)
        print_success(f"Loaded {session.image_size[0]}x{session.image_size[1]} px")
    except Exception as e:
        print_error(f"Failed to load image: {e}")
        return 1

    # Detect colors
    print("\nDetecting colors...")
    regions = session.detect_colors()
    print_success(f"Found {len(regions)} color regions")

    # Show detected regions
    _display_regions(session)

    # Interactive mode
    if args.interactive or not args.output:
        _interactive_configure(session)

    # Show summary
    print("\n" + session.summary())

    return 0


def _display_regions(session) -> None:
    """Display detected stim regions."""
    from Builder.common.utils import color_to_hex

    print("\nDetected color regions:")
    headers = ["#", "Color", "Pixels", "Type", "Status"]
    rows = []

    for i, region in enumerate(session.get_stim_regions(), 1):
        hex_color = color_to_hex(region.color)
        region_type = "background" if region.is_background else "stimulus"
        status = "configured" if region.is_configured else "pending"
        rows.append([i, hex_color, region.pixel_count, region_type, status])

    print_table(headers, rows)


def _interactive_configure(session) -> None:
    """Interactive configuration of stim regions."""
    from Builder.StimBuilder.models import StimType
    from Builder.common.utils import color_to_hex

    active_regions = session.active_regions
    if not active_regions:
        print("\nNo stimulus regions to configure.")
        return

    print_header("Configure Stimulus Regions")
    print("Note: Timing parameters (start, duration, BCL) are set in simulation.\n")

    for i, region in enumerate(active_regions, 1):
        hex_color = color_to_hex(region.color)
        print(f"\n--- Region {i}/{len(active_regions)}: {hex_color} ({region.pixel_count} px) ---")

        # Option to skip or mark as background
        action = prompt_choice(
            "Action:",
            ["Configure as stimulus", "Mark as background", "Skip"],
            default=1
        )

        if action == 1:  # Mark as background
            session.mark_as_background(region.color)
            print_success("Marked as background")
            continue
        elif action == 2:  # Skip
            continue

        # Configure stimulus properties
        label = prompt_string("Label (e.g., 'S1_pacing', 'S2_premature')")

        stim_type_idx = prompt_choice(
            "Stimulus type:",
            ["Current injection (uA/cm²)", "Voltage clamp (mV)"],
            default=1
        )

        if stim_type_idx == 0:
            amplitude = prompt_float("Amplitude (uA/cm²)", default=50.0)
            session.configure_current_injection(
                color=region.color,
                label=label,
                amplitude=amplitude
            )
        else:
            voltage = prompt_float("Voltage (mV)", default=-80.0)
            session.configure_voltage_clamp(
                color=region.color,
                label=label,
                voltage=voltage
            )

        print_success(f"Configured '{label}'")

    # Summary
    if session.all_regions_configured:
        print_success("\nAll stimulus regions configured!")
    else:
        unconfigured = len(session.unconfigured_regions)
        print(f"\n{unconfigured} region(s) still need configuration.")

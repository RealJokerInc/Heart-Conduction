"""
Builder UI - Flask Server

Run with:
    cd "/Users/catecholamines/Documents/Heart Conduction/Monodomain/Engine_V5.4"
    ../../venv/bin/python -m ui.server
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from io import BytesIO

from flask import Flask, render_template, request, jsonify, session, send_file, redirect, url_for
from PIL import Image
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mesh_builder import MeshBuilderSession
from mesh_builder.common.utils import color_to_hex
from mesh_builder.export import export_mesh

app = Flask(__name__)
app.secret_key = 'builder-secret-key-change-in-production'

# Store sessions in memory (for simplicity)
sessions = {}

# Presets - Conductive tissues
CONDUCTIVE_PRESETS = {
    'myocardial': {
        'label': 'Myocardial',
        'cell_type': 'myocardial',
        'D_xx': 0.001,
        'D_yy': 0.0003,
        'D_xy': 0.0,
    },
    'endocardial': {
        'label': 'Endocardial',
        'cell_type': 'endocardial',
        'D_xx': 0.002,
        'D_yy': 0.0006,
        'D_xy': 0.0,
    },
    'epicardial': {
        'label': 'Epicardial',
        'cell_type': 'epicardial',
        'D_xx': 0.0012,
        'D_yy': 0.00036,
        'D_xy': 0.0,
    },
}

# Non-conductive (background/infarct) - user names these
NONCONDUCTIVE_PRESET = {
    'cell_type': 'non_conductive',
    'D_xx': 0.0,
    'D_yy': 0.0,
    'D_xy': 0.0,
    'is_non_conductive': True,
}

# Special preset for conductive tissue (transparent/white areas)
CONDUCTIVE_TISSUE_PRESET = {
    'label': 'Conductive Tissue',
    'is_conductive': True,
}


def get_session_id():
    """Get or create session ID."""
    if 'session_id' not in session:
        import uuid
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']


def get_builder_session():
    """Get the MeshBuilder session for current user."""
    sid = get_session_id()
    return sessions.get(sid)


def set_builder_session(builder_session, image_data=None):
    """Store the MeshBuilder session."""
    sid = get_session_id()
    sessions[sid] = {
        'builder': builder_session,
        'image_data': image_data,
        'groups': [],
    }


# --- Routes ---

@app.route('/')
def start():
    """Start page."""
    return render_template('start.html')


@app.route('/upload')
def upload():
    """Upload page."""
    return render_template('upload.html')


@app.route('/loading')
def loading():
    """Loading page."""
    return render_template('loading.html')


@app.route('/workspace')
def workspace():
    """Workspace page."""
    data = get_builder_session()
    if data is None:
        return redirect(url_for('upload'))
    return render_template('workspace.html', conductive_presets=CONDUCTIVE_PRESETS)


# --- API Endpoints ---

@app.route('/api/upload', methods=['POST'])
def api_upload():
    """Handle file upload."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Read file data
    file_data = file.read()
    filename = file.filename.lower()

    # Convert SVG to PNG if needed
    if filename.endswith('.svg'):
        try:
            import cairosvg
            file_data = cairosvg.svg2png(bytestring=file_data)
        except Exception as e:
            return jsonify({'error': f'SVG conversion failed: {str(e)}'}), 400

    # Store in session for processing
    sid = get_session_id()
    sessions[sid] = {'pending_image': file_data}

    return jsonify({'success': True, 'filename': file.filename})


@app.route('/api/process', methods=['POST'])
def api_process():
    """Process the uploaded image."""
    sid = get_session_id()
    data = sessions.get(sid)

    if not data or 'pending_image' not in data:
        return jsonify({'error': 'No image to process'}), 400

    image_data = data['pending_image']

    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            tmp.write(image_data)
            tmp_path = tmp.name

        # Create MeshBuilder session
        builder = MeshBuilderSession()
        builder.load_image(tmp_path)

        # Clean up temp file
        os.unlink(tmp_path)

        # Detect colors
        builder.detect_colors()

        # Filter small groups
        builder.filter_small_groups(min_percent=0.1)

        # Set default dimensions (10cm x 10cm, dx=0.01cm)
        builder.set_dimensions(10.0, 10.0, 0.01)

        # Build groups list
        groups = []
        unlabeled_count = 0
        for i, group in enumerate(builder.get_color_groups(), 1):
            # Auto-detect background: white (255,255,255) or transparent (alpha=0)
            # Initialize as Myocardial (conductive)
            is_white = (group.color[:3] == (255, 255, 255))
            is_transparent = (len(group.color) == 4 and group.color[3] == 0)
            is_background = is_white or is_transparent

            if is_background:
                # Auto-configure as Myocardial
                label = 'Myocardial'
                preset = 'myocardial'
                configured = True
                is_conductive = True
                is_non_conductive = False
                group_number = None  # No group number for auto-configured
                # Also configure in the builder
                builder.mark_as_background(group.color)
            else:
                unlabeled_count += 1
                group_number = unlabeled_count  # Store the assigned group number
                label = f'Unlabeled Group {group_number}'
                preset = None
                configured = False
                is_conductive = False
                is_non_conductive = False

            groups.append({
                'index': i,
                'color': list(group.color),
                'hex': color_to_hex(group.color),
                'pixel_count': group.pixel_count,
                'label': label,
                'preset': preset,
                'configured': configured,
                'is_conductive': is_conductive,
                'is_non_conductive': is_non_conductive,
                'group_number': group_number,  # For consistent naming
            })

        # Store session
        sessions[sid] = {
            'builder': builder,
            'image_data': image_data,
            'groups': groups,
        }

        return jsonify({'success': True, 'group_count': len(groups)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/session')
def api_session():
    """Get current session data."""
    data = get_builder_session()
    if data is None:
        return jsonify({'error': 'No session'}), 404

    builder = data['builder']
    dims = builder.tissue_dimensions if builder else (1.0, 1.0)
    dx = builder.dx if builder else 0.01
    resolution = builder.get_mesh_resolution() if builder else (101, 101)

    return jsonify({
        'groups': data['groups'],
        'image_size': list(builder.image_size) if builder else None,
        'dimensions': {
            'width': dims[0],
            'height': dims[1],
            'dx': dx,
            'nx': resolution[0],
            'ny': resolution[1],
        }
    })


@app.route('/api/configure', methods=['POST'])
def api_configure():
    """Configure a group with a preset or custom name."""
    data = get_builder_session()
    if data is None:
        return jsonify({'error': 'No session'}), 404

    req = request.json
    group_index = req.get('index')
    config_type = req.get('type')  # 'conductive', 'non_conductive', or 'conductive_tissue'
    preset_key = req.get('preset')  # For conductive: 'myocardial', 'endocardial', 'epicardial'
    custom_name = req.get('name')   # For non_conductive: custom label

    builder = data['builder']

    # Find and update group
    for group in data['groups']:
        if group['index'] == group_index:
            color = tuple(group['color'])

            if config_type == 'conductive' and preset_key in CONDUCTIVE_PRESETS:
                # Conductive tissue type (endo, epi, myo) - shown as dashed white
                preset = CONDUCTIVE_PRESETS[preset_key]
                builder.mark_as_background(color)  # Mark as background for dashed display
                group['label'] = preset['label']
                group['is_conductive'] = True
                group['is_non_conductive'] = False
                group['preset'] = preset_key

            elif config_type == 'non_conductive':
                # Non-conductive (infarct, scar, etc.) - user provides name
                label = custom_name or 'Infarct'
                builder.configure_group(
                    color=color,
                    label=label,
                    cell_type='non_conductive',
                    D_xx=0.0,
                    D_yy=0.0,
                    D_xy=0.0,
                )
                group['label'] = label
                group['is_conductive'] = False
                group['is_non_conductive'] = True
                group['preset'] = 'non_conductive'

            else:
                return jsonify({'error': 'Invalid configuration'}), 400

            group['configured'] = True
            break

    return jsonify({'success': True})


def create_diagonal_pattern(height, width):
    """Create a diagonal stripe pattern (white with grey lines at 45 degrees)."""
    pattern = np.ones((height, width, 3), dtype=np.uint8) * 255  # White base

    # Draw grey diagonal lines
    stripe_width = 4  # pixels between stripes
    grey = np.array([154, 140, 152], dtype=np.uint8)  # #9a8c98

    for i in range(height):
        for j in range(width):
            # Diagonal stripe pattern at 45 degrees
            if (i + j) % (stripe_width * 2) < stripe_width:
                pattern[i, j] = grey

    return pattern


@app.route('/api/image')
def api_image():
    """Get the processed image."""
    data = get_builder_session()
    if data is None or data['builder'] is None:
        return jsonify({'error': 'No session'}), 404

    # Get original image
    original_array = data['builder'].image_array
    img_array = original_array.copy()

    # Get number of channels
    num_channels = original_array.shape[-1] if len(original_array.shape) == 3 else 1

    # Apply diagonal pattern to background colors
    for group in data['groups']:
        if group.get('is_conductive', False):
            bg_color = group['color']

            # Create mask using FULL color (including alpha if present)
            # This prevents transparent-black from matching opaque-black
            if num_channels == 4 and len(bg_color) == 4:
                mask = np.all(original_array == bg_color, axis=-1)
            elif num_channels == 4:
                mask = np.all(original_array[..., :3] == bg_color[:3], axis=-1)
            else:
                mask = np.all(original_array[..., :3] == bg_color[:3], axis=-1)

            # Create diagonal pattern
            pattern = create_diagonal_pattern(img_array.shape[0], img_array.shape[1])

            # Apply pattern to masked pixels
            if num_channels == 4:
                img_array[mask, :3] = pattern[mask]
                img_array[mask, 3] = 255  # Make fully opaque
            else:
                img_array[mask] = pattern[mask]

    img = Image.fromarray(img_array)

    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)

    return send_file(buffer, mimetype='image/png')


@app.route('/api/image/highlight/<int:group_index>')
def api_image_highlight(group_index):
    """Get image with specific group highlighted by index."""
    data = get_builder_session()
    if data is None or data['builder'] is None:
        return jsonify({'error': 'No session'}), 404

    # Find the group by index
    target_group = None
    for group in data['groups']:
        if group['index'] == group_index:
            target_group = group
            break

    if target_group is None:
        return jsonify({'error': 'Group not found'}), 404

    # Get original image for mask creation
    original_array = data['builder'].image_array
    num_channels = original_array.shape[-1] if len(original_array.shape) == 3 else 1

    # Get the full color for this group
    target_color = target_group['color']

    # Create mask for the selected color using FULL color match
    if num_channels == 4 and len(target_color) == 4:
        highlight_mask = np.all(original_array == target_color, axis=-1)
    else:
        highlight_mask = np.all(original_array[..., :3] == target_color[:3], axis=-1)

    # Create display image
    img_array = original_array.copy()

    # Apply diagonal pattern to background colors
    for group in data['groups']:
        if group.get('is_conductive', False):
            bg_color = group['color']

            if num_channels == 4 and len(bg_color) == 4:
                bg_mask = np.all(original_array == bg_color, axis=-1)
            else:
                bg_mask = np.all(original_array[..., :3] == bg_color[:3], axis=-1)

            pattern = create_diagonal_pattern(img_array.shape[0], img_array.shape[1])
            if num_channels == 4:
                img_array[bg_mask, :3] = pattern[bg_mask]
                img_array[bg_mask, 3] = 255  # Make fully opaque
            else:
                img_array[bg_mask] = pattern[bg_mask]

    # Now apply highlight dimming
    img_array = img_array.astype(np.float32)

    # Dim non-selected pixels
    img_array[~highlight_mask] = img_array[~highlight_mask] * 0.3
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)

    # Marching ants border is drawn client-side via canvas for animation

    img = Image.fromarray(img_array)
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)

    return send_file(buffer, mimetype='image/png')


@app.route('/api/dimensions', methods=['POST'])
def api_dimensions():
    """Update tissue dimensions."""
    data = get_builder_session()
    if data is None:
        return jsonify({'error': 'No session'}), 404

    req = request.json
    width = req.get('width', 10.0)
    height = req.get('height', 10.0)
    dx = req.get('dx', 0.01)

    # Update the builder session
    builder = data['builder']
    if builder:
        builder.set_dimensions(width, height, dx)

    return jsonify({'success': True})


def find_contours_numpy(mask):
    """Find boundary contours using Moore neighborhood tracing (no cv2 required)."""
    h, w = mask.shape
    if h == 0 or w == 0:
        return []

    # Pad mask to handle edges safely
    padded = np.pad(mask.astype(np.uint8), 1, mode='constant', constant_values=0)

    # Find boundary pixels: pixels that are True but have at least one False neighbor (8-connectivity)
    boundary = np.zeros_like(mask, dtype=bool)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            shifted = padded[1+dy:h+1+dy, 1+dx:w+1+dx]
            boundary |= (mask & (shifted == 0))

    # Get all boundary pixels as a set for fast lookup
    boundary_coords = set(map(tuple, np.argwhere(boundary)))  # (y, x) tuples

    if len(boundary_coords) == 0:
        return []

    # Moore neighborhood: 8 directions clockwise starting from right
    # (dx, dy) pairs
    directions = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

    def trace_contour(start, remaining):
        """Trace a single contour starting from a boundary pixel."""
        contour = []
        current = start
        remaining.discard(current)

        # Find initial direction (first background neighbor)
        y, x = current
        start_dir = 0
        for i, (dx, dy) in enumerate(directions):
            if padded[y + 1 + dy, x + 1 + dx] == 0:
                start_dir = i
                break

        contour.append([int(x), int(y)])  # Store as [x, y], convert to Python int
        prev_dir = start_dir

        # Trace the contour
        max_steps = len(boundary_coords) * 4 + 100  # Safety limit
        for _ in range(max_steps):
            y, x = current

            # Search for next boundary pixel, starting from direction after where we came from
            # (backtrack direction + 1, going clockwise)
            search_start = (prev_dir + 5) % 8  # Opposite direction + 1

            found = False
            for i in range(8):
                dir_idx = (search_start + i) % 8
                dx, dy = directions[dir_idx]
                ny, nx = y + dy, x + dx

                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx]:
                    # Check if this is a boundary pixel
                    if (ny, nx) in remaining or (ny, nx) == start:
                        if (ny, nx) == start and len(contour) > 2:
                            # Completed the loop
                            return contour
                        if (ny, nx) != start:
                            current = (ny, nx)
                            remaining.discard(current)
                            contour.append([int(nx), int(ny)])  # Convert to Python int
                            prev_dir = dir_idx
                            found = True
                            break

            if not found:
                break

        return contour

    # Find all contours
    contours = []
    remaining = boundary_coords.copy()

    while remaining:
        start = min(remaining)  # Start from top-left most point
        contour = trace_contour(start, remaining)
        if len(contour) > 10:  # Skip tiny contours
            contours.append(contour)

    return contours


@app.route('/api/boundary/<int:group_index>')
def api_boundary(group_index):
    """Get ordered boundary contours of a group for marching ants animation."""
    data = get_builder_session()
    if data is None or data['builder'] is None:
        return jsonify({'error': 'No session'}), 404

    # Find the group by index
    target_group = None
    for group in data['groups']:
        if group['index'] == group_index:
            target_group = group
            break

    if target_group is None:
        return jsonify({'error': 'Group not found'}), 404

    # Get original image
    original_array = data['builder'].image_array
    target_color = target_group['color']

    # Create mask for the selected color
    num_channels = original_array.shape[-1] if len(original_array.shape) == 3 else 1
    if num_channels == 4 and len(target_color) == 4:
        mask = np.all(original_array == target_color, axis=-1)
    else:
        mask = np.all(original_array[..., :3] == target_color[:3], axis=-1)

    # Find contours using pure numpy
    paths = find_contours_numpy(mask)

    return jsonify({
        'contours': paths,
        'image_size': [original_array.shape[1], original_array.shape[0]]  # [width, height]
    })


@app.route('/api/pixel/<int:x>/<int:y>')
def api_pixel(x, y):
    """Get which group a pixel belongs to."""
    data = get_builder_session()
    if data is None or data['builder'] is None:
        return jsonify({'error': 'No session'}), 404

    original_array = data['builder'].image_array
    h, w = original_array.shape[:2]

    # Bounds check
    if x < 0 or x >= w or y < 0 or y >= h:
        return jsonify({'error': 'Out of bounds'}), 400

    # Get pixel color at position (convert numpy types to Python int)
    pixel_color = [int(c) for c in original_array[y, x]]

    # Find matching group
    for group in data['groups']:
        if group['color'] == pixel_color:
            return jsonify({
                'group_index': group['index'],
                'color': pixel_color,
                'label': group['label']
            })

    return jsonify({'group_index': None, 'color': pixel_color})


@app.route('/api/export', methods=['POST'])
def api_export():
    """Export the configured mesh to a .npz file."""
    data = get_builder_session()
    if data is None or data['builder'] is None:
        return jsonify({'error': 'No session'}), 404

    builder = data['builder']

    # Ensure all conductive groups have conductivity configured on the builder.
    # The UI marks conductive groups as "background" for display but may not
    # have called configure_group() with D values. Do it now from the presets.
    for group_info in data['groups']:
        if not group_info.get('configured', False):
            return jsonify({'error': f"Group '{group_info['label']}' not configured"}), 400

        color = tuple(group_info['color'])
        preset_key = group_info.get('preset')

        if group_info.get('is_conductive') and preset_key in CONDUCTIVE_PRESETS:
            preset = CONDUCTIVE_PRESETS[preset_key]
            # Ensure the builder has this group configured with conductivity
            # (mark_as_background doesn't set conductivity)
            builder.mark_as_tissue(color)
            builder.configure_group(
                color=color,
                label=preset['label'],
                cell_type=preset['cell_type'],
                D_xx=preset['D_xx'],
                D_yy=preset['D_yy'],
                D_xy=preset['D_xy'],
            )

    try:
        # Export to temp file, then send as download
        with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as tmp:
            tmp_path = tmp.name

        export_mesh(builder, tmp_path)

        return send_file(
            tmp_path,
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name='mesh.npz',
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/reset', methods=['POST'])
def api_reset():
    """Reset current session."""
    sid = get_session_id()
    if sid in sessions:
        del sessions[sid]
    return jsonify({'success': True})


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.DEBUG)
    print("Starting Builder UI at http://localhost:5001")
    app.run(debug=True, port=5001, host='0.0.0.0')

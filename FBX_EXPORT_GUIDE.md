# FBX Export Guide

This guide explains how to export generated motion to FBX files using the standalone wooden boy exporter.

## Overview

The FBX export functionality has been extracted from the ComfyUI nodes and made available as a standalone module. It supports:
- **Wooden Boy Export**: Export to the default wooden boy character template
- **Custom FBX Retargeting**: Retarget motion to any custom FBX skeleton (e.g., Mixamo characters)
- **Gradio UI Integration**: Upload custom FBX directly in the web interface
- **Batch Processing**: Export multiple motions at once
- **NPZ to FBX Conversion**: Convert saved motion NPZ files to FBX

## Files

### 1. `hymotion/utils/fbx_export.py`
Standalone FBX export module that works independently of ComfyUI.

**Key Functions:**
- `export_motion_to_fbx(motion_data, output_path, text_description, template_fbx)` - Export single motion
- `export_motion_batch_to_fbx(motion_data_list, output_dir, ...)` - Export multiple motions
- `load_motion_from_npz(npz_path)` - Load motion from NPZ file

### 2. `test_fbx_export.py`
Test script to verify FBX export functionality.

### 3. Updated `gradio_app.py`
Integrated FBX export function: `export_wooden_boy_fbx(smpl_data_list, output_dir, text)`

## Usage Examples

### 1. Command Line (Direct Export)

Export an NPZ file to FBX (wooden boy):

```bash
python hymotion/utils/fbx_export.py --input output/gradio/motion_xxx_000.npz --output output/test.fbx
```

Export with custom FBX template (retargeting):

```bash
python hymotion/utils/fbx_export.py --input output/gradio/motion_xxx_000.npz --output output/custom.fbx --template path/to/custom_character.fbx
```

### 2. Test Script (Single File)

Test export with a motion NPZ file:

```bash
python test_fbx_export.py --motion_npz output/gradio/motion_xxx_000.npz --output_dir output/test_fbx
```

### 3. Test Script (Batch Mode)

Export all NPZ files in a directory:

```bash
python test_fbx_export.py --motion_npz_dir output/gradio --output_dir output/test_fbx
```

### 4. Python API

Use in your Python code:

```pythonwooden boy FBX
success = export_motion_to_fbx(
    motion_data=motion_data,
    output_path="output/my_animation.fbx",
    text_description="A person walking forward"
)

# Export to custom FBX character (with retargeting)
success = export_motion_to_fbx(
    motion_data=motion_data,
    output_path="output/custom_animation.fbx",
    text_description="A person walking forward",
    template_fbx="path/to/mixamo_character.fbxdio/motion_xxx_000.npz")

# Export to FBX
success = export_motion_to_fbx(
    motion_data=motion_data,
    output_path="output/my_animation.fbx",
    text_description="A person walking forward"
)
```

### 5. From Gradio App

The function `export_wooden_boy_fbx()` is integrated into gradio_app.py and can be called after motion generation:

```python
from gradio_app import export_wooden_boy_fbx

# smpl_data_list is the motion output from generate_motion
fbx_files = export_wooden_boy_fbx(
    smpl_data_list=smpl_data_list,
    output_dir="output/fbx",
    text="A person walking"
)
```

## Input Data Format

The motion data should be a dictionary containing:

**Required Keys:**
- `rot6d`: Rotation data in 6D format, shape `(num_frames, 22, 6)`
- `transl`: Translation data, shape `(num_frames, 3)`

**Optional Keys:**
- `text`: Text description of the motion
- `duration`: Duration in seconds
- `seed`: Random seed used for generation
- `keypoints3d`: 3D keypoints (not used for FBX export)
- `root_rotations_mat`: Root rotation matrices (not used for FBX export)

## Workflow Integration

### Workflow 1: Gradio Web Interface (Easiest)

1. **Start Gradio App**
   ```bash
   python gradio_app.py
   ```

2. **Generate Motion with Custom Character**
   - Enter text prompt
   - Upload custom FBX (optional)
   - Click Generate
   - Download FBX files

3. **Import into 3D Software** (Blender, Maya, Unreal Engine, etc.)

### Workflow 2: Command Line

1. **Generate Motion** (using local_infer.py)
   ```bash
   python local_infer.py --text "a person walks forward" --seeds 42
   ```

2. **Motion is Saved as NPZ** (in `output/gradio/` or specified directory)

3. **Export to FBX** (wooden boy or custom)
   ```bash
   # Wooden boy
   python hymotion/utils/fbx_export.py --input output/gradio/motion_xxx_000.npz --output output/result.fbx
   
   # Custom character
   python hymotion/utils/fbx_export.py --input output/gradio/motion_xxx_000.npz --output output/custom.fbx --template path/to/character.fbx
   ```

4. **Import FBX into 3D Software** (Blender, Maya, Unreal Engine, etc.)

## Troubleshooting

### Issue: "FBX SDK not installed"
**Solution:** Install the FBX Python SDK:
- Download from Autodesk's website
- Or install via pip if available: `pip install fbx`

### Issue: "Template FBX not found"
**Solution:** Ensure the wooden boy template exists at:
```
assets/wooden_models/boy_Rigging_smplx_tex.fbx
```
Custom FBX Retargeting

### Method 1: Gradio Web Interface (Recommended)

1. Generate motion using text prompt
2. Upload your custom FBX character (e.g., from Mixamo) in the "Custom FBX Skeleton" section
3. Click Generate - the motion will be automatically retargeted to your character
4. Download both wooden boy and retargeted FBX files

### Method 2: Command Line

Use the `fbx_export.py` module with `--template` parameter:

```bash
python hymotion/utils/fbx_export.py \
    --input output/gradio/motion_xxx_000.npz \
    --output output/retargeted.fbx \
    --template path/to/custom_character.fbx
```

### Method 3: Standalone Retargeting Tool

For advanced control, use `retarget_fbx.py` directly
```bash
cd /path/to/HY-Motion-1.0
python test_fbx_export.py ...
``**Wooden boy export** uses the SMPL-H to wooden FBX converter (`smplh2woodfbx.py`)
- **Custom FBX export** uses automatic retargeting via `retarget_fbx.py`
- Output FBX files include:
  - `.fbx` - The animation file
  - `.npz` - Motion data in rot6d format (for retargeting)
  - `.txt` - Text description (if provided)
- Frame rate is fixed at 30 FPS
- The wooden boy template is a simple wooden mannequin suitable for animation preview
- Custom FBX characters (e.g., Mixamo) are automatically retargeted using bone mapping
- Retargeting supports automatic height scaling and orientation adjustment
```bash
python hymotion/utils/retarget_fbx.py \
    --source output/gradio/motion_xxx_000.npz \
    --target input/custom_character.fbx \
    --output output/retargeted.fbx \
    --yaw 0.0 \
    --scale 0.0
```

## Notes
 with custom FBX support:

### Basic Workflow:
1. Enter your text prompt (e.g., "a person walks forward")
2. Click "Generate Motion" button
3. Wooden boy FBX is automatically generated
4. Download files from the interface

### Custom Character Workflow:
### From Gradio (with custom FBX):
```
output/gradio/
├── 20260112_143052_a7f3b1d2_000.fbx              # Wooden boy animation
├── 20260112_143052_a7f3b1d2_000.npz              # Motion data (rot6d format)
├── 20260112_143052_a7f3b1d2_000.txt              # Text description
├── retargeted_20260112_143052_a7f3b1d2_000.fbx   # Custom character animation
├── retargeted_20260112_143052_a7f3b1d2_000.txt   # Text description
└── ...
```

### From test script
2. (Optional) Open "Custom FBX Skeleton" accordion
3. Upload your custom FBX character (e.g., Mixamo character)
4. Click "Generate Motion" button
5. System generates:
   - Wooden boy FBX (for preview)
   - Retargeted FBX (your custom character with the motion)
   - NPZ files (for both formats)
6. Download all files from the interface

### Files Generated:
- `timestamp_id_000.fbx` - Wooden boy animation
- `timestamp_id_000.npz` - Motion data (rot6d format)
- `timestamp_id_000.txt` - Text description
- `timestamp_id_000.glb` - GLB format for web preview
- `retargeted_timestamp_id_000.fbx` - Custom character animation (if custom FBX uploaded)
- `retargeted_timestamp_id_000.glb` - Custom character GLB for web preview
- `retargeted_timestamp_id_000.txt` - Text description for retargeted animation

All files are saved to `output/gradio/` directory

## Example Filenames

After running the test script, you'll get files like:
```
output/test_fbx/
├── motion_20260111_143052_a7f3b1d2_000_export.fbx
├── motion_20260111_143052_a7f3b1d2_000_export.txt
└── ...
```

## 3D Preview Feature

When you upload a custom FBX character in the Gradio interface, an interactive 3D preview is automatically generated showing all characters side-by-side:

### Features:
- **Multiple Characters**: Wooden boy and retargeted characters displayed together
- **1 Unit Spacing**: Each character positioned exactly 1.0 unit apart on X-axis
- **Interactive Controls**:
  - Left-click drag: Rotate camera
  - Right-click drag: Pan camera
  - Mouse wheel: Zoom in/out
  - Play/Pause buttons: Control all animations
  - Reset Camera button: Return to default view
- **Real-time Animation**: All characters animate simultaneously
- **Character Labels**: Each model labeled with its filename

### Testing 3D Preview:
```bash
python test_3d_preview.py
```

### Requirements for 3D Preview:
```bash
pip install 'trimesh[easy]'
```

Or install Blender for higher quality conversion (auto-detected if available).

## Integration with Gradio

The FBX export is automatically available in the Gradio interface with 3D preview support:
1. Generate motion using text prompt
2. (Optional) Upload custom FBX character
3. Motion is generated and saved to `output/gradio/`
4. If custom FBX uploaded: retargeting happens automatically + 3D preview generated
5. Download FBX files and view 3D preview in the interface

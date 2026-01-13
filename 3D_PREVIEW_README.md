# 3D Preview Feature

## Overview

The 3D preview feature automatically generates an interactive web-based viewer when you upload custom FBX characters in the Gradio interface. All generated FBX files (wooden boy + retargeted characters) are displayed side-by-side with 1 unit spacing.

## Features

✨ **Multiple Characters**: View wooden boy and all retargeted characters together  
📏 **1 Unit Spacing**: Each character positioned exactly 1.0 unit apart on X-axis  
🎮 **Interactive Controls**: Rotate, pan, zoom with mouse  
▶️ **Animation Controls**: Play/pause all animations simultaneously  
🏷️ **Character Labels**: Each model labeled with its filename  
🌐 **Web-based**: Runs in browser using Three.js (no installation needed)

## How It Works

### Workflow:

1. **Generate Motion** in Gradio with text prompt
2. **Upload Custom FBX** character (optional)
3. **System Processes**:
   - Generates wooden boy FBX
   - Retargets motion to custom FBX
   - Converts all FBX files to GLB format
   - Creates 3D preview HTML with Three.js
4. **View Results**: Interactive 3D viewer shows all characters

### Technical Details:

- **Conversion**: FBX → GLB (using trimesh or Blender)
- **Renderer**: Three.js with OrbitControls
- **Spacing**: `character.position.x = index * 1.0`
- **Animation**: GLTF animations played via Three.js AnimationMixer
- **Lighting**: Ambient + 2 directional lights with shadows
- **Ground**: Grid helper for reference

## Installation

### Required:
```bash
pip install 'trimesh[easy]'
```

### Optional (for better quality):
Install Blender - will be auto-detected if available:
- Download from https://www.blender.org/
- Blender provides higher quality FBX to GLB conversion

## Controls

### Mouse:
- **Left-click + drag**: Rotate camera around scene
- **Right-click + drag**: Pan camera position
- **Scroll wheel**: Zoom in/out

### Buttons:
- **▶ Play All**: Resume all animations
- **⏸ Pause All**: Pause all animations
- **🎥 Reset Camera**: Return to default camera position

## File Structure

```
output/gradio/
├── 20260112_143052_a7f3b1d2_000.fbx              # Wooden boy FBX
├── 20260112_143052_a7f3b1d2_000.glb              # Wooden boy GLB (for preview)
├── 20260112_143052_a7f3b1d2_000.npz              # Motion data
├── retargeted_20260112_143052_a7f3b1d2_000.fbx   # Custom character FBX
└── retargeted_20260112_143052_a7f3b1d2_000.glb   # Custom character GLB (for preview)
```

## Testing

Test the conversion without using Gradio:

```bash
python test_3d_preview.py
```

This will:
1. Find FBX files in `output/gradio/`
2. Convert them to GLB
3. Report success/failure

## Troubleshooting

### "Failed to convert FBX to GLB"
**Solution:**
```bash
pip install 'trimesh[easy]'
```
Or install Blender for more robust conversion.

### "No preview shown"
**Possible causes:**
1. FBX files are corrupted
2. trimesh not installed
3. Blender not found (if relying on it)

**Solution:**
- Check console for error messages
- Verify FBX files can be opened in 3D software
- Install trimesh: `pip install 'trimesh[easy]'`

### "Characters not spacing correctly"
The spacing is hardcoded to 1.0 unit on X-axis. If characters overlap:
- Check character scales in original FBX files
- Larger characters may need more spacing (modify `x_offset` in code)

## Code Reference

### Main Components:

1. **fbx_to_glb.py**: Converts FBX files to web-friendly GLB format
   - Uses trimesh (fallback) or Blender (preferred)
   - Preserves animations and skeletal structure

2. **gradio_app.py**: 
   - `_generate_3d_preview()`: Orchestrates GLB conversion
   - `_create_multi_model_viewer_html()`: Generates Three.js viewer
   - Handles base64 encoding of GLB files for embedding

3. **test_3d_preview.py**: Test script for conversion

### Customization:

To modify spacing, edit in `gradio_app.py`:
```python
'x_offset': i * 1.0  # Change 1.0 to desired spacing
```

To modify camera position, edit in HTML template:
```javascript
camera.position.set(glbDataList.length * 0.5, 2, 5);
```

## Performance

- **Small models** (< 1MB FBX): Near-instant conversion
- **Large models** (> 10MB FBX): May take 5-30 seconds
- **Browser rendering**: Smooth on modern GPUs
- **Multiple characters**: 3-5 characters recommended for best performance

## Browser Compatibility

✅ Chrome/Edge (recommended)  
✅ Firefox  
✅ Safari  
⚠️ Older browsers may have limited Three.js support

## Future Enhancements

Potential improvements:
- [ ] Adjustable spacing via UI slider
- [ ] Individual character play/pause controls
- [ ] Timeline scrubbing for animation control
- [ ] Export preview as video
- [ ] Side-by-side comparison mode
- [ ] Custom camera angles (front/side/top views)

## Credits

- **Three.js**: 3D rendering library
- **trimesh**: Python library for 3D meshes
- **Blender**: Open source 3D creation suite (optional)

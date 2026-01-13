"""
Convert FBX to GLB format for web viewing
"""
import os
import sys
import subprocess
import tempfile
from typing import List, Optional

def fbx_to_glb(fbx_path: str, output_path: Optional[str] = None) -> Optional[str]:
    """
    Convert FBX file to GLB format using Blender (if available).
    
    Args:
        fbx_path: Path to input FBX file
        output_path: Path to output GLB file (optional, auto-generated if None)
    
    Returns:
        Path to GLB file if successful, None otherwise
    """
    if not os.path.exists(fbx_path):
        print(f"FBX file not found: {fbx_path}")
        return None
    
    if output_path is None:
        output_path = fbx_path.replace('.fbx', '.glb')
    
    # Try using Blender for conversion
    try:
        import bpy
        return _fbx_to_glb_blender_api(fbx_path, output_path)
    except ImportError:
        # Try using Blender as subprocess
        return _fbx_to_glb_blender_subprocess(fbx_path, output_path)


def _fbx_to_glb_blender_api(fbx_path: str, output_path: str) -> Optional[str]:
    """Convert using Blender Python API (if running inside Blender)."""
    try:
        import bpy
        
        # Clear scene
        bpy.ops.wm.read_factory_settings(use_empty=True)
        
        # Import FBX
        bpy.ops.import_scene.fbx(filepath=fbx_path)
        
        # Export as GLB
        bpy.ops.export_scene.gltf(
            filepath=output_path,
            export_format='GLB',
            export_animations=True
        )
        
        print(f"Converted {fbx_path} to {output_path}")
        return output_path
    except Exception as e:
        print(f"Blender API conversion failed: {e}")
        return None


def _fbx_to_glb_blender_subprocess(fbx_path: str, output_path: str) -> Optional[str]:
    """Convert using Blender as subprocess."""
    # Try to find Blender executable
    blender_paths = [
        "blender",  # In PATH
        "C:/Program Files/Blender Foundation/Blender/blender.exe",
        "C:/Program Files/Blender Foundation/Blender 3.6/blender.exe",
        "C:/Program Files/Blender Foundation/Blender 4.0/blender.exe",
        "/usr/bin/blender",
        "/Applications/Blender.app/Contents/MacOS/Blender",
    ]
    
    blender_exe = None
    for path in blender_paths:
        if os.path.exists(path) or path == "blender":
            blender_exe = path
            break
    
    if not blender_exe:
        print("Blender not found. Install Blender for FBX to GLB conversion.")
        print("Or install: pip install trimesh[easy]")
        return _fbx_to_glb_trimesh(fbx_path, output_path)
    
    # Create temporary Python script for Blender
    script = f"""
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=r"{fbx_path}")
bpy.ops.export_scene.gltf(filepath=r"{output_path}", export_format='GLB', export_animations=True)
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        script_path = f.name
    
    try:
        # Run Blender in background
        result = subprocess.run(
            [blender_exe, "--background", "--python", script_path],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0 and os.path.exists(output_path):
            print(f"Converted {fbx_path} to {output_path}")
            return output_path
        else:
            print(f"Blender conversion failed: {result.stderr}")
            return _fbx_to_glb_trimesh(fbx_path, output_path)
    except Exception as e:
        print(f"Blender subprocess failed: {e}")
        return _fbx_to_glb_trimesh(fbx_path, output_path)
    finally:
        try:
            os.unlink(script_path)
        except:
            pass


def _fbx_to_glb_trimesh(fbx_path: str, output_path: str) -> Optional[str]:
    """Convert using trimesh library (fallback)."""
    try:
        import trimesh
        
        # Try loading with pyglet/assimp backend
        try:
            # Load FBX - trimesh needs assimp for FBX support
            scene = trimesh.load(fbx_path, force='scene', process=True)
        except Exception as e:
            print(f"Trimesh FBX loading failed: {e}")
            print("Note: FBX support in trimesh requires pyassimp or other backends")
            return None
        
        # Export as GLB
        scene.export(output_path, file_type='glb')
        
        print(f"Converted {fbx_path} to {output_path} (using trimesh)")
        return output_path
    except ImportError:
        print("ERROR: trimesh not installed. Install with: pip install trimesh[easy]")
        return None
    except Exception as e:
        print(f"Trimesh conversion failed: {e}")
        print("Tip: Install pyassimp for FBX support: pip install pyassimp")
        return None


def convert_fbx_batch(fbx_files: List[str]) -> List[str]:
    """
    Convert multiple FBX files to GLB format.
    
    Args:
        fbx_files: List of FBX file paths
    
    Returns:
        List of GLB file paths (only successful conversions)
    """
    glb_files = []
    for fbx_file in fbx_files:
        if not fbx_file.endswith('.fbx'):
            continue
        
        glb_file = fbx_to_glb(fbx_file)
        if glb_file:
            glb_files.append(glb_file)
    
    return glb_files


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python fbx_to_glb.py <input.fbx> [output.glb]")
        sys.exit(1)
    
    fbx_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = fbx_to_glb(fbx_path, output_path)
    
    if result:
        print(f"Success: {result}")
        sys.exit(0)
    else:
        print("Conversion failed")
        sys.exit(1)

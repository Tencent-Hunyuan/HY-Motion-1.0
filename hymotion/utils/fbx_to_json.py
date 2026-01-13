"""
Alternative FBX to GLB converter using FBX SDK directly
This converts FBX to a simpler format that can be viewed in the browser
"""
import os
import numpy as np
import json
from typing import Optional

def fbx_to_json_animation(fbx_path: str, output_path: Optional[str] = None) -> Optional[str]:
    """
    Extract animation data from FBX and save as JSON for web viewing.
    This is a fallback when GLB conversion is not available.
    
    Args:
        fbx_path: Path to input FBX file
        output_path: Path to output JSON file (optional)
    
    Returns:
        Path to JSON file if successful, None otherwise
    """
    try:
        import fbx
        from fbx import FbxManager, FbxIOSettings, FbxImporter, FbxScene
    except ImportError:
        print("FBX SDK not available")
        return None
    
    if not os.path.exists(fbx_path):
        return None
    
    if output_path is None:
        output_path = fbx_path.replace('.fbx', '.json')
    
    # Initialize FBX SDK
    manager = FbxManager.Create()
    ios = FbxIOSettings.Create(manager, "IOSRoot")
    manager.SetIOSettings(ios)
    
    # Create scene
    scene = FbxScene.Create(manager, "Scene")
    
    # Import FBX
    importer = FbxImporter.Create(manager, "")
    if not importer.Initialize(fbx_path, -1, manager.GetIOSettings()):
        print(f"Failed to initialize FBX importer: {importer.GetStatus().GetErrorString()}")
        manager.Destroy()
        return None
    
    if not importer.Import(scene):
        print(f"Failed to import FBX: {importer.GetStatus().GetErrorString()}")
        importer.Destroy()
        manager.Destroy()
        return None
    
    importer.Destroy()
    
    # Extract animation data
    animation_data = extract_animation_from_scene(scene)
    
    if animation_data:
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(animation_data, f)
        
        manager.Destroy()
        return output_path
    
    manager.Destroy()
    return None


def extract_animation_from_scene(scene):
    """Extract skeleton and animation data from FBX scene."""
    # This is a simplified version - full implementation would need more work
    # For now, we'll just note that this could be expanded
    return None


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = fbx_to_json_animation(sys.argv[1])
        if result:
            print(f"Converted to: {result}")
        else:
            print("Conversion failed")

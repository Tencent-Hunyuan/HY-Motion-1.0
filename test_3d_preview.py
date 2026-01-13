"""
Test script for FBX to GLB conversion and 3D preview
"""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hymotion.utils.fbx_to_glb import fbx_to_glb, convert_fbx_batch

def test_single_conversion():
    """Test converting a single FBX file"""
    # Find a test FBX file in output/gradio
    output_dir = "output/gradio"
    
    if not os.path.exists(output_dir):
        print(f"Output directory not found: {output_dir}")
        print("Generate some motion first using gradio_app.py")
        return False
    
    # Find the first FBX file
    fbx_files = [f for f in os.listdir(output_dir) if f.endswith('.fbx')]
    
    if not fbx_files:
        print(f"No FBX files found in {output_dir}")
        print("Generate some motion first using gradio_app.py")
        return False
    
    test_fbx = os.path.join(output_dir, fbx_files[0])
    print(f"Testing conversion: {test_fbx}")
    
    # Convert to GLB
    glb_file = fbx_to_glb(test_fbx)
    
    if glb_file and os.path.exists(glb_file):
        print(f"✓ Success: {glb_file}")
        print(f"  Size: {os.path.getsize(glb_file) / 1024:.1f} KB")
        return True
    else:
        print("✗ Conversion failed")
        return False


def test_batch_conversion():
    """Test batch conversion of multiple FBX files"""
    output_dir = "output/gradio"
    
    if not os.path.exists(output_dir):
        print(f"Output directory not found: {output_dir}")
        return False
    
    # Find all FBX files
    fbx_files = [os.path.join(output_dir, f) 
                 for f in os.listdir(output_dir) 
                 if f.endswith('.fbx')]
    
    if not fbx_files:
        print(f"No FBX files found in {output_dir}")
        return False
    
    # Take first 3 files for testing
    test_files = fbx_files[:3]
    print(f"\nTesting batch conversion of {len(test_files)} files:")
    for f in test_files:
        print(f"  - {os.path.basename(f)}")
    
    # Convert batch
    glb_files = convert_fbx_batch(test_files)
    
    print(f"\n✓ Successfully converted {len(glb_files)}/{len(test_files)} files")
    for glb in glb_files:
        print(f"  - {os.path.basename(glb)} ({os.path.getsize(glb) / 1024:.1f} KB)")
    
    return len(glb_files) > 0


if __name__ == "__main__":
    print("=" * 60)
    print("FBX to GLB Conversion Test")
    print("=" * 60)
    
    print("\n1. Testing single file conversion...")
    success1 = test_single_conversion()
    
    print("\n2. Testing batch conversion...")
    success2 = test_batch_conversion()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("✓ All tests passed!")
        print("\nNote: If you upload a custom FBX in the Gradio interface,")
        print("you'll see a 3D preview showing all characters side-by-side")
        print("with 1 unit spacing on the X-axis.")
    else:
        print("✗ Some tests failed")
        print("\nMake sure you have:")
        print("  1. Generated some motion using gradio_app.py")
        print("  2. Installed trimesh: pip install 'trimesh[easy]'")
        print("  3. Or install Blender for better quality conversion")
    print("=" * 60)

"""
Standalone FBX Export Module
Exports motion data to FBX files (wooden boy template).
Works independently of ComfyUI.
"""

import os
import sys

# Add parent directory to path for direct script execution
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import time
import uuid
import numpy as np
import torch
from typing import Dict, Any, Optional

# Global FBX converter cache
_fbx_converter = None
_fbx_converter_path = None


def get_timestamp():
    """Generate timestamp string"""
    t = time.time()
    ms = int((t - int(t)) * 1000)
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(t)) + f"{ms:03d}"


def export_motion_to_fbx(
    motion_data: Dict[str, Any],
    output_path: str,
    text_description: str = "",
    template_fbx: Optional[str] = None
) -> bool:
    """
    Export motion data to FBX file.
    Supports wooden boy template or custom FBX skeletons (with retargeting).
    
    Args:
        motion_data: Dict containing 'rot6d' and 'transl' (numpy arrays or torch tensors)
                    rot6d shape: (num_frames, 22, 6)
                    transl shape: (num_frames, 3)
        output_path: Path to save output FBX file
        text_description: Optional text description to save alongside FBX
        template_fbx: Path to template FBX (default: wooden boy, custom: use retargeting)
        
    Returns:
        Success status
    """
    # Determine if using custom FBX or wooden boy
    current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_template = os.path.join(current_dir, "assets", "wooden_models", "boy_Rigging_smplx_tex.fbx")
    
    is_custom_fbx = (template_fbx is not None and 
                     os.path.exists(template_fbx) and 
                     not template_fbx.endswith("boy_Rigging_smplx_tex.fbx"))
    
    if is_custom_fbx:
        return _export_with_retargeting(motion_data, output_path, template_fbx, text_description)
    else:
        return _export_wooden_boy(motion_data, output_path, text_description)


def _export_wooden_boy(
    motion_data: Dict[str, Any],
    output_path: str,
    text_description: str = ""
) -> bool:
    """Export using wooden boy template."""
    global _fbx_converter, _fbx_converter_path
    
    try:
        # Import required modules
        try:
            from hymotion.pipeline.body_model import construct_smpl_data_dict
        except ImportError:
            try:
                from ..pipeline.body_model import construct_smpl_data_dict
            except ImportError:
                from body_model import construct_smpl_data_dict
        
        # Use wooden boy template
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        template_fbx = os.path.join(current_dir, "assets", "wooden_models", "boy_Rigging_smplx_tex.fbx")
        
        if not os.path.exists(template_fbx):
            print(f"ERROR: Template FBX not found: {template_fbx}")
            return False
        
        # Lazy load FBX converter
        if _fbx_converter is None or _fbx_converter_path != template_fbx:
            try:
                import fbx
                try:
                    from hymotion.utils.smplh2woodfbx import SMPLH2WoodFBX
                except ImportError:
                    from .smplh2woodfbx import SMPLH2WoodFBX
                    
                _fbx_converter = SMPLH2WoodFBX(template_fbx_path=template_fbx)
                _fbx_converter_path = template_fbx
                print(f"[FBX Export] Loaded FBX converter with template: {template_fbx}")
            except ImportError:
                print("ERROR: FBX SDK not installed")
                return False
            except Exception as e:
                print(f"ERROR: Failed to load FBX converter: {e}")
                return False
        
        # Extract and convert motion data
        rot6d = motion_data.get('rot6d')
        transl = motion_data.get('transl')
        
        if rot6d is None or transl is None:
            print("ERROR: motion_data must contain 'rot6d' and 'transl'")
            return False
        
        # Convert to torch tensors if needed
        if isinstance(rot6d, np.ndarray):
            rot6d = torch.from_numpy(rot6d).float()
        if isinstance(transl, np.ndarray):
            transl = torch.from_numpy(transl).float()
        
        # Ensure CPU tensors
        rot6d = rot6d.cpu() if hasattr(rot6d, 'cpu') else rot6d
        transl = transl.cpu() if hasattr(transl, 'cpu') else transl
        
        # Construct SMPL data dict
        smpl_data = construct_smpl_data_dict(rot6d, transl)
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Convert and save
        print(f"[FBX Export] Exporting to: {output_path}")
        success = _fbx_converter.convert_npz_to_fbx(smpl_data, output_path)
        
        if success:
            print(f"[FBX Export] Successfully exported FBX: {output_path}")
            
            # Save text description if provided
            if success and text_description:
                txt_path = output_path.replace(".fbx", ".txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text_description)
        
        return success
        
    except Exception as e:
        print(f"ERROR: FBX export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def _export_with_retargeting(
    motion_data: Dict[str, Any],
    output_path: str,
    template_fbx: str,
    text_description: str = "",
    yaw_offset: float = 0.0,
    scale: float = 0.0
) -> bool:
    """Export using retargeting to custom FBX skeleton."""
    try:
        # Import required modules
        try:
            from hymotion.pipeline.body_model import construct_smpl_data_dict
        except ImportError:
            try:
                from ..pipeline.body_model import construct_smpl_data_dict
            except ImportError:
                from body_model import construct_smpl_data_dict
        
        try:
            from hymotion.utils.retarget_fbx import (
                load_npz, load_fbx, load_bone_mapping, retarget_animation,
                apply_retargeted_animation, save_fbx, HAS_FBX_SDK
            )
        except ImportError:
            try:
                from retarget_fbx import (
                    load_npz, load_fbx, load_bone_mapping, retarget_animation,
                    apply_retargeted_animation, save_fbx, HAS_FBX_SDK
                )
            except ImportError:
                print("ERROR: retarget_fbx module not found")
                return False
        
        if not HAS_FBX_SDK:
            print("ERROR: FBX SDK not installed")
            return False
        
        # Extract and convert motion data
        rot6d = motion_data.get('rot6d')
        transl = motion_data.get('transl')
        
        # Check which format we have
        has_rot6d = 'rot6d' in motion_data and 'transl' in motion_data
        has_smplh = 'poses' in motion_data and 'trans' in motion_data
        
        if not has_rot6d and not has_smplh:
            print(f"ERROR: motion_data must contain either ('rot6d' + 'transl') or ('poses' + 'trans')")
            print(f"       Found keys: {list(motion_data.keys())}")
            return False
        
        # Create temporary NPZ file
        import tempfile
        temp_dir = os.path.dirname(output_path)
        temp_npz = os.path.join(temp_dir, f"_temp_{uuid.uuid4().hex[:8]}.npz")
        
        # Prepare data dict with all required fields
        data_dict = {}
        
        if has_rot6d:
            # Convert from rot6d format to SMPL-H format
            rot6d = motion_data['rot6d']
            transl = motion_data['transl']
            
            # Convert to torch tensors if needed
            if isinstance(rot6d, np.ndarray):
                rot6d = torch.from_numpy(rot6d).float()
            if isinstance(transl, np.ndarray):
                transl = torch.from_numpy(transl).float()
            
            # Add basic motion data
            if hasattr(rot6d, 'cpu'):
                data_dict['rot6d'] = rot6d.cpu().numpy()
            else:
                data_dict['rot6d'] = np.array(rot6d)
                
            if hasattr(transl, 'cpu'):
                data_dict['transl'] = transl.cpu().numpy()
            else:
                data_dict['transl'] = np.array(transl)
            
            # Add other fields if present
            for key in ['keypoints3d', 'root_rotations_mat']:
                if key in motion_data:
                    val = motion_data[key]
                    if hasattr(val, 'cpu'):
                        data_dict[key] = val.cpu().numpy()
                    elif isinstance(val, torch.Tensor):
                        data_dict[key] = val.numpy()
                    else:
                        data_dict[key] = val
            
            # Add SMPL-H full poses from construct_smpl_data_dict
            rot6d_tensor = torch.from_numpy(data_dict['rot6d'])
            transl_tensor = torch.from_numpy(data_dict['transl'])
            smpl_data = construct_smpl_data_dict(rot6d_tensor, transl_tensor)
            # Add all SMPL-H fields that retarget_fbx might need
            for k, v in smpl_data.items():
                if k not in data_dict:
                    data_dict[k] = v
            
            # Compute keypoints3d using body model (required by retarget_fbx.load_npz)
            if 'keypoints3d' not in data_dict:
                print("[FBX Export] Computing keypoints3d from SMPL-H poses using body model...")
                try:
                    from hymotion.pipeline.body_model import WoodenMesh
                except ImportError:
                    try:
                        from ..pipeline.body_model import WoodenMesh
                    except ImportError:
                        from body_model import WoodenMesh
                
                # Initialize body model
                body_model = WoodenMesh()
                
                # Forward pass to get keypoints3d
                # Note: forward() expects (B, J, 6) where B=batch (frames), J=joints
                # rot6d_tensor already has shape (T, J, 6), so T is treated as batch
                with torch.no_grad():
                    result = body_model.forward({
                        'rot6d': rot6d_tensor,  # (T, J, 6) - T frames as batch
                        'trans': transl_tensor  # (T, 3)
                    })
                    keypoints3d = result['keypoints3d'].cpu().numpy()  # (T, 52, 3)
                    data_dict['keypoints3d'] = keypoints3d
                
                print(f"[FBX Export] Generated keypoints3d with shape: {keypoints3d.shape}")
        else:
            # Already has SMPL-H format, just copy all fields
            print("[FBX Export] Using existing SMPL-H format data")
            for key in ['poses', 'trans', 'betas', 'gender', 'mocap_framerate', 'num_frames', 'Rh']:
                if key in motion_data:
                    data_dict[key] = motion_data[key]
            
            # Compute keypoints3d if missing (required by retarget_fbx.load_npz)
            if 'keypoints3d' not in motion_data:
                print("[FBX Export] Computing keypoints3d from SMPL-H poses using body model...")
                try:
                    from hymotion.pipeline.body_model import WoodenMesh, batch_rodrigues
                except ImportError:
                    try:
                        from ..pipeline.body_model import WoodenMesh, batch_rodrigues
                    except ImportError:
                        from body_model import WoodenMesh, batch_rodrigues
                
                try:
                    from hymotion.utils.geometry import rotation_matrix_to_rot6d
                except ImportError:
                    try:
                        from ..utils.geometry import rotation_matrix_to_rot6d
                    except ImportError:
                        from geometry import rotation_matrix_to_rot6d
                
                # Initialize body model  
                body_model = WoodenMesh()
                
                # Convert poses from angle-axis to rot6d for body model
                poses = motion_data['poses']
                trans = motion_data['trans']
                
                # Reshape poses to (T, J, 3) where T=frames, J=joints (52 for SMPL-H or 22 for body)
                T = poses.shape[0]
                poses_reshaped = poses.reshape(T, -1, 3)
                J = poses_reshaped.shape[1]
                
                print(f"[FBX Export] poses shape: {poses.shape} -> reshaped: {poses_reshaped.shape} (T={T}, J={J})")
                
                # CRITICAL: WoodenMesh body model only supports 22 body joints (no fingers)
                # SMPL-H has 52 joints total (22 body + 30 hand joints)
                # Extract only the first 22 joints for keypoints3d computation
                if J > 22:
                    print(f"[FBX Export] Note: Extracting first 22 body joints from {J} total joints for body model")
                    poses_body = poses_reshaped[:, :22, :]  # Only first 22 joints
                else:
                    poses_body = poses_reshaped
                
                # Convert angle-axis to rotation matrices then to rot6d
                # Process in batches of frames
                poses_tensor = torch.from_numpy(poses_body).float()  # (T, 22, 3)
                
                # Flatten to (T*22, 3) for batch_rodrigues
                poses_flat = poses_tensor.reshape(-1, 3)
                rot_mats_flat = batch_rodrigues(poses_flat)  # (T*22, 3, 3)
                rot_mats = rot_mats_flat.reshape(T, 22, 3, 3)  # (T, 22, 3, 3)
                rot6d = rotation_matrix_to_rot6d(rot_mats)  # (T, 22, 6)
                
                print(f"[FBX Export] rot6d shape: {rot6d.shape}")
                
                trans_tensor = torch.from_numpy(trans).float()
                
                print(f"[FBX Export] trans shape: {trans_tensor.shape}")
                
                # Forward pass to get keypoints3d for all frames
                # Need to reshape for forward() which expects (B, J, 6) where B=batch of frames
                with torch.no_grad():
                    # Process all frames as a batch
                    result = body_model.forward({
                        'rot6d': rot6d,  # (T, J, 6) - each frame is a "batch" item
                        'trans': trans_tensor  # (T, 3)
                    })
                    keypoints3d = result['keypoints3d'].cpu().numpy()  # (T, 52, 3)
                    data_dict['keypoints3d'] = keypoints3d
                
                print(f"[FBX Export] Generated keypoints3d with shape: {keypoints3d.shape}")
            else:
                data_dict['keypoints3d'] = motion_data['keypoints3d']
        
        # Ensure all required fields for retarget_fbx are present
        required_fields = ['poses', 'trans', 'betas', 'gender', 'mocap_framerate', 'Rh']
        for field in required_fields:
            if field not in data_dict:
                print(f"[FBX Export] Warning: {field} missing from motion data")
        
        # retarget_fbx.load_npz expects 'transl' key, but SMPL-H uses 'trans'
        # Add both to be safe
        if 'trans' in data_dict and 'transl' not in data_dict:
            data_dict['transl'] = data_dict['trans']
        
        # Save temporary NPZ
        print(f"[FBX Export] Saving temp NPZ with keys: {list(data_dict.keys())}")
        np.savez(temp_npz, **data_dict)
        
        print(f"[FBX Export] Loading skeletons for retargeting...")
        # Load source (NPZ) and target (FBX) skeletons
        src_skel = load_npz(temp_npz)
        tgt_man, tgt_scene, tgt_skel = load_fbx(template_fbx)
        
        # Load bone mappings
        mapping = load_bone_mapping("")  # Use built-in Mixamo mappings
        
        print(f"[FBX Export] Retargeting animation...")
        # Retarget animation
        force_scale = scale if scale > 0 else 0.0
        rots, locs = retarget_animation(src_skel, tgt_skel, mapping, force_scale, yaw_offset, neutral_fingers=True)
        
        print(f"[FBX Export] Applying animation and saving...")
        # Apply and save
        src_time_mode = tgt_scene.GetGlobalSettings().GetTimeMode()
        apply_retargeted_animation(tgt_scene, tgt_skel, rots, locs, src_skel.frame_start, src_skel.frame_end, src_time_mode)
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Save FBX
        save_fbx(tgt_man, tgt_scene, output_path)
        
        print(f"[FBX Export] Successfully exported custom FBX: {output_path}")
        
        # Cleanup temp file
        try:
            if os.path.exists(temp_npz):
                os.remove(temp_npz)
        except Exception as cleanup_error:
            print(f"[FBX Export] Warning: Could not remove temp file {temp_npz}: {cleanup_error}")
        
        # Save text description if provided
        if text_description:
            txt_path = output_path.replace(".fbx", ".txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text_description)
            print(f"[FBX Export] Saved description: {txt_path}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Custom FBX export failed: {e}")
        import traceback
        traceback.print_exc()
        # Cleanup temp file on error
        try:
            if 'temp_npz' in locals() and os.path.exists(temp_npz):
                os.remove(temp_npz)
        except Exception as cleanup_error:
            print(f"[FBX Export] Warning: Could not remove temp file during error cleanup: {cleanup_error}")
        return False


def export_motion_batch_to_fbx(
    motion_data_list: list,
    output_dir: str,
    filename_prefix: str = "motion",
    text_description: str = "",
    template_fbx: Optional[str] = None
) -> list:
    """
    Export multiple motion samples to FBX files.
    
    Args:
        motion_data_list: List of motion data dicts (each with 'rot6d' and 'transl')
        output_dir: Output directory
        filename_prefix: Filename prefix
        text_description: Text description for all samples
        template_fbx: Path to template FBX
        
    Returns:
        List of output file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    ts = get_timestamp()
    uid = str(uuid.uuid4())[:8]
    
    paths = []
    for i, motion_data in enumerate(motion_data_list):
        output_path = os.path.join(output_dir, f"{filename_prefix}_{ts}_{uid}_{i:03d}.fbx")
        success = export_motion_to_fbx(
            motion_data=motion_data,
            output_path=output_path,
            text_description=text_description,
            template_fbx=template_fbx
        )
        if success:
            paths.append(output_path)
    
    return paths


def load_motion_from_npz(npz_path: str) -> Dict[str, Any]:
    """
    Load motion data from NPZ file.
    Handles both formats:
    - Raw format: rot6d + transl (from ComfyUI or early pipeline)
    - Processed format: poses + trans + betas + gender (from gradio export)
    
    Args:
        npz_path: Path to NPZ file
        
    Returns:
        Dict with motion data
    """
    data = np.load(npz_path, allow_pickle=True)
    motion_data = {}
    
    # Load all available fields
    for key in ['rot6d', 'transl', 'keypoints3d', 'root_rotations_mat', 
                'poses', 'trans', 'betas', 'gender', 'mocap_framerate', 'num_frames', 'Rh']:
        if key in data:
            motion_data[key] = data[key]
    
    # Load metadata
    if 'text' in data:
        motion_data['text'] = str(data['text'])
    if 'duration' in data:
        motion_data['duration'] = float(data['duration'])
    if 'seed' in data:
        motion_data['seed'] = int(data['seed'])
    
    return motion_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export motion NPZ to FBX")
    parser.add_argument("--input", required=True, help="Input NPZ file")
    parser.add_argument("--output", required=True, help="Output FBX file")
    parser.add_argument("--template", help="Template FBX file (default: wooden boy)")
    
    args = parser.parse_args()
    
    # Load motion data
    print(f"Loading motion from: {args.input}")
    motion_data = load_motion_from_npz(args.input)
    text = motion_data.get('text', '')
    
    # Export to FBX
    success = export_motion_to_fbx(
        motion_data=motion_data,
        output_path=args.output,
        text_description=text,
        template_fbx=args.template
    )
    
    sys.exit(0 if success else 1)

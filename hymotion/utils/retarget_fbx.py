"""
FBX Retargeting Module for HY-Motion

This module provides animation retargeting functionality to transfer motion data
from SMPL-H format to custom FBX skeletons (e.g., Mixamo characters).

Original code adapted from:
    ComfyUI-HyMotion by Aero-Ex
    https://github.com/Aero-Ex/ComfyUI-HyMotion

Features:
    - Automatic bone mapping for Mixamo rigs (mixamorig: prefix)
    - Fuzzy bone matching for other skeleton naming conventions
    - SMPL-H to target skeleton animation transfer
    - Support for finger animations with neutral rest pose
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

# SMPL-H Mean Hand Pose Constants (from ComfyUI-HyMotion/body_model.py)
LEFT_HAND_MEAN_AA = np.array([
    0.1117,  0.0429, -0.4164,  0.1088, -0.0660, -0.7562, -0.0964, -0.0909,
    -0.1885, -0.1181,  0.0509, -0.5296, -0.1437,  0.0552, -0.7049, -0.0192,
    -0.0923, -0.3379, -0.4570, -0.1963, -0.6255, -0.2147, -0.0660, -0.5069,
    -0.3697, -0.0603, -0.0795, -0.1419, -0.0859, -0.6355, -0.3033, -0.0579,
    -0.6314, -0.1761, -0.1321, -0.3734,  0.8510,  0.2769, -0.0915, -0.4998,
    0.0266,  0.0529,  0.5356,  0.0460, -0.2774]
)
RIGHT_HAND_MEAN_AA = np.array([
    0.1117, -0.0429,  0.4164,  0.1088,  0.0660,  0.7562, -0.0964,  0.0909,
    0.1885, -0.1181, -0.0509,  0.5296, -0.1437, -0.0552,  0.7049, -0.0192,
    0.0923,  0.3379, -0.4570,  0.1963,  0.6255, -0.2147,  0.0660,  0.5069,
    -0.3697,  0.0603,  0.0795, -0.1419,  0.0859,  0.6355, -0.3033,  0.0579,
    0.6314, -0.1761,  0.1321,  0.3734,  0.8510, -0.2769,  0.0915, -0.4998,
    -0.0266, -0.0529,  0.5356, -0.0460,  0.2774]
)

# Attempt to import FBX SDK
try:
    import fbx
    from fbx import *
    HAS_FBX_SDK = True
except ImportError:
    HAS_FBX_SDK = False

# =============================================================================
# Math Utilities
# =============================================================================

def fbx_matrix_to_numpy(fbx_mat) -> np.ndarray:
    """Convert FBX matrix (FbxMatrix or FbxAMatrix) to numpy 4x4."""
    mat = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            mat[i, j] = fbx_mat.Get(i, j)
    return mat

def matrix_to_quaternion(mat: np.ndarray) -> np.ndarray:
    """Convert FBX 4x4 or 3x3 matrix to quaternion [w, x, y, z].
    FBX uses Row-Major vectors (v * M), so we transpose for SciPy (M @ v)."""
    m33 = mat[:3, :3].T
    rot = R.from_matrix(m33)
    q = rot.as_quat()  # [x, y, z, w]
    return np.array([q[3], q[0], q[1], q[2]])

def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    """Inverse of quaternion [w, x, y, z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]]) / np.sum(q**2)

def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def solve_rotation_between_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Returns a 3x3 rotation matrix that aligns v1 with v2 via shortest path."""
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm < 1e-9 or v2_norm < 1e-9: return np.eye(3)
    v1 = v1 / v1_norm
    v2 = v2 / v2_norm
    
    dot = np.dot(v1, v2)
    if dot > 0.999999: return np.eye(3)
    if dot < -0.999999:
        # 180 deg: Pick a perpendicular axis
        axis = np.array([0, 1, 0]) if abs(v1[0]) > 0.9 else np.array([1, 0, 0])
        axis = np.cross(v1, axis)
        axis /= (np.linalg.norm(axis) + 1e-9)
        return R.from_rotvec(axis * np.pi).as_matrix()
    
    axis = np.cross(v1, v2)
    axis_len = np.linalg.norm(axis)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    return R.from_rotvec(axis / axis_len * angle).as_matrix()

def look_at_matrix(fwd: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    """Creates a 3x3 rotation matrix where X is fwd and Z is up_hint."""
    f = fwd / (np.linalg.norm(fwd) + 1e-9)
    s = np.cross(up_hint, f)
    s = s / (np.linalg.norm(s) + 1e-9)
    u = np.cross(f, s)
    return np.stack((f, s, u), axis=-1)

# =============================================================================
# Core Data Structures
# =============================================================================

# =============================================================================
# Core Data Structures
# =============================================================================

class BoneData:
    def __init__(self, name: str):
        self.name = name
        self.parent_name = None
        self.local_matrix = np.eye(4)
        self.world_matrix = np.eye(4)
        self.head: np.ndarray = np.zeros(3)
        self.has_skeleton_attr: bool = False
        self.rest_rotation = np.array([1, 0, 0, 0])
        self.animation: dict[int, np.ndarray] = {}  # frame -> local [w, x, y, z]
        self.world_animation: dict[int, np.ndarray] = {} # frame -> world [w, x, y, z]
        self.location_animation: dict[int, np.ndarray] = {} # frame -> local pos
        self.world_location_animation: dict[int, np.ndarray] = {} # frame -> world pos

class Skeleton:
    def __init__(self, name: str = "Skeleton"):
        self.name = name
        self.bones: dict[str, BoneData] = {}
        self.all_nodes: dict[str, str] = {} # node_name -> node_name (for hierarchy)
        self.node_rest_rotations: dict[str, np.ndarray] = {}  # node -> world_rest_q [w,x,y,z]
        self.fps = 30.0
        self.frame_start = 0
        self.frame_end = 0

    def add_bone(self, bone: BoneData):
        self.bones[bone.name.lower()] = bone
    
    def get_bone_case_insensitive(self, name: str) -> BoneData:
        lower_name = name.lower()
        # 1. Direct match
        if lower_name in self.bones: return self.bones[lower_name]
        # 2. Strip prefix from query
        if ":" in lower_name:
            stripped = lower_name.split(":")[-1]
            if stripped in self.bones: return self.bones[stripped]
        # 3. Strip prefix from stored bone names
        for bname, bone in self.bones.items():
            if ":" in bname:
                if bname.split(":")[-1] == lower_name: return bone
        return None

# =============================================================================
# NPZ Support (SMPL-H)
# =============================================================================

def rot6d_to_matrix_np(d6: np.ndarray) -> np.ndarray:
    """Numpy version of 6D rotation to 3x3 matrix.
    Correctly handles the [3, 2] viewing used in SMPL-H/HyMotion.
    """
    shape = d6.shape[:-1]
    # Reshape to (Batch, 3, 2) where :
    # Column 0 = d6[..., 0], d6[..., 2], d6[..., 4]
    # Column 1 = d6[..., 1], d6[..., 3], d6[..., 5]
    x = d6.reshape(-1, 3, 2)
    a1 = x[..., 0]
    a2 = x[..., 1]
    
    b1 = a1 / (np.linalg.norm(a1, axis=1, keepdims=True) + 1e-9)
    b2 = a2 - np.sum(b1 * a2, axis=1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=1, keepdims=True) + 1e-9)
    b3 = np.cross(b1, b2, axis=1)
    return np.stack((b1, b2, b3), axis=-1).reshape(*shape, 3, 3)

def load_npz(filepath: str) -> Skeleton:
    """Load motion data from NPZ file (HyMotion/SMPL-H format)."""
    data = np.load(filepath)
    # Typical HyMotion NPZ structure:
    # keypoints3d (T, 52, 3), rot6d (T, 22, 6), transl (T, 3), root_rotations_mat (T, 3, 3)
    kps = data['keypoints3d']
    transl = data['transl']
    rot6d = data.get('rot6d')
    root_mat = data.get('root_rotations_mat')
    
    T = kps.shape[0]
    # Global coordinates = Local keypoints + Translation
    global_kps = kps + transl[:, np.newaxis, :]
    
    names = [
        "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2", "L_Ankle", "R_Ankle", "Spine3", 
        "L_Foot", "R_Foot", "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", 
        "L_Wrist", "R_Wrist", "L_Index1", "L_Index2", "L_Index3", "L_Middle1", "L_Middle2", "L_Middle3", "L_Pinky1", "L_Pinky2", 
        "L_Pinky3", "L_Ring1", "L_Ring2", "L_Ring3", "L_Thumb1", "L_Thumb2", "L_Thumb3", "R_Index1", "R_Index2", "R_Index3", 
        "R_Middle1", "R_Middle2", "R_Middle3", "R_Pinky1", "R_Pinky2", "R_Pinky3", "R_Ring1", "R_Ring2", "R_Ring3", "R_Thumb1", 
        "R_Thumb2", "R_Thumb3"
    ]
    
    parents = [
        -1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14, 16, 17, 18, 19,
        20, 22, 23, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35,
        21, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50
    ]
    
    skel = Skeleton(os.path.basename(filepath))
    skel.frame_start = 0
    skel.frame_end = T - 1
    skel.fps = 30.0
    
    # Reconstruct World Rotations via Forward Kinematics
    world_rots = np.zeros((T, 52, 3, 3))
    
    # 1. Compute Canonical Rest Pose (All Identity Local)
    # This keeps the character stable and prevents 'morphing'
    rest_world_rots = np.zeros((52, 3, 3))
    rest_world_rots[0] = np.eye(3) # Root
    for i in range(1, 52):
        p = parents[i]
        rest_world_rots[i] = rest_world_rots[p] # Local Identity
            
    # Child lookup for vector-based solving (fingers)
    children = {}
    for idx, p in enumerate(parents):
        if p != -1 and p not in children: # Find first child
            children[p] = idx

    rot_mats_all = rot6d_to_matrix_np(rot6d) if rot6d is not None else None
    
    # HYMOTION FIX: If NPZ only has 22 joints, inject the SMPL-H Mean Hand Pose.
    # This matches the behavior of the ComfyUI-HyMotion FBX export.
    if rot_mats_all is not None and rot_mats_all.shape[1] == 22:
        print(f"Injecting SMPL-H Mean Hand Pose (Relaxed) for {T} frames...")
        # left: 15 joints, right: 15 joints -> Total 52
        l_hand_mats = R.from_rotvec(LEFT_HAND_MEAN_AA.reshape(15, 3)).as_matrix()
        r_hand_mats = R.from_rotvec(RIGHT_HAND_MEAN_AA.reshape(15, 3)).as_matrix()
        
        # Expand for all T frames
        l_hand_batch = np.tile(l_hand_mats[np.newaxis, :], (T, 1, 1, 1))
        r_hand_batch = np.tile(r_hand_mats[np.newaxis, :], (T, 1, 1, 1))
        
        # Cat: (T, 22, 3, 3) + (T, 15, 3, 3) + (T, 15, 3, 3) -> (T, 52, 3, 3)
        rot_mats_all = np.concatenate([rot_mats_all, l_hand_batch, r_hand_batch], axis=1)

    if rot_mats_all is not None and root_mat is not None:
        for f in range(T):
            # 1. Root (Pelvis)
            world_rots[f, 0] = root_mat[f]
            # 2. Hierarchy FK
            for i in range(1, 52):
                p = parents[i]
                if i < rot_mats_all.shape[1]:
                    # Core joints (0-21)
                    world_rots[f, i] = world_rots[f, p] @ rot_mats_all[f, i]
                else:
                    # Finger joint: Recursive segment alignment
                    # Align this segment's orientation with its parent segment's line
                    child = children.get(i)
                    if child is not None:
                        # Vector of the parent segment
                        v_parent_seg = global_kps[f, i] - global_kps[f, p]
                        # Vector of current segment
                        v_curr_seg = global_kps[f, child] - global_kps[f, i]
                        
                        # Find rotation that aligns parent segment direction with current segment
                        R_align = solve_rotation_between_vectors(v_parent_seg, v_curr_seg)
                        
                        # Apply this change to the parent's world orientation.
                        # This propagates the curl and spread down the finger chain.
                        world_rots[f, i] = R_align @ world_rots[f, p]
                    else:
                        # Finger tip: follow parent
                        world_rots[f, i] = world_rots[f, p]
    else:
        # Fallback
        for f in range(T):
            world_rots[f] = rest_world_rots

    for i, name in enumerate(names):
        bone = BoneData(name)
        p_idx = parents[i]
        if p_idx != -1: bone.parent_name = names[p_idx]
        
        for f in range(T):
            # SciPy expects [x,y,z,w] internally, matrix_to_quaternion handles it.
            # Convert world matrices (Column-Major) to Row-Major for the utility
            bone.world_animation[f] = matrix_to_quaternion(world_rots[f, i].T)
            bone.world_location_animation[f] = global_kps[f, i]
            
        # NPZ/SMPL Rest Pose: Identity (Pancake Hand).
        # This makes the NPZ's 3D curl behave as an additive animation offset.
        bone.rest_rotation = matrix_to_quaternion(rest_world_rots[i].T)
        
        bone.head = global_kps[0, i]
        bone.world_matrix = np.eye(4)
        bone.world_matrix[:3, :3] = rest_world_rots[i]
        bone.world_matrix[3, :3] = global_kps[0, i]
        
        skel.add_bone(bone)
        skel.all_nodes[name] = name
        skel.node_rest_rotations[name] = bone.rest_rotation
        
    return skel

# =============================================================================
BASE_BONE_MAPPING = {
    "hips": "mixamorig:hips",
    "pelvis": "mixamorig:hips",
    "spine": "mixamorig:spine",
    "spine1": "mixamorig:spine",
    "spine2": "mixamorig:spine1",
    "spine3": "mixamorig:spine2",
    "chest": "mixamorig:spine2",

    "neck": "mixamorig:neck",
    "head": "mixamorig:head",
    "leftupleg": "mixamorig:leftupleg",
    "rightupleg": "mixamorig:rightupleg",
    "leftleg": "mixamorig:leftleg",
    "rightleg": "mixamorig:rightleg",
    "leftfoot": "mixamorig:leftfoot",
    "rightfoot": "mixamorig:rightfoot",
    "leftshoulder": "mixamorig:leftshoulder",
    "rightshoulder": "mixamorig:rightshoulder",
    "leftarm": "mixamorig:leftarm",
    "rightarm": "mixamorig:rightarm",
    "leftforearm": "mixamorig:leftforearm",
    "rightforearm": "mixamorig:rightforearm",
    "lefthand": "mixamorig:lefthand",
    "righthand": "mixamorig:righthand",

    # SMPL-H Naming Style (Explicit)
    "l_collar": "mixamorig:leftshoulder", "r_collar": "mixamorig:rightshoulder",
    "l_shoulder": "mixamorig:leftarm", "r_shoulder": "mixamorig:rightarm",
    "l_elbow": "mixamorig:leftforearm", "r_elbow": "mixamorig:rightforearm",
    "l_wrist": "mixamorig:lefthand", "r_wrist": "mixamorig:righthand",
    
    # SMPL-H Uppercase variants (CRITICAL for proper matching)
    "L_Wrist": "mixamorig:lefthand", "R_Wrist": "mixamorig:righthand",
    "L_Elbow": "mixamorig:leftforearm", "R_Elbow": "mixamorig:rightforearm",
    "L_Shoulder": "mixamorig:leftarm", "R_Shoulder": "mixamorig:rightarm",
    "L_Collar": "mixamorig:leftshoulder", "R_Collar": "mixamorig:rightshoulder",

    # Explicit SMPL-H and Common Hand variations
    "left_hand": "mixamorig:lefthand", "right_hand": "mixamorig:righthand",
    "left_wrist": "mixamorig:lefthand", "right_wrist": "mixamorig:righthand",
    "lhand": "mixamorig:lefthand", "rhand": "mixamorig:righthand",
    
    # Fingers - Left Hand
    "leftthumb1": "mixamorig:lefthandthumb1", 
    "leftthumbmedial": "mixamorig:lefthandthumb1",
    "leftthumb2": "mixamorig:lefthandthumb2", 
    "leftthumbdistal": "mixamorig:lefthandthumb2",
    "leftthumb3": "mixamorig:lefthandthumb3",
    "l_thumb1": "mixamorig:lefthandthumb1", 
    "l_thumb2": "mixamorig:lefthandthumb2", 
    "l_thumb3": "mixamorig:lefthandthumb3",
    
    "leftindex1": "mixamorig:lefthandindex1", 
    "leftindexmedial": "mixamorig:lefthandindex1",
    "leftindex2": "mixamorig:lefthandindex2", 
    "leftindexdistal": "mixamorig:lefthandindex2",
    "leftindex3": "mixamorig:lefthandindex3",
    "l_index1": "mixamorig:lefthandindex1", 
    "l_index2": "mixamorig:lefthandindex2", 
    "l_index3": "mixamorig:lefthandindex3",
    
    "leftmiddle1": "mixamorig:lefthandmiddle1",
    "leftmiddle2": "mixamorig:lefthandmiddle2",
    "leftmiddle3": "mixamorig:lefthandmiddle3",
    "l_middle1": "mixamorig:lefthandmiddle1", 
    "l_middle2": "mixamorig:lefthandmiddle2", 
    "l_middle3": "mixamorig:lefthandmiddle3",
    
    "leftring1": "mixamorig:lefthandring1", 
    "leftringmedial": "mixamorig:lefthandring1",
    "leftring2": "mixamorig:lefthandring2", 
    "leftringdistal": "mixamorig:lefthandring2",
    "leftring3": "mixamorig:lefthandring3",
    "l_ring1": "mixamorig:lefthandring1", 
    "l_ring2": "mixamorig:lefthandring2", 
    "l_ring3": "mixamorig:lefthandring3",
    
    "leftpinky1": "mixamorig:lefthandpinky1", 
    "leftlittlemedial": "mixamorig:lefthandpinky1",
    "leftpinky2": "mixamorig:lefthandpinky2", 
    "leftlittledistal": "mixamorig:lefthandpinky2",
    "leftpinky3": "mixamorig:lefthandpinky3",
    "l_pinky1": "mixamorig:lefthandpinky1", 
    "l_pinky2": "mixamorig:lefthandpinky2", 
    "l_pinky3": "mixamorig:lefthandpinky3",
    
    # Fingers - Right Hand
    "rightthumb1": "mixamorig:righthandthumb1",
    "rightthumbmedial": "mixamorig:righthandthumb1",
    "rightthumb2": "mixamorig:righthandthumb2",
    "rightthumbdistal": "mixamorig:righthandthumb2",
    "rightthumb3": "mixamorig:righthandthumb3",
    "r_thumb1": "mixamorig:righthandthumb1", 
    "r_thumb2": "mixamorig:righthandthumb2", 
    "r_thumb3": "mixamorig:righthandthumb3",
    
    "rightindex1": "mixamorig:righthandindex1",
    "rightindexmedial": "mixamorig:righthandindex1",
    "rightindex2": "mixamorig:righthandindex2",
    "rightindexdistal": "mixamorig:righthandindex2",
    "rightindex3": "mixamorig:righthandindex3",
    "r_index1": "mixamorig:righthandindex1", 
    "r_index2": "mixamorig:righthandindex2", 
    "r_index3": "mixamorig:righthandindex3",
    
    "rightmiddle1": "mixamorig:righthandmiddle1",
    "rightmiddle2": "mixamorig:righthandmiddle2",
    "rightmiddle3": "mixamorig:righthandmiddle3",
    "r_middle1": "mixamorig:righthandmiddle1", 
    "r_middle2": "mixamorig:righthandmiddle2", 
    "r_middle3": "mixamorig:righthandmiddle3",
    
    "rightring1": "mixamorig:righthandring1",
    "rightringmedial": "mixamorig:righthandring1",
    "rightring2": "mixamorig:righthandring2",
    "rightringdistal": "mixamorig:righthandring2",
    "rightring3": "mixamorig:righthandring3",
    "r_ring1": "mixamorig:righthandring1", 
    "r_ring2": "mixamorig:righthandring2", 
    "r_ring3": "mixamorig:righthandring3",
    
    "rightpinky1": "mixamorig:righthandpinky1",
    "rightlittlemedial": "mixamorig:righthandpinky1",
    "rightpinky2": "mixamorig:righthandpinky2",
    "rightlittledistal": "mixamorig:righthandpinky2",
    "rightpinky3": "mixamorig:righthandpinky3",
    "r_pinky1": "mixamorig:righthandpinky1", 
    "r_pinky2": "mixamorig:righthandpinky2", 
    "r_pinky3": "mixamorig:righthandpinky3",
    
    
    # Feet/Toes
    "l_foot": "mixamorig:lefttoebase",
    "r_foot": "mixamorig:righttoebase",
    
    # SMPL-H Uppercase leg/foot variants (CRITICAL for proper matching)
    "L_Hip": "mixamorig:leftupleg", "R_Hip": "mixamorig:rightupleg",
    "L_Knee": "mixamorig:leftleg", "R_Knee": "mixamorig:rightleg",
    "L_Ankle": "mixamorig:leftfoot", "R_Ankle": "mixamorig:rightfoot",
    "L_Foot": "mixamorig:lefttoebase", "R_Foot": "mixamorig:righttoebase",
}

# Comprehensive bone name aliases for fuzzy matching
# Format: target_keyword: [list of possible source names]
FUZZY_ALIASES = {
    # Core skeleton
    'hips': ['pelvis', 'root_joint', 'spine_01', 'hip', 'root', 'cog', 'center', 'base'],
    'spine': ['spine', 'chest', 'back', 'torso', 'spine1', 'spine2', 'spine3'],
    'neck': ['neck', 'neck_01', 'neckbase'],
    'head': ['head', 'head_top', 'skull', 'cranium'],
    
    # Legs
    'upleg': ['thigh', 'hip', 'upperleg', 'leg_upper', 'femur'],
    'leg': ['knee', 'leg', 'lowerleg', 'leg_lower', 'shin', 'calf'],
    'foot': ['ankle', 'foot', 'ankle_01'],
    'toe': ['toe', 'ball', 'toebase', 'foot_end'],
    
    # Arms
    'shoulder': ['collar', 'clavicle', 'shoulder_01', 'scapula'],
    'arm': ['shoulder', 'upperarm', 'arm', 'arm_upper', 'humerus'],
    'forearm': ['elbow', 'forearm', 'arm_lower', 'lowerarm', 'ulna'],
    'hand': ['wrist', 'hand', 'palm'],
    
    # Fingers
    'thumb': ['thumb', 'pollex'],
    'index': ['index', 'pointer'],
    'middle': ['middle', 'long'],
    'ring': ['ring', 'third'],
    'pinky': ['pinky', 'little', 'small'],
}

# Extended bone keywords for classification
BONE_KEYWORDS = {
    'root': ['hips', 'pelvis', 'root', 'cog', 'center'],
    'spine': ['spine', 'chest', 'back', 'torso'],
    'neck': ['neck'],
    'head': ['head', 'skull'],
    'leg': ['upleg', 'thigh', 'knee', 'leg', 'ankle', 'foot', 'toe'],
    'arm': ['shoulder', 'collar', 'clavicle', 'arm', 'elbow', 'forearm', 'hand', 'wrist'],
    'finger': ['thumb', 'index', 'middle', 'ring', 'pinky', 'digit'],
}

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings for typo detection"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def normalize_bone_name(name: str) -> str:
    """Normalize bone name by removing prefixes and common variations"""
    name_lower = name.lower()
    
    # Remove common prefixes
    prefixes = ['mixamorig:', 'bip01_', 'bip001_', 'joint_', 'bone_', 'def_', 'rig_', 'valvebiped_']
    for prefix in prefixes:
        if name_lower.startswith(prefix):
            name_lower = name_lower[len(prefix):]
    
    # Remove underscores and dots for comparison
    name_lower = name_lower.replace('_', '').replace('.', '').replace('-', '')
    
    return name_lower

def detect_side(name: str) -> tuple[bool, bool]:
    """
    Detect if bone is left or right side with improved logic.
    Returns: (is_left, is_right)
    """
    name_lower = name.lower()
    
    # Explicit left markers
    is_left = any(x in name_lower for x in [
        'left', '.l', '_l', 'l_', 'lhand', 'lfoot', 'larm', 'lleg'
    ]) or (name_lower.startswith('l') and len(name_lower) > 1 and name_lower[1] in ['_', '.'])
    
    # Explicit right markers
    is_right = any(x in name_lower for x in [
        'right', '.r', '_r', 'r_', 'rhand', 'rfoot', 'rarm', 'rleg'
    ]) or (name_lower.startswith('r') and len(name_lower) > 1 and name_lower[1] in ['_', '.'])
    
    return is_left, is_right

def classify_bone(name: str) -> list[str]:
    """Classify bone into categories (root, spine, leg, arm, finger, etc.)"""
    name_norm = normalize_bone_name(name)
    categories = []
    
    for category, keywords in BONE_KEYWORDS.items():
        if any(kw in name_norm for kw in keywords):
            categories.append(category)
    
    return categories

def calculate_bone_similarity(source_name: str, target_name: str, use_aliases: bool = True) -> float:
    """
    Calculate similarity score between two bone names (0.0 = no match, 1.0 = perfect match)
    Uses multiple matching strategies.
    """
    s_norm = normalize_bone_name(source_name)
    t_norm = normalize_bone_name(target_name)
    
    # Strategy 1: Exact match (perfect score)
    if s_norm == t_norm:
        return 1.0
    
    # ⚠️ CRITICAL: Strict category separation
    s_categories = classify_bone(source_name)
    t_categories = classify_bone(target_name)
    
    # Prevent 'hand' bones from matching 'finger' bones (common failure case)
    if ('hand' in s_categories or 'arm' in s_categories) and 'finger' in t_categories:
        # A generic hand bone should never match a specific finger bone
        return 0.0
    if 'finger' in s_categories and ('hand' in t_categories or 'arm' in t_categories):
        return 0.0

    # Strategy 2: One contains the other (substring match)
    if s_norm in t_norm or t_norm in s_norm:
        overlap = min(len(s_norm), len(t_norm))
        total = max(len(s_norm), len(t_norm))
        score = 0.85 * (overlap / total)
        
        # Apply category penalty if they don't share enough categories
        if s_categories and t_categories:
            intersect = set(s_categories) & set(t_categories)
            if not intersect:
                score *= 0.5 # Major penalty for body part mismatch (e.g. arm containing leg substring)
        return score
    
    # Strategy 3: Alias matching
    if use_aliases:
        for target_keyword, source_aliases in FUZZY_ALIASES.items():
            if target_keyword in t_norm:
                for alias in source_aliases:
                    if alias in s_norm:
                        return 0.75  # Good match via alias
    
    # Strategy 4: Levenshtein distance (typo tolerance)
    edit_distance = levenshtein_distance(s_norm, t_norm)
    max_len = max(len(s_norm), len(t_norm))
    if max_len > 0:
        similarity = 1.0 - (edit_distance / max_len)
        if similarity > 0.6:  # Only accept if reasonably similar
            score = similarity * 0.7
            if s_categories and t_categories:
                if not (set(s_categories) & set(t_categories)):
                    score *= 0.3 # Heavy penalty for category mismatch
            return score
    
    # Strategy 5: Category matching (same type of bone)
    if s_categories and t_categories:
        overlap = len(set(s_categories) & set(t_categories))
        if overlap > 0:
            return 0.4 * (overlap / max(len(s_categories), len(t_categories)))
    
    return 0.0  # No match

def find_best_bone_match(
    target_bone_name: str,
    source_skeleton: 'Skeleton',
    already_mapped_sources: set[str],
    require_side_match: bool = True
) -> tuple[str, float]:
    """
    Find the best matching source bone for a target bone.
    Returns: (source_bone_name, confidence_score)
    """
    t_left, t_right = detect_side(target_bone_name)
    best_match = None
    best_score = 0.0
    
    for s_bone_name in source_skeleton.bones.keys():
        # Skip already mapped bones
        if s_bone_name in already_mapped_sources:
            continue
        
        # Side matching check
        if require_side_match:
            s_left, s_right = detect_side(s_bone_name)
            # If one is sided and the other isn't, or they're opposite sides, skip
            if (t_left != s_left) or (t_right != s_right):
                continue
        
        # Calculate similarity
        score = calculate_bone_similarity(s_bone_name, target_bone_name)
        
        if score > best_score:
            best_score = score
            best_match = s_bone_name
    
    return best_match, best_score

def get_skeleton_height(skeleton: Skeleton, mapping: list) -> float:
    y_coords = []
    keywords = ['hips', 'spine', 'neck', 'head', 'arm', 'leg', 'foot', 'ankle', 'knee', 'shoulder', 'elbow', 'pelvis', 'joint', 'mixamo']
    y_min, y_max = 999999.0, -999999.0
    found_any = False
    for _, bone in skeleton.bones.items():
        name = bone.name.lower()
        if any(k in name for k in keywords):
            h_val = bone.head[1]
            # Ignore absolute zero if we can, as it's often a failure to sample
            if abs(h_val) < 1e-6: continue
            y_min = min(y_min, h_val)
            y_max = max(y_max, h_val)
            y_coords.append(h_val)
            found_any = True
    
    if not found_any or y_max <= y_min: return 1.0
    return y_max - y_min

def load_bone_mapping(filepath: str) -> dict[str, str]:
    mapping = BASE_BONE_MAPPING.copy()
    if not filepath or not os.path.exists(filepath):
        print(f"Using hardcoded bone mappings (no JSON file needed)")
        return mapping
    with open(filepath, 'r') as f:
        data = json.load(f)
    bones = data.get("bones", {})
    for key, values in bones.items():
        if isinstance(values, list):
            if len(values) >= 2:
                # Text.py style: [source_name, ..., target_name]
                src = values[0].lower()
                tgt = values[-1].lower()
                mapping[src] = tgt
            elif len(values) == 1:
                mapping[key.lower()] = values[0].lower()
        elif isinstance(values, str):
            mapping[key.lower()] = values.lower()
    return mapping

def get_fbx_rotation_order_str(node: fbx.FbxNode) -> str:
    order = node.RotationOrder.Get()
    mapping = {0:'xyz', 1:'xzy', 2:'yzx', 3:'yxz', 4:'zxy', 5:'zyx'}
    return mapping.get(order, 'xyz')

# =============================================================================
# FBX Logic
# =============================================================================

def collect_skeleton_nodes(node: fbx.FbxNode, skeleton: Skeleton, parent_name: str = None, depth: int = 0, sampling_time: fbx.FbxTime = None):
    attr = node.GetNodeAttribute()
    node_name = node.GetName()
    is_bone = False
    
    if attr:
        attr_type = attr.GetAttributeType()
        if attr_type in [3, 4]: is_bone = True
        elif attr_type == 2 and (node.GetChildCount() > 0 or parent_name): is_bone = True
    
    name_lower = node_name.lower()
    keywords = ['hips', 'hip', 'spine', 'neck', 'head', 'arm', 'leg', 'foot', 'ankle', 'knee', 'shoulder', 'elbow', 'pelvis', 'joint', 'mixamo', 'thigh', 'upper', 'forearm', 'hand', 'finger', 'clavicle', 'collar', 'toe', 'thumb', 'index', 'middle', 'ring', 'pinky', 'upleg', 'downleg', 'wrist', 'chest', 'belly']
    if any(k in name_lower for k in keywords): is_bone = True
    
    # Debug reject
    if not is_bone and depth < 3:
        pass # print(f"  SKIPPING: {node_name}")
    elif is_bone:
        pass # print(f"  MATCHED: {node_name}")
    
    t_eval = sampling_time if sampling_time else fbx.FbxTime(0)
    global_mat_fbx = node.EvaluateGlobalTransform(t_eval)
    local_mat_fbx = node.EvaluateLocalTransform(t_eval)
    
    skeleton.node_rest_rotations[node_name] = matrix_to_quaternion(fbx_matrix_to_numpy(global_mat_fbx))
    
    # Bind pose fallback (prioritize for all nodes for consistency)
    scene = node.GetScene()
    if scene:
        for i in range(scene.GetPoseCount()):
            pose = scene.GetPose(i)
            if pose and pose.IsBindPose():
                idx = pose.Find(node)
                if idx != -1:
                    np_pose = fbx_matrix_to_numpy(pose.GetMatrix(idx))
                    q = matrix_to_quaternion(np_pose)
                    skeleton.node_rest_rotations[node_name] = q
                    break
    
    if is_bone:
        # Check for duplicates (case-insensitive)
        existing = skeleton.get_bone_case_insensitive(node_name)
        is_current_real = (attr and attr.GetAttributeType() in [3, 4])
        
        if existing:
            # PRIORITIZE: If current is a real bone node but existing was just a keyword null/mesh
            if is_current_real and not existing.has_skeleton_attr:
                skeleton.bones.pop(existing.name.lower(), None)
            else:
                is_bone = False

    if is_bone:
        bone = BoneData(node_name)
        bone.has_skeleton_attr = (attr and attr.GetAttributeType() in [3, 4])
        bone.parent_name = parent_name
        bone.local_matrix = fbx_matrix_to_numpy(local_mat_fbx)
        bone.world_matrix = fbx_matrix_to_numpy(global_mat_fbx)
        # Use BindPose for head if possible
        t_global = global_mat_fbx.GetT()
        bone.head = np.array([t_global[0], t_global[1], t_global[2]])
        
        # Check if we already found a better orientation from BindPose
        bone.rest_rotation = skeleton.node_rest_rotations[node_name]
        
        # Re-check pose specifically for Head translation if it was overwritten
        if scene:
            for i in range(scene.GetPoseCount()):
                pose = scene.GetPose(i)
                if pose and pose.IsBindPose():
                    idx = pose.Find(node)
                    if idx != -1:
                        np_pose = fbx_matrix_to_numpy(pose.GetMatrix(idx))
                        if np.linalg.norm(np_pose[3, :3]) > 1e-4:
                            bone.head = np_pose[3, :3]
                        break
        
        skeleton.add_bone(bone)
        parent_name = node_name
    
    # Still record this node in the global node map for parent lookups
    skeleton.all_nodes[node_name] = node_name
        
    for i in range(node.GetChildCount()):
        collect_skeleton_nodes(node.GetChild(i), skeleton, parent_name, depth + 1, sampling_time)

def extract_animation(scene: FbxScene, skeleton: Skeleton):
    stack = scene.GetCurrentAnimationStack()
    if not stack: return
    time_span = stack.GetLocalTimeSpan()
    start = time_span.GetStart()
    stop = time_span.GetStop()
    mode = scene.GetGlobalSettings().GetTimeMode()
    skeleton.frame_start = int(start.GetFrameCount(mode))
    skeleton.frame_end = int(stop.GetFrameCount(mode))
    skeleton.fps = FbxTime.GetFrameRate(mode)
    
    def sample(node):
        bone = skeleton.get_bone_case_insensitive(node.GetName())
        if bone:
            for f in range(skeleton.frame_start, skeleton.frame_end + 1):
                t = FbxTime()
                t.SetFrame(f, mode)
                lmat = fbx_matrix_to_numpy(node.EvaluateLocalTransform(t))
                wmat = fbx_matrix_to_numpy(node.EvaluateGlobalTransform(t))
                bone.animation[f] = matrix_to_quaternion(lmat)
                bone.world_animation[f] = matrix_to_quaternion(wmat)
                # Save both Local and World Translation for root analysis
                bone.location_animation[f] = lmat[3, :3]
                bone.world_location_animation[f] = wmat[3, :3] # NEW: world-space pos
        for i in range(node.GetChildCount()): sample(node.GetChild(i))
    sample(scene.GetRootNode())

def load_fbx(filepath: str, sample_rest_frame: int = None):
    manager = FbxManager.Create()
    scene = FbxScene.Create(manager, "Scene")
    importer = FbxImporter.Create(manager, "")
    importer.Initialize(filepath, -1, manager.GetIOSettings())
    importer.Import(scene)
    importer.Destroy()
    
    stack = scene.GetCurrentAnimationStack()
    t_sample = None
    if sample_rest_frame is not None:
        t_sample = FbxTime()
        t_sample.SetFrame(sample_rest_frame, scene.GetGlobalSettings().GetTimeMode())
    else:
        scene.SetCurrentAnimationStack(None)
        
    skeleton = Skeleton(os.path.basename(filepath))
    collect_skeleton_nodes(scene.GetRootNode(), skeleton, sampling_time=t_sample)
    scene.SetCurrentAnimationStack(stack)
    return manager, scene, skeleton

def apply_retargeted_animation(scene, skeleton, ret_rots, ret_locs, fstart, fend, source_time_mode=None):
    if source_time_mode: scene.GetGlobalSettings().SetTimeMode(source_time_mode)
    tmode = scene.GetGlobalSettings().GetTimeMode()
    
    # Clear old stacks
    for i in range(scene.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(FbxAnimStack.ClassId)) - 1, -1, -1):
        s = scene.GetSrcObject(fbx.FbxCriteria.ObjectType(FbxAnimStack.ClassId), i)
        scene.DisconnectSrcObject(s)
        s.Destroy()
        
    stack = FbxAnimStack.Create(scene, "Take 001")
    layer = FbxAnimLayer.Create(scene, "BaseLayer")
    stack.AddMember(layer)
    scene.SetCurrentAnimationStack(stack)
    
    def apply_node(node):
        name = node.GetName()
        if name in ret_rots:
            node.LclRotation.ModifyFlag(fbx.FbxPropertyFlags.EFlags.eAnimatable, True)
            ord_str = get_fbx_rotation_order_str(node)
            # PreRotation and PostRotation handling
            pv = node.PreRotation.Get()
            pq = R.from_euler('xyz', [pv[0], pv[1], pv[2]], degrees=True).as_quat()
            pre_inv = quaternion_inverse(np.array([pq[3], pq[0], pq[1], pq[2]]))
            
            post_v = node.PostRotation.Get()
            post_q = R.from_euler('xyz', [post_v[0], post_v[1], post_v[2]], degrees=True).as_quat()
            post_inv = quaternion_inverse(np.array([post_q[3], post_q[0], post_q[1], post_q[2]]))
            
            cx = node.LclRotation.GetCurve(layer, "X", True)
            cy = node.LclRotation.GetCurve(layer, "Y", True)
            cz = node.LclRotation.GetCurve(layer, "Z", True)
            cx.KeyModifyBegin(); cy.KeyModifyBegin(); cz.KeyModifyBegin()
            
            for f, q_local in ret_rots[name].items():
                t = FbxTime()
                t.SetFrame(f, tmode)
                # Correct FBX rotation solve: LclR = Pre.inv * (Parent.inv * World) * Post.inv
                q_final = quaternion_multiply(pre_inv, quaternion_multiply(q_local, post_inv))
                # Convert to Euler with correct order
                rot_q = R.from_quat([q_final[1], q_final[2], q_final[3], q_final[0]])
                # Get order mapping for curves
                ord_lower = ord_str.lower()
                e = rot_q.as_euler(ord_lower, degrees=True)
                
                # Map SciPy output (which follows ord_lower) to curves [cx, cy, cz]
                curve_map = {'x': cx, 'y': cy, 'z': cz}
                for i, char in enumerate(ord_lower):
                    c = curve_map[char]
                    val = e[i]
                    idx = c.KeyAdd(t)[0]
                    c.KeySetValue(idx, float(val))
                    c.KeySetInterpolation(idx, fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationLinear)
            cx.KeyModifyEnd(); cy.KeyModifyEnd(); cz.KeyModifyEnd()
            
        if name in ret_locs:
            node.LclTranslation.ModifyFlag(fbx.FbxPropertyFlags.EFlags.eAnimatable, True)
            tx = node.LclTranslation.GetCurve(layer, "X", True)
            ty = node.LclTranslation.GetCurve(layer, "Y", True)
            tz = node.LclTranslation.GetCurve(layer, "Z", True)
            tx.KeyModifyBegin(); ty.KeyModifyBegin(); tz.KeyModifyBegin()
            for f, loc in ret_locs[name].items():
                t = FbxTime()
                t.SetFrame(f, tmode)
                for c, val in zip([tx, ty, tz], loc):
                    idx = c.KeyAdd(t)[0]
                    c.KeySetValue(idx, float(val))
                    c.KeySetInterpolation(idx, fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationLinear)
            tx.KeyModifyEnd(); ty.KeyModifyEnd(); tz.KeyModifyEnd()
            
        for i in range(node.GetChildCount()): apply_node(node.GetChild(i))
    apply_node(scene.GetRootNode())

def retarget_animation(src_skel: Skeleton, tgt_skel: Skeleton, mapping: dict[str, str], force_scale: float = 0.0, yaw_offset: float = 0.0, neutral_fingers: bool = True):
    print("Retargeting Animation...")
    ret_rots = {}
    ret_locs = {}
    
    yaw_q_raw = R.from_euler('y', yaw_offset, degrees=True).as_quat()
    yaw_q = np.array([yaw_q_raw[3], yaw_q_raw[0], yaw_q_raw[1], yaw_q_raw[2]])
    
    # 1. Base Mapping
    active = []
    mapped_targets = set()
    mapped_sources = set()

    for s_key, t_key in mapping.items():
        s_bone = src_skel.get_bone_case_insensitive(s_key)
        t_bone = tgt_skel.get_bone_case_insensitive(t_key)
        
        if s_bone and t_bone:
            # Skip if either bone is already part of a mapping
            if t_bone.name in mapped_targets or s_bone.name in mapped_sources:
                continue

            s_rest = s_bone.rest_rotation
            
            # NEUTRAL FINGERS: If the source is in a curled pose, it 'cancels out' the motion.
            if neutral_fingers:
                is_finger = any(f in s_bone.name.lower() for f in ['index', 'middle', 'ring', 'pinky', 'thumb', 'toe'])
                if is_finger:
                    pname = s_bone.parent_name
                    p_bone = src_skel.get_bone_case_insensitive(pname)
                    if p_bone:
                        s_rest = p_bone.rest_rotation
            
            off = quaternion_multiply(quaternion_inverse(s_rest), t_bone.rest_rotation)
            active.append((s_bone, t_bone, off))
            mapped_targets.add(t_bone.name)
            mapped_sources.add(s_bone.name)
            
    # 2. Smart fuzzy matching for unmapped bones using advanced algorithm
    # Sort target bones: Hips/Root first, then limbs, then fingers
    def sort_key(b):
        n = b.name.lower()
        if 'hips' in n or 'root' in n or 'pelvis' in n: return 0
        if 'spine' in n or 'chest' in n: return 1
        if 'neck' in n or 'head' in n: return 2
        if 'leg' in n or 'arm' in n: return 3
        return 10  # Fingers and other bones last
    
    tgt_list = sorted(tgt_skel.bones.values(), key=sort_key)
    
    # Track matches with confidence scores for reporting
    fuzzy_matches = []
    
    for t_bone in tgt_list:
        if t_bone.name in mapped_targets:
            continue
        
        # Use the advanced bone matching system
        best_source_name, confidence = find_best_bone_match(
            t_bone.name,
            src_skel,
            mapped_sources,
            require_side_match=True
        )
        
        # Accept match if confidence is above threshold
        if best_source_name and confidence >= 0.5:  # 50% confidence minimum
            s_bone = src_skel.bones[best_source_name]
            
            s_rest = s_bone.rest_rotation
            if neutral_fingers:
                is_finger = any(f in s_bone.name.lower() for f in ['index', 'middle', 'ring', 'pinky', 'thumb', 'toe'])
                if is_finger:
                    pname = s_bone.parent_name
                    p_bone = src_skel.get_bone_case_insensitive(pname)
                    if p_bone:
                        s_rest = p_bone.rest_rotation

            off = quaternion_multiply(quaternion_inverse(s_rest), t_bone.rest_rotation)
            active.append((s_bone, t_bone, off))
            mapped_sources.add(s_bone.name)
            mapped_targets.add(t_bone.name)
            fuzzy_matches.append((s_bone.name, t_bone.name, confidence))
    
    # Print mapping results with confidence scores
    print(f"\n[Retarget] Bone Mapping Results:")
    print(f"  Total: {len(active)} bones mapped")
    if fuzzy_matches:
        print(f"  Fuzzy matched: {len(fuzzy_matches)} bones")
        for s_name, t_name, conf in fuzzy_matches:
            print(f"    {s_name} → {t_name} (confidence: {conf:.2f})")
            
    print(f"DEBUG: Final Mapping Count: {len(active)} bones")
    # Sort for cleaner debug output
    final_mappings = sorted(active, key=lambda x: x[1].name)
    for s, t, _ in final_mappings:
        print(f"  - {s.name} -> {t.name}")
            
    src_h = get_skeleton_height(src_skel, [])
    tgt_h = get_skeleton_height(tgt_skel, [])
    scale = force_scale if force_scale > 1e-4 else (tgt_h / src_h if src_h > 0.01 else 1.0)
    print(f"Scale: {scale:.4f}")
    
    tgt_world_anims = {}
    frames = range(src_skel.frame_start, src_skel.frame_end + 1)
    
    # 1. World Rotations
    for s_bone, t_bone, off in active:
        tgt_world_anims[t_bone.name] = {}
        for f in frames:
            s_rot = s_bone.world_animation.get(f, s_bone.rest_rotation)
            # Apply retargeting offset
            t_rot = quaternion_multiply(s_rot, off)
            # IMPORTANT: Apply global yaw offset to ALL bones to preserve target world space consistency
            if yaw_offset != 0:
                t_rot = quaternion_multiply(yaw_q, t_rot)
            tgt_world_anims[t_bone.name][f] = t_rot
            
        is_root = "hips" in t_bone.name.lower() or "pelvis" in s_bone.name.lower()
        if is_root:
            ret_locs[t_bone.name] = {}
            t_rest_world_pos = t_bone.world_matrix[3, :3]
            t_rest_loc = t_bone.local_matrix[3, :3]
            pname = t_bone.parent_name
            
            # Proxy bone logic: How does the target point move in source's space?
            # ROBUSTNESS: Convert Target Rest Position to Source units for internal math
            t_rest_source_units = t_rest_world_pos / scale
            
            s_rest_mat_inv = np.linalg.inv(s_bone.world_matrix)
            # Offset of Target Hips in Source Hips local space
            p_homog = np.append(t_rest_source_units, 1.0)
            p_local = (p_homog @ s_rest_mat_inv)[:3]
            
            for f in frames:
                s_q = s_bone.world_animation.get(f, s_bone.rest_rotation)
                s_p = s_bone.world_location_animation.get(f, s_bone.world_matrix[3, :3])
                
                # Proxy bone logic: v' = v * R + T (In Source Space)
                s_r = R.from_quat([s_q[1], s_q[2], s_q[3], s_q[0]]).as_matrix()
                p_world_f = p_local @ s_r.T + s_p
                # Displacement in SOURCE units
                disp = (p_world_f - t_rest_source_units)
                
                # IMPORTANT: Rotate displacement by the Rest-Pose Offset
                # This ensures that 'Forward' movement maps correctly even if rigs face different ways
                off_rot = R.from_quat([off[1], off[2], off[3], off[0]])
                disp = off_rot.apply(disp)
                
                disp_scaled = disp * scale
                
                # Apply global yaw offset
                if yaw_offset != 0:
                    rot_disp = R.from_quat([yaw_q[1], yaw_q[2], yaw_q[3], yaw_q[0]])
                    disp_scaled = rot_disp.apply(disp_scaled)
                
                # Convert to target parent space
                prot = tgt_world_anims.get(pname, {}).get(f)
                if prot is None:
                    prot = tgt_skel.node_rest_rotations.get(pname, np.array([1, 0, 0, 0]))
                    if yaw_offset != 0:
                        prot = quaternion_multiply(yaw_q, prot)
                p_rot_inv = R.from_quat([prot[1], prot[2], prot[3], prot[0]]).inv()
                local_disp = p_rot_inv.apply(disp_scaled)
                
                ret_locs[t_bone.name][f] = t_rest_loc + local_disp

    # 2. Local Rotations
    for s_bone, t_bone, _ in active:
        ret_rots[t_bone.name] = {}
        pname = t_bone.parent_name
        
        # We need the parent's TARGET world orientation at frame F
        # If the parent is NOT mapped, we use its REST orientation (or sampled if we had it)
        for f in frames:
            prot = tgt_world_anims.get(pname, {}).get(f)
            if prot is None:
                # Fallback: Check if target skeleton has this node's rest orientation
                prot = tgt_skel.node_rest_rotations.get(pname, np.array([1, 0, 0, 0]))
                if yaw_offset != 0:
                    prot = quaternion_multiply(yaw_q, prot)
                
            # Target world rotation to local space: Local = ParentWorld.inv @ World
            l_rot = quaternion_multiply(quaternion_inverse(prot), tgt_world_anims[t_bone.name][f])
            ret_rots[t_bone.name][f] = l_rot
            
    return ret_rots, ret_locs

def copy_textures_for_scene(scene, output_fbx_path):
    """Copy all texture files referenced by the scene to the output FBX location"""
    import shutil
    
    output_dir = os.path.dirname(os.path.abspath(output_fbx_path))
    fbx_filename = os.path.basename(output_fbx_path)
    fbx_base = os.path.splitext(fbx_filename)[0]
    
    # Create texture folder named after the FBX file
    texture_dir = os.path.join(output_dir, f"{fbx_base}_textures")
    
    copied_count = 0
    
    # Iterate through all materials in the scene
    for i in range(scene.GetMaterialCount()):
        material = scene.GetMaterial(i)
        
        # Check common material properties for textures
        texture_props = [
            FbxSurfaceMaterial.sDiffuse,
            FbxSurfaceMaterial.sNormalMap,
            FbxSurfaceMaterial.sSpecular,
            FbxSurfaceMaterial.sEmissive,
            FbxSurfaceMaterial.sBump,
            "DiffuseColor",
            "NormalMap",
            "SpecularColor",
        ]
        
        for prop_name in texture_props:
            prop = material.FindProperty(prop_name)
            if prop.IsValid():
                # Get texture count for this property  
                tex_count = prop.GetSrcObjectCount()
                for j in range(tex_count):
                    texture = prop.GetSrcObject(j)
                    # Check if it's a file texture (has GetFileName method)
                    if texture and hasattr(texture, 'GetFileName'):
                        original_path = texture.GetFileName()
                        if original_path and os.path.exists(original_path):
                            # Create texture directory if needed
                            os.makedirs(texture_dir, exist_ok=True)
                            
                            # Copy texture file
                            filename = os.path.basename(original_path)
                            dest_path = os.path.join(texture_dir, filename)
                            
                            if not os.path.exists(dest_path):
                                shutil.copy2(original_path, dest_path)
                                copied_count += 1
                                print(f"  Copied: {filename}")
                            
                            # Update texture path to be in the same directory as FBX
                            # This ensures the web browser can access it
                            relative_path = os.path.join(f"{fbx_base}_textures", filename)
                            texture.SetFileName(relative_path)
                            texture.SetRelativeFileName(relative_path)
    
    if copied_count > 0:
        print(f"[Retarget] Copied {copied_count} texture file(s) to {os.path.basename(texture_dir)}")
    
    return copied_count

def save_fbx(manager, scene, path):
    """Save FBX with materials and textures preserved (matching HY-Motion implementation)"""
    
    exporter = FbxExporter.Create(manager, "")
    
    # Get or create IO settings
    ios = manager.GetIOSettings()
    if not ios:
        ios = FbxIOSettings.Create(manager, "IOSRoot")
        manager.SetIOSettings(ios)
    
    # CRITICAL: Use the CORRECT FBX SDK constants (from HY-Motion's working code)
    # These will embed the materials and textures directly into the FBX file
    try:
        ios.SetBoolProp(EXP_FBX_EMBEDDED, True)  # Embed media
        ios.SetBoolProp(EXP_FBX_MATERIAL, True)   # Include materials
        ios.SetBoolProp(EXP_FBX_TEXTURE, True)    # Include textures
        print("[Retarget] Configured FBX export with embedded materials")
    except Exception as e:
        print(f"[Retarget] Warning: Could not set embedded export properties: {e}")
        print("[Retarget]  Trying fallback properties...")
        # Fallback to string-based properties (some FBX SDK versions use these)
        try:
            ios.SetBoolProp("Export|AdvOptGrp|Fbx|Material", True)
            ios.SetBoolProp("Export|AdvOptGrp|Fbx|Texture", True)
            ios.SetBoolProp("Export|AdvOptGrp|Fbx|Model", True)
            ios.SetBoolProp("Export|AdvOptGrp|Fbx|Animation", True)
            ios.SetBoolProp("Export|AdvOptGrp|Fbx|Shape", True)
            ios.SetBoolProp("Export|AdvOptGrp|Fbx|Skin", True)
            print("[Retarget]  Fallback properties set")
        except Exception as fallback_error:
            print(f"[Retarget]  Fallback also failed: {fallback_error}")
    
    # Initialize exporter
    file_format = manager.GetIOPluginRegistry().GetNativeWriterFormat()
    if not exporter.Initialize(path, file_format, ios):
        raise RuntimeError(f"Failed to initialize FBX exporter: {exporter.GetStatus().GetErrorString()}")
    
    # Export the scene
    if not exporter.Export(scene):
        raise RuntimeError(f"Failed to export FBX: {exporter.GetStatus().GetErrorString()}")
    
    exporter.Destroy()
    print(f"[Retarget] Saved FBX to: {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', '-s', required=True)
    parser.add_argument('--target', '-t', required=True)
    parser.add_argument('--mapping', '-m', default='', help='Optional bone mapping file (uses hardcoded mappings if not provided)')
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--yaw', '-y', type=float, default=0.0)
    parser.add_argument('--scale', '-sc', type=float, default=0.0)
    parser.add_argument('--no-neutral', dest='neutral', action='store_false', help="Disable neutral finger rest-pose")
    parser.set_defaults(neutral=True)
    args = parser.parse_args()
    
    mapping = load_bone_mapping(args.mapping)
    if args.source.lower().endswith('.npz'):
        print(f"Loading NPZ Source: {args.source}")
        src_man, src_scene = None, None
        src_skel = load_npz(args.source)
    else:
        # Sampling characters often use frame 0 as bind, but real FBX files have a Bind Pose 
        # reachable by setting current animation stack to None.
        src_man, src_scene, src_skel = load_fbx(args.source, sample_rest_frame=None)
        src_h = get_skeleton_height(src_skel, [])
        # FALLBACK: If Bind Pose is collapsed (height ~0), try frame 0
        if src_h < 0.1:
            print("DEBUG: Bind Pose collapsed, falling back to frame 0 for rest pose.")
            src_man, src_scene, _ = load_fbx(args.source, sample_rest_frame=0)
            # Refresh skeleton with sampled data
            src_skel = Skeleton(os.path.basename(args.source))
            collect_skeleton_nodes(src_scene.GetRootNode(), src_skel, sampling_time=FbxTime())
        
        extract_animation(src_scene, src_skel)
    
    tgt_man, tgt_scene, tgt_skel = load_fbx(args.target)
    
    rots, locs = retarget_animation(src_skel, tgt_skel, mapping, args.scale, args.yaw, args.neutral)
    
    src_time_mode = src_scene.GetGlobalSettings().GetTimeMode() if src_scene else tgt_scene.GetGlobalSettings().GetTimeMode()
    apply_retargeted_animation(tgt_scene, tgt_skel, rots, locs, src_skel.frame_start, src_skel.frame_end, src_time_mode)
    
    save_fbx(tgt_man, tgt_scene, args.output)
    print("Done!")

if __name__ == "__main__":
    main()

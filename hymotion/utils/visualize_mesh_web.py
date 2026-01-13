import json
import os
import re
import threading
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch import Tensor

_FILE_ACCESS_LOCK = threading.Lock()

# Template directory path
_TEMPLATE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "scripts", "gradio", "templates"
)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal attacks
    Args:
        filename: original filename
    Returns:
        sanitized filename
    """
    if not filename:
        return ""

    # remove all path traversal characters
    filename = re.sub(r"\.\.(/|\\\\\\)?", "", filename)
    filename = filename.strip("./\\")

    # only allow letters, numbers, underscores, hyphens and dots
    # dots are only allowed once in the extension
    filename = re.sub(r"[^a-zA-Z0-9_.-]", "", filename)

    # prevent multiple consecutive dots
    while ".." in filename:
        filename = filename.replace("..", ".")

    # prevent starting with a dot (hidden file)
    if filename.startswith("."):
        filename = filename[1:]

    # limit file name length
    if len(filename) > 255:
        filename = filename[:255]

    return filename


def sanitize_folder_name(folder_name: str) -> str:
    """
    Sanitize folder name to prevent path traversal attacks
    Args:
        folder_name: original folder name
    Returns:
        sanitized folder name
    """
    if not folder_name:
        return "output"  # default folder

    # remove all path traversal characters
    folder_name = re.sub(r"\.\.(/|\\\\\\)?", "", folder_name)
    folder_name = folder_name.strip("./\\")

    # only allow letters, numbers, underscores, hyphens and slashes (for subdirectories)
    # but need to ensure slashes don't cause path traversal
    folder_name = re.sub(r"[^a-zA-Z0-9_./-]", "", folder_name)

    # split path and clean each part
    parts = folder_name.split("/")
    cleaned_parts = []
    for part in parts:
        if part and part not in [".", ".."]:
            # clean each part
            part = re.sub(r"[^a-zA-Z0-9_-]", "", part)
            if part:
                cleaned_parts.append(part)

    # recombine, allow at most 3 levels of directory depth
    if len(cleaned_parts) > 3:
        cleaned_parts = cleaned_parts[:3]

    return "/".join(cleaned_parts) if cleaned_parts else "output"


def safe_path_join(base_dir: str, *paths: str) -> str:
    """
    Safe path joining, ensure the resulting path is within base_dir
    Args:
        base_dir: base directory
        *paths: paths to join
    Returns:
        joined path
    Raises:
        ValueError: if path traversal is detected
    """
    # clean all paths
    cleaned_paths = []
    for path in paths:
        if path:
            # clean each path part
            path = re.sub(r"\.\.(/|\\\\\\)?", "", path)
            path = path.strip("./\\")
            path = re.sub(r"[^a-zA-Z0-9_.-]", "", path)
            if path:
                cleaned_paths.append(path)

    # join paths
    full_path = os.path.join(base_dir, *cleaned_paths)

    # ensure the resulting path is within base_dir
    base_dir = os.path.realpath(base_dir)
    full_path = os.path.realpath(os.path.normpath(full_path))

    if os.path.commonpath([base_dir, full_path]) != base_dir:
        raise ValueError(f"Path traversal detected: {full_path} is outside {base_dir}")

    return full_path


def _get_root_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def get_output_dir(sub_path: str = "") -> str:
    output_base = _get_root_dir()
    if not os.path.exists(output_base):
        os.makedirs(output_base, exist_ok=True)
    if sub_path:
        parts = [p for p in sub_path.replace("\\", "/").split("/") if p]
    else:
        parts = []
    return safe_path_join(output_base, *parts)


def save_visualization_data(
    output: Dict[str, Union[Tensor, list[str]]],
    text: str,
    rewritten_text: Union[str, list[str]],
    timestamp: str,
    output_dir: Optional[str] = None,
    output_filename: Optional[str] = None,
):
    from ..pipeline.body_model import construct_smpl_data_dict

    if output_dir is None:
        output_dir = get_output_dir(sub_path="output/gradio")
    os.makedirs(output_dir, exist_ok=True)

    # for metadata
    base_filename = output_filename if output_filename else timestamp
    meta_path = safe_path_join(output_dir, f"{base_filename}_meta.json")
    if isinstance(rewritten_text, str):
        rewritten_text = [rewritten_text]
    batch_size = output["rot6d"].shape[0]
    meta_data = {
        "timestamp": timestamp,
        "text": text,
        "text_rewrite": rewritten_text,
        "num_samples": batch_size,
        "base_filename": base_filename,
    }

    with _FILE_ACCESS_LOCK:
        with open(meta_path, "w") as f:
            json.dump(meta_data, f, indent=2)

    # for smpl data
    rot6d = output["rot6d"]
    transl = output["transl"]

    all_smpl_data = []  # for FBX generator

    for bb in range(batch_size):
        # build data
        smpl_data = construct_smpl_data_dict(rot6d[bb].clone(), transl[bb].clone())
        all_smpl_data.append(smpl_data)

        # prepare dictionary to save into NPZ
        npz_dict = {}
        npz_dict["gender"] = np.array([smpl_data.get("gender", "neutral")], dtype=str)

        for key in ["Rh", "trans", "poses", "betas"]:
            if key in smpl_data:
                val = smpl_data[key]
                if isinstance(val, (list, tuple)):
                    val = np.array(val)
                elif isinstance(val, torch.Tensor):
                    val = val.cpu().numpy()
                npz_dict[key] = val

        # save single NPZ
        sample_filename = f"{base_filename}_{bb:03d}.npz"
        sample_path = safe_path_join(output_dir, sample_filename)

        with _FILE_ACCESS_LOCK:
            np.savez_compressed(sample_path, **npz_dict)

    # construct memory dictionary to return (for compatibility)
    memory_data = {
        "timestamp": timestamp,
        "text": text,
        "text_rewrite": rewritten_text,
        "smpl_data": all_smpl_data,
        "meta_data": [],
        "rot6d": rot6d,  # Add rot6d format data for FBX retargeting
        "transl": transl,
    }

    # return base filename, subsequent logic will use this as a basis for finding _meta.json or _000.npz
    return memory_data, base_filename


def get_cached_captions(folder_name: str, file_name: str) -> List[dict]:
    """read _meta.json to get text"""

    folder_name = sanitize_folder_name(folder_name)
    file_name = sanitize_filename(file_name)

    base_dir = get_output_dir(folder_name)
    # try to add suffix or find
    meta_path = safe_path_join(base_dir, f"{file_name}_meta.json")

    if not os.path.exists(meta_path):
        if "_" in file_name:
            prefix = file_name.rsplit("_", 1)[0]
            prefix = sanitize_filename(prefix)
            meta_path_alt = safe_path_join(base_dir, f"{prefix}_meta.json")
            if os.path.exists(meta_path_alt):
                meta_path = meta_path_alt
            else:
                return []
        else:
            return []

    try:
        with _FILE_ACCESS_LOCK:
            with open(meta_path, "r") as f:
                data = json.load(f)

        text = data.get("text", "")
        text_rewrite = data.get("text_rewrite", [])

        captions = []
        for i, t in enumerate(text_rewrite):
            item = {"short caption+": f"{t}", "start_time": None, "end_time": None}
            if text and text != t:
                item["short caption"] = text
            captions.append(item)
        return captions
    except Exception as e:
        print(f"Error reading meta json: {e}")
        return []


def get_cached_smpl_frames(folder_name: str, file_name: str) -> List[list]:
    """
    read logic needs to be adjusted:
    1. if file_name is the base name, load all samples
    2. if file_name is a specific sample name, only load that sample
    """
    folder_name = sanitize_folder_name(folder_name)
    file_name = sanitize_filename(file_name)

    base_dir = get_output_dir(folder_name)

    npz_direct_path = safe_path_join(base_dir, f"{file_name}.npz")
    meta_path = safe_path_join(base_dir, f"{file_name}_meta.json")

    target_indices = []
    base_name = file_name

    if os.path.isfile(npz_direct_path):
        try:
            if "_" in file_name:
                prefix, suffix = file_name.rsplit("_", 1)
                if suffix.isdigit():
                    num_samples = 1
                    base_name = prefix
                    target_indices = [int(suffix)]
                else:
                    pass
            else:
                pass
        except ValueError:
            pass
        if not target_indices:
            return []
    elif os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
                num_samples = meta.get("num_samples", 0)
                target_indices = range(num_samples)
        except Exception as e:
            print(f"Error reading meta: {e}")
            return []
    else:
        return []

    all_people = []

    for i in target_indices:
        npz_path = safe_path_join(base_dir, f"{base_name}_{i:03d}.npz")
        if not os.path.exists(npz_path):
            continue

        try:
            with _FILE_ACCESS_LOCK:
                with np.load(npz_path, allow_pickle=False) as data:
                    # read single person data
                    gender = str(data["gender"][0])
                    Rh = data["Rh"]
                    Th = data["trans"]
                    poses = data["poses"]
                    betas = data["betas"]

                    if poses.ndim == 3:
                        poses = poses.reshape(poses.shape[0], -1)

                    person_frames = []
                    for f in range(len(poses)):
                        frame = {
                            "id": i,
                            "gender": gender,
                            "Rh": Rh[f : f + 1].tolist(),
                            "Th": Th[f : f + 1].tolist(),
                            "poses": poses[f : f + 1].tolist(),
                            "shapes": betas.tolist(),
                        }
                        person_frames.append([frame])
                    all_people.append(person_frames)
        except Exception as e:
            print(f"Error loading {npz_path}: {e}")

    # merge
    combined_frames = []
    max_frames = max(len(p) for p in all_people) if all_people else 0
    for f_idx in range(max_frames):
        frame_content = []
        for person_seq in all_people:
            if f_idx < len(person_seq):
                frame_content.extend(person_seq[f_idx])
        combined_frames.append(frame_content)

    return combined_frames


def generate_static_html_content(
    folder_name: str,
    file_name: str,
    hide_captions: bool = False,
) -> str:
    """
    Generate static HTML content with embedded SMPL data and captions.
    All JavaScript code is embedded directly in the HTML template,
    so no external static resources are needed.

    Args:
        folder_name: The folder name containing the NPZ/meta files
        file_name: The base file name (without extension)
        hide_captions: Whether to hide captions in the visualization

    Returns:
        The HTML content as a string
    """
    # Load SMPL data
    smpl_frames = get_cached_smpl_frames(folder_name, file_name)
    if not smpl_frames:
        raise ValueError(f"No SMPL data found for {folder_name}/{file_name}")

    # Load captions
    captions = []
    if not hide_captions:
        captions = get_cached_captions(folder_name, file_name)

    # Generate caption HTML
    caption_html = _generate_caption_html(captions, hide_captions)

    # Convert SMPL data to JSON
    smpl_data_json = json.dumps(smpl_frames, ensure_ascii=False)

    # Load template
    template_path = os.path.join(_TEMPLATE_DIR, "index_wooden_static.html")
    with open(template_path, "r", encoding="utf-8") as f:
        template_content = f.read()

    # Replace placeholders with actual data
    html_content = template_content.replace("{{ smpl_data_json }}", smpl_data_json)
    html_content = html_content.replace("{{ caption_html }}", caption_html)

    print(f">>> Generated static HTML content for {folder_name}/{file_name}")
    return html_content


def generate_static_html(
    folder_name: str,
    file_name: str,
    output_dir: str,
    hide_captions: bool = False,
) -> str:
    """
    Generate a static HTML file with embedded SMPL data and captions.
    All JavaScript code is embedded directly in the HTML template,
    so no external static resources are needed.

    Args:
        folder_name: The folder name containing the NPZ/meta files
        file_name: The base file name (without extension)
        output_dir: Directory to save the generated HTML file
        hide_captions: Whether to hide captions in the visualization

    Returns:
        The path to the generated HTML file
    """
    html_content = generate_static_html_content(folder_name, file_name, hide_captions)

    # Generate output path
    os.makedirs(output_dir, exist_ok=True)
    output_html_path = os.path.join(output_dir, f"{file_name}_vis.html")

    # Write HTML file
    with _FILE_ACCESS_LOCK:
        with open(output_html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    print(f">>> Generated static HTML: {output_html_path}")
    return output_html_path


def _generate_caption_html(captions: List[dict], hide_captions: bool = False) -> str:
    """
    Generate the caption overlay HTML.

    Args:
        captions: List of caption dictionaries
        hide_captions: Whether to hide captions

    Returns:
        HTML string for caption overlay
    """
    if hide_captions or not captions:
        return ""

    caption_items = []
    for caption in captions:
        # Get the display text (prefer rewritten text)
        text = caption.get("short caption+") or caption.get("short caption") or "No caption"
        caption_items.append(f'<div class="caption-item">{text}</div>')

    captions_html = "\n".join(caption_items)

    return f'''
    <div class="caption-overlay">
        <div class="motion-info" id="motion-info">
            <div class="captions-section">
                {captions_html}
            </div>
        </div>
    </div>
    '''

import argparse
import codecs as cs
import json
import os
import os.path as osp
import random
import re
import textwrap
from typing import List, Optional, Tuple, Union

import torch
from huggingface_hub import snapshot_download

import gradio as gr


def try_to_download_model():
    repo_id = "tencent/HY-Motion-1.0"
    target_folder = "HY-Motion-1.0-Lite"
    print(f">>> start download ", repo_id, target_folder)
    local_dir = snapshot_download(repo_id=repo_id, allow_patterns=f"{target_folder}/*", local_dir="./ckpts/tencent")
    final_model_path = os.path.join(local_dir, target_folder)
    print(f">>> Final model path: {final_model_path}")
    return final_model_path


# Import spaces for Hugging Face Zero GPU support
try:
    import spaces

    SPACES_AVAILABLE = True
except ImportError:
    SPACES_AVAILABLE = False

    # Create a dummy decorator when spaces is not available
    class spaces:
        @staticmethod
        def GPU(func=None, duration=None):
            def decorator(fn):
                return fn

            if func is not None:
                return func
            return decorator


from hymotion.utils.t2m_runtime import T2MRuntime

NUM_WORKERS = torch.cuda.device_count() if torch.cuda.is_available() else 1

# Global runtime instance for Zero GPU lazy loading
_global_runtime = None
_global_args = None


def _init_runtime_if_needed():
    """Initialize runtime lazily for Zero GPU support."""
    global _global_runtime, _global_args
    if _global_runtime is not None:
        return _global_runtime

    if _global_args is None:
        raise RuntimeError("Runtime args not set. Call set_runtime_args() first.")

    args = _global_args
    cfg = osp.join(args.model_path, "config.yml")
    ckpt = osp.join(args.model_path, "latest.ckpt")

    skip_model_loading = False
    if not os.path.exists(ckpt):
        print(f">>> [WARNING] Checkpoint file not found: {ckpt}")
        print(f">>> [WARNING] Model loading will be skipped. Motion generation will not be available.")
        skip_model_loading = True

    print(">>> Initializing T2MRuntime...")
    if "USE_HF_MODELS" not in os.environ:
        os.environ["USE_HF_MODELS"] = "1"

    skip_text = False
    _global_runtime = T2MRuntime(
        config_path=cfg,
        ckpt_name=ckpt,
        skip_text=skip_text,
        device_ids=None,
        skip_model_loading=skip_model_loading,
        disable_prompt_engineering=args.disable_prompt_engineering,
        prompt_engineering_host=args.prompt_engineering_host,
        prompt_engineering_model_path=args.prompt_engineering_model_path,
    )
    return _global_runtime


@spaces.GPU(duration=120)
def generate_motion_on_gpu(
    text: str,
    seeds_csv: str,
    motion_duration: float,
    cfg_scale: float,
    output_format: str,
    original_text: str,
    output_dir: str,
) -> Tuple[str, List[str]]:
    """
    GPU-decorated function for motion generation.
    This function will request GPU allocation on Hugging Face Zero GPU.
    """
    runtime = _init_runtime_if_needed()

    html_content, fbx_files, _ = runtime.generate_motion(
        text=text,
        seeds_csv=seeds_csv,
        duration=motion_duration,
        cfg_scale=cfg_scale,
        output_format=output_format,
        original_text=original_text,
        output_dir=output_dir,
    )
    return html_content, fbx_files


# define data sources
DATA_SOURCES = {
    "example_prompts": "examples/example_prompts/example_subset.json",
}

# create interface
APP_CSS = """
    :root{
    --primary-start:#667eea; --primary-end:#764ba2;
    --secondary-start:#4facfe; --secondary-end:#00f2fe;
    --accent-start:#f093fb; --accent-end:#f5576c;
    --page-bg:linear-gradient(135deg,#f5f7fa 0%,#c3cfe2 100%);
    --card-bg:linear-gradient(135deg,#ffffff 0%,#f8f9fa 100%);
    --radius:12px;
    --iframe-bg:#ffffff;
    }

    /* Dark mode variables */
    [data-theme="dark"], .dark {
    --page-bg:linear-gradient(135deg,#1a1a1a 0%,#2d3748 100%);
    --card-bg:linear-gradient(135deg,#2d3748 0%,#374151 100%);
    --text-primary:#f7fafc;
    --text-secondary:#e2e8f0;
    --border-color:#4a5568;
    --input-bg:#374151;
    --input-border:#4a5568;
    --iframe-bg:#1a1a2e;
    }

    /* Page and card */
    .gradio-container{
    background:var(--page-bg) !important;
    min-height:100vh !important;
    color:var(--text-primary, #333) !important;
    }

    .main-header{
    background:transparent !important; border:none !important; box-shadow:none !important;
    padding:0 !important; margin:10px 0 16px !important;
    text-align:center !important;
    }

    .main-header h1, .main-header p, .main-header li {
    color:var(--text-primary, #333) !important;
    }

    .left-panel,.right-panel{
    background:var(--card-bg) !important;
    border:1px solid var(--border-color, #e9ecef) !important;
    border-radius:15px !important;
    box-shadow:0 4px 20px rgba(0,0,0,.08) !important;
    padding:24px !important;
    }

    .gradio-accordion{
    border:1px solid var(--border-color, #e1e5e9) !important;
    border-radius:var(--radius) !important;
    margin:12px 0 !important; background:transparent !important;
    }

    .gradio-accordion summary{
    background:transparent !important;
    padding:14px 18px !important;
    font-weight:600 !important;
    color:var(--text-primary, #495057) !important;
    }

    .gradio-group{
    background:transparent !important; border:none !important;
    border-radius:8px !important; padding:12px 0 !important; margin:8px 0 !important;
    }

    /* Input class style - dark mode adaptation */
    .gradio-textbox input,.gradio-textbox textarea,.gradio-dropdown .wrap{
    border-radius:8px !important;
    border:2px solid var(--input-border, #e9ecef) !important;
    background:var(--input-bg, #fff) !important;
    color:var(--text-primary, #333) !important;
    transition:.2s all !important;
    }

    .gradio-textbox input:focus,.gradio-textbox textarea:focus,.gradio-dropdown .wrap:focus-within{
    border-color:var(--primary-start) !important;
    box-shadow:0 0 0 3px rgba(102,126,234,.1) !important;
    }

    .gradio-slider input[type="range"]{
    background:linear-gradient(to right,var(--primary-start),var(--primary-end)) !important;
    border-radius:10px !important;
    }

    .gradio-checkbox input[type="checkbox"]{
    border-radius:4px !important;
    border:2px solid var(--input-border, #e9ecef) !important;
    transition:.2s all !important;
    }

    .gradio-checkbox input[type="checkbox"]:checked{
    background:linear-gradient(45deg,var(--primary-start),var(--primary-end)) !important;
    border-color:var(--primary-start) !important;
    }

    /* Label text color adaptation */
    .gradio-textbox label, .gradio-dropdown label, .gradio-slider label,
    .gradio-checkbox label, .gradio-html label {
    color:var(--text-primary, #333) !important;
    }

    .gradio-textbox .info, .gradio-dropdown .info, .gradio-slider .info,
    .gradio-checkbox .info {
    color:var(--text-secondary, #666) !important;
    }

    /* Status information - dark mode adaptation */
    .gradio-textbox[data-testid*="状态信息"] input{
    background:var(--input-bg, linear-gradient(135deg,#f8f9fa 0%,#e9ecef 100%)) !important;
    border:2px solid var(--input-border, #dee2e6) !important;
    color:var(--text-primary, #495057) !important;
    font-weight:500 !important;
    }

    /* Button base class and variant */
    .generate-button,.rewrite-button,.dice-button{
    border:none !important; color:#fff !important; font-weight:600 !important;
    border-radius:8px !important; transition:.3s all !important;
    box-shadow:0 4px 15px rgba(0,0,0,.12) !important;
    }

    .generate-button{ background:linear-gradient(45deg,var(--primary-start),var(--primary-end)) !important; }
    .rewrite-button{ background:linear-gradient(45deg,var(--secondary-start),var(--secondary-end)) !important; }
    .dice-button{
    background:linear-gradient(45deg,var(--accent-start),var(--accent-end)) !important;
    height:40px !important;
    }

    .generate-button:hover,.rewrite-button:hover{ transform:translateY(-2px) !important; }
    .dice-button:hover{
    transform:scale(1.05) !important;
    box-shadow:0 4px 12px rgba(240,147,251,.28) !important;
    }

    .dice-container{
    display:flex !important;
    align-items:flex-end !important;
    justify-content:center !important;
    }

    /* Right panel clipping overflow, avoid double scrollbars */
    .right-panel{
    background:var(--card-bg) !important;
    border:1px solid var(--border-color, #e9ecef) !important;
    border-radius:15px !important;
    box-shadow:0 4px 20px rgba(0,0,0,.08) !important;
    padding:24px !important; overflow:hidden !important;
    }

    /* Main content row - ensure equal heights */
    .main-row {
    display: flex !important;
    align-items: stretch !important;
    }

    /* Flask area - match left panel height */
    .flask-display{
    padding:0 !important; margin:0 !important; border:none !important;
    box-shadow:none !important; background:var(--iframe-bg) !important;
    border-radius:10px !important; position:relative !important;
    height:100% !important; min-height:750px !important;
    display:flex !important; flex-direction:column !important;
    }

    .flask-display iframe{
    width:100% !important; flex:1 !important; min-height:750px !important;
    border:none !important; border-radius:10px !important; display:block !important;
    background:var(--iframe-bg) !important;
    }

    /* Right panel should stretch to match left panel */
    .right-panel{
    background:var(--card-bg) !important;
    border:1px solid var(--border-color, #e9ecef) !important;
    border-radius:15px !important;
    box-shadow:0 4px 20px rgba(0,0,0,.08) !important;
    padding:24px !important; overflow:hidden !important;
    display:flex !important; flex-direction:column !important;
    }

    /* Ensure dropdown menu is visible in dark mode */
    [data-theme="dark"] .gradio-dropdown .wrap,
    .dark .gradio-dropdown .wrap {
    background:var(--input-bg) !important;
    color:var(--text-primary) !important;
    }

    [data-theme="dark"] .gradio-dropdown .option,
    .dark .gradio-dropdown .option {
    background:var(--input-bg) !important;
    color:var(--text-primary) !important;
    }

    [data-theme="dark"] .gradio-dropdown .option:hover,
    .dark .gradio-dropdown .option:hover {
    background:var(--border-color) !important;
    }

    .footer{
    text-align:center !important;
    margin-top:20px !important;
    padding:10px !important;
    color:var(--text-secondary, #666) !important;
    }
"""

HEADER_BASE_MD = "# HY-Motion-1.0: Text-to-Motion Playground"

FOOTER_MD = "*This is a Beta version, any issues or feedback are welcome!*"

HTML_OUTPUT_PLACEHOLDER = """
<div style='height: 750px; width: 100%; border-radius: 8px; border-color: #e5e7eb; border-style: solid; border-width: 1px; display: flex; justify-content: center; align-items: center;'>
    <div style='text-align: center; font-size: 16px; color: #6b7280;'>
        <p style="color: #8d8d8d;">Welcome to HY-Motion-1.0!</p>
        <p style="color: #8d8d8d;">No motion visualization here yet.</p>
    </div>
</div>
"""


def load_examples_from_txt(txt_path: str, example_record_fps=30, max_duration=12):
    """Load examples from txt file."""

    def _parse_line(line: str) -> Optional[Tuple[str, float]]:
        line = line.strip()
        if line and not line.startswith("#"):
            parts = line.split("#")
            if len(parts) >= 2:
                text = parts[0].strip()
                duration = int(parts[1]) / example_record_fps
                duration = min(duration, max_duration)
            else:
                text = line.strip()
                duration = 5.0
            return text, duration
        return None

    examples: List[Tuple[str, float]] = []
    if os.path.exists(txt_path):
        try:
            if txt_path.endswith(".txt"):
                with cs.open(txt_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines:
                        result = _parse_line(line)
                        if result is None:
                            continue
                        text, duration = result
                        examples.append((text, duration))
            elif txt_path.endswith(".json"):
                with cs.open(txt_path, "r", encoding="utf-8") as f:
                    lines = json.load(f)
                    for key, value in lines.items():
                        if "_raw_chn" in key or "GENERATE_PROMPT_FORMAT" in key:
                            continue
                        for line in value:
                            result = _parse_line(line)
                            if result is None:
                                continue
                            text, duration = result
                            examples.append((text, duration))
            print(f">>> Loaded {len(examples)} examples from {txt_path}")
        except Exception as e:
            print(f">>> Failed to load examples from {txt_path}: {e}")
    else:
        print(f">>> Examples file not found: {txt_path}")

    return examples


class T2MGradioUI:
    def __init__(self, runtime: T2MRuntime, args: argparse.Namespace):
        self.runtime = runtime
        self.args = args

        # Check if rewrite is available:
        # - Either prompt_engineering_host is provided (use remote API)
        # - Or local prompter model exists (use local model)
        print(f">>> args: {vars(args)}")

        has_remote_host = args.prompt_engineering_host is not None and args.prompt_engineering_host.strip() != ""

        # Check if local prompter model exists
        local_prompter_path = "./ckpts/Text2MotionPrompter"
        has_local_prompter = os.path.exists(local_prompter_path) and os.path.isdir(local_prompter_path)

        if has_local_prompter:
            print(f">>> Local prompter model found at: {local_prompter_path}")

        self.prompt_engineering_available = (
            has_remote_host or has_local_prompter
        ) and not args.disable_prompt_engineering

        print(
            f">>> Prompt engineering available: {self.prompt_engineering_available} (remote: {has_remote_host}, local: {has_local_prompter})"
        )

        self.all_example_data = {}
        self._init_example_data()

    def _init_example_data(self):
        for source_name, file_path in DATA_SOURCES.items():
            examples = load_examples_from_txt(file_path)
            if examples:
                self.all_example_data[source_name] = examples
            else:
                # provide default examples as fallback
                self.all_example_data[source_name] = [
                    ("Twist at the waist and punch across the body.", 3.0),
                    ("A person is running then takes big leap.", 3.0),
                    ("A person holds a railing and walks down a set of stairs.", 5.0),
                    (
                        "A man performs a fluid and rhythmic hip-hop style dance, incorporating body waves, arm gestures, and side steps.",
                        5.0,
                    ),
                ]
        print(f">>> Loaded data sources: {list(self.all_example_data.keys())}")

    def _get_header_text(self):
        return HEADER_BASE_MD

    def _generate_random_seeds(self):
        seeds = [random.randint(0, 999) for _ in range(4)]
        return ",".join(map(str, seeds))

    def _prompt_engineering(
        self, text: str, duration: float, enable_rewrite: bool = True, enable_duration_est: bool = True
    ):
        if not text.strip():
            return "", gr.update(interactive=False), gr.update()

        call_llm = enable_rewrite or enable_duration_est
        if not call_llm:
            print(f"\t>>> Using original duration and original text...")
            predicted_duration = duration
            rewritten_text = text
        else:
            print(f"\t>>> Using LLM to estimate duration/rewrite text...")
            try:
                predicted_duration, rewritten_text = self.runtime.rewrite_text_and_infer_time(text=text)
            except Exception as e:
                print(f"\t>>> Text rewriting/duration prediction failed: {e}")
                return (
                    f"❌ Text rewriting/duration prediction failed: {str(e)}",
                    gr.update(interactive=False),
                    gr.update(),
                )
            if not enable_rewrite:
                rewritten_text = text
            if not enable_duration_est:
                predicted_duration = duration

        return rewritten_text, gr.update(interactive=True), gr.update(value=predicted_duration)

    def _generate_motion(
        self,
        original_text: str,
        rewritten_text: str,
        seed_input: str,
        duration: float,
        cfg_scale: float,
        custom_fbx_path: Optional[str] = None,
    ) -> Tuple[str, List[str]]:
        # When rewrite is not available, use original_text directly
        if not self.prompt_engineering_available:
            text_to_use = original_text.strip()
            if not text_to_use:
                return "Error: Input text is empty, please enter text first", []
        else:
            text_to_use = rewritten_text.strip()
            if not text_to_use:
                return "Error: Rewritten text is empty, please rewrite the text first", []

        try:
            # Use runtime from global if available (for Zero GPU), otherwise use self.runtime
            runtime = _global_runtime if _global_runtime is not None else self.runtime
            fbx_ok = getattr(runtime, "fbx_available", False)
            req_format = "fbx" if fbx_ok else "dict"

            # Use GPU-decorated function for Zero GPU support
            html_content, fbx_files = generate_motion_on_gpu(
                text=text_to_use,
                seeds_csv=seed_input,
                motion_duration=duration,
                cfg_scale=cfg_scale,
                output_format=req_format,
                original_text=original_text,
                output_dir=self.args.output_dir,
            )
            
            # If custom FBX is provided, retarget the generated motion to it
            preview_html = None
            if custom_fbx_path and fbx_ok:
                print(f">>> Starting custom FBX retargeting: {custom_fbx_path}")
                fbx_files, preview_html = self._retarget_to_custom_fbx(fbx_files, custom_fbx_path)
            
            # Escape HTML content for srcdoc attribute
            escaped_html = html_content.replace('"', "&quot;")
            # Return iframe with srcdoc - directly embed HTML content
            iframe_html = f"""
                <iframe
                    srcdoc="{escaped_html}"
                    width="100%"
                    height="750px"
                    style="border: none; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);"
                ></iframe>
            """
            
            # If we have 3D preview, return it as well
            if preview_html:
                preview_iframe = f"""
                    <iframe
                        srcdoc="{preview_html.replace('"', '&quot;')}"
                        width="100%"
                        height="750px"
                        style="border: none; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); margin-top: 10px;"
                    ></iframe>
                """
                iframe_html = iframe_html + preview_iframe
            
            return iframe_html, fbx_files
        except Exception as e:
            print(f"\t>>> Motion generation failed: {e}")
            import traceback
            traceback.print_exc()
            return (
                f"❌ Motion generation failed: {str(e)}\n\nPlease check the input parameters or try again later",
                [],
            )
    
    def _retarget_to_custom_fbx(self, original_fbx_files, custom_fbx_path):
        """Retarget generated motion to custom FBX skeleton."""
        from hymotion.utils.fbx_export import export_motion_to_fbx, load_motion_from_npz
        
        retargeted_files = []
        
        # Process each NPZ file (find matching NPZ for each FBX)
        for fbx_file in original_fbx_files:
            if not fbx_file.endswith('.fbx'):
                continue
                
            # Find corresponding NPZ file (same base name)
            npz_file = fbx_file.replace('.fbx', '.npz')
            if not os.path.exists(npz_file):
                print(f"Warning: NPZ file not found for {fbx_file}")
                continue
            
            try:
                # Load motion data from NPZ
                motion_data = load_motion_from_npz(npz_file)
                
                # Generate output filename
                base_name = os.path.basename(fbx_file).replace('.fbx', '')
                output_fbx = os.path.join(
                    os.path.dirname(fbx_file),
                    f"retargeted_{base_name}.fbx"
                )
                
                # Get text description if available
                txt_file = fbx_file.replace('.fbx', '.txt')
                text_description = ""
                if os.path.exists(txt_file):
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        text_description = f.read().strip()
                
                # Export with retargeting
                success = export_motion_to_fbx(
                    motion_data=motion_data,
                    output_path=output_fbx,
                    text_description=text_description,
                    template_fbx=custom_fbx_path,
                )
                
                if success:
                    retargeted_files.append(output_fbx)
                    # Copy text description to retargeted file
                    if text_description:
                        retargeted_txt = output_fbx.replace('.fbx', '.txt')
                        with open(retargeted_txt, 'w', encoding='utf-8') as f:
                            f.write(text_description)
                        retargeted_files.append(retargeted_txt)
                    print(f">>> Retargeted: {os.path.basename(output_fbx)}")
                else:
                    print(f">>> Failed to retarget: {os.path.basename(fbx_file)}")
                    
            except Exception as e:
                print(f">>> Error retargeting {fbx_file}: {e}")
                import traceback
                traceback.print_exc()
        
        if retargeted_files:
            print(f">>> Retargeting complete: {len(retargeted_files)} files")
            combined_files = original_fbx_files + retargeted_files
            
            # Try to generate 3D preview for all FBX files
            try:
                preview_html = self._generate_3d_preview(combined_files)
            except Exception as e:
                print(f">>> 3D preview generation failed: {e}")
                preview_html = None
            
            return combined_files, preview_html
        else:
            print(">>> No retargeted files created")
            return original_fbx_files, None
    
    def _generate_3d_preview(self, fbx_files):
        """Generate 3D preview HTML for multiple FBX files with spacing."""
        # Filter only FBX files
        fbx_only = [f for f in fbx_files if f.endswith('.fbx')]
        
        if not fbx_only:
            return None
        
        print(f">>> Generating 3D preview for {len(fbx_only)} FBX files...")
        
        # Read FBX files and encode as base64
        import base64
        fbx_data_list = []
        for i, fbx_file in enumerate(fbx_only):
            try:
                with open(fbx_file, 'rb') as f:
                    fbx_data = base64.b64encode(f.read()).decode('utf-8')
                    fbx_data_list.append({
                        'data': fbx_data,
                        'name': os.path.basename(fbx_file),
                        'x_offset': i * 1.0  # Space each model 1 unit apart on X-axis
                    })
            except Exception as e:
                print(f">>> Failed to read {fbx_file}: {e}")
        
        if not fbx_data_list:
            return None
        
        # Generate Three.js HTML with multiple models
        html_content = self._create_multi_model_viewer_html(fbx_data_list)
        
        return html_content
    
    def _create_multi_model_viewer_html(self, fbx_data_list):
        """Create Three.js HTML viewer for multiple FBX models."""
        import json
        
        fbx_json = json.dumps(fbx_data_list)
        
        html = f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ margin: 0; overflow: hidden; background: #1a1a1a; }}
        #canvas {{ width: 100%; height: 100vh; display: block; }}
        #info {{
            position: absolute;
            top: 10px;
            left: 10px;
            color: white;
            font-family: monospace;
            background: rgba(0,0,0,0.7);
            padding: 10px;
            border-radius: 5px;
            font-size: 12px;
        }}
        #controls {{
            position: absolute;
            bottom: 10px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0,0,0,0.7);
            padding: 10px;
            border-radius: 5px;
            color: white;
            font-family: monospace;
        }}
        button {{
            background: #4CAF50;
            color: white;
            border: none;
            padding: 8px 16px;
            margin: 2px;
            cursor: pointer;
            border-radius: 3px;
        }}
        button:hover {{ background: #45a049; }}
    </style>
</head>
<body>
    <div id="info">
        <div>Models: {len(fbx_data_list)}</div>
        <div>Controls: Left-click drag to rotate, Right-click drag to pan, Scroll to zoom</div>
    </div>
    <div id="controls">
        <button onclick="playAll()">▶ Play All</button>
        <button onclick="pauseAll()">⏸ Pause All</button>
        <button onclick="resetCamera()">🎥 Reset Camera</button>
    </div>
    <canvas id="canvas"></canvas>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/fflate@0.7.4/umd/index.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/FBXLoader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    
    <script>
        const fbxDataList = {fbx_json};
        
        let scene, camera, renderer, controls;
        let mixers = [];
        let clock = new THREE.Clock();
        
        init();
        animate();
        
        function init() {{
            // Scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x1a1a1a);
            scene.fog = new THREE.Fog(0x1a1a1a, 5, 15);
            
            // Camera
            camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 100);
            camera.position.set(fbxDataList.length * 0.5, 1.2, 3.5);
            
            // Renderer
            const canvas = document.getElementById('canvas');
            renderer = new THREE.WebGLRenderer({{ canvas: canvas, antialias: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.shadowMap.enabled = true;
            
            // Controls
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.target.set(fbxDataList.length * 0.5, 1, 0);
            controls.update();
            
            // Lights
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            scene.add(ambientLight);
            
            const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
            dirLight.position.set(5, 10, 5);
            dirLight.castShadow = true;
            scene.add(dirLight);
            
            const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
            dirLight2.position.set(-5, 5, -5);
            scene.add(dirLight2);
            
            // Ground plane
            const groundGeometry = new THREE.PlaneGeometry(fbxDataList.length * 2 + 4, 10);
            const groundMaterial = new THREE.MeshStandardMaterial({{ 
                color: 0x333333,
                roughness: 0.8,
                metalness: 0.2
            }});
            const ground = new THREE.Mesh(groundGeometry, groundMaterial);
            ground.rotation.x = -Math.PI / 2;
            ground.receiveShadow = true;
            scene.add(ground);
            
            // Grid helper
            const gridHelper = new THREE.GridHelper(fbxDataList.length * 2 + 4, 20, 0x666666, 0x444444);
            scene.add(gridHelper);
            
            // Load all FBX models
            const loader = new THREE.FBXLoader();
            
            fbxDataList.forEach((fbxData, index) => {{
                const binaryData = Uint8Array.from(atob(fbxData.data), c => c.charCodeAt(0));
                const blob = new Blob([binaryData], {{ type: 'application/octet-stream' }});
                const url = URL.createObjectURL(blob);
                
                loader.load(url, (fbx) => {{
                    // Scale down the model to reasonable size
                    fbx.scale.set(0.01, 0.01, 0.01);
                    
                    // Position models with spacing
                    fbx.position.x = fbxData.x_offset;
                    fbx.position.y = 0;
                    fbx.position.z = 0;
                    
                    // Enable shadows
                    fbx.traverse((node) => {{
                        if (node.isMesh) {{
                            node.castShadow = true;
                            node.receiveShadow = true;
                        }}
                    }});
                    
                    scene.add(fbx);
                    
                    // Setup animation
                    if (fbx.animations && fbx.animations.length > 0) {{
                        const mixer = new THREE.AnimationMixer(fbx);
                        fbx.animations.forEach((clip) => {{
                            const action = mixer.clipAction(clip);
                            action.play();
                        }});
                        mixers.push(mixer);
                    }}
                    
                    // Add label
                    addLabel(fbxData.name, fbxData.x_offset);
                }}, undefined, (error) => {{
                    console.error('Error loading FBX:', error);
                }});
            }});
            
            // Window resize
            window.addEventListener('resize', onWindowResize);
        }}
        
        function addLabel(text, xPos) {{
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            canvas.width = 512;
            canvas.height = 64;
            
            context.fillStyle = 'rgba(0, 0, 0, 0.7)';
            context.fillRect(0, 0, canvas.width, canvas.height);
            
            context.font = '24px monospace';
            context.fillStyle = 'white';
            context.textAlign = 'center';
            context.fillText(text.replace('.fbx', ''), canvas.width / 2, canvas.height / 2 + 8);
            
            const texture = new THREE.CanvasTexture(canvas);
            const spriteMaterial = new THREE.SpriteMaterial({{ map: texture }});
            const sprite = new THREE.Sprite(spriteMaterial);
            sprite.position.set(xPos, 2.5, 0);
            sprite.scale.set(1, 0.125, 1);
            scene.add(sprite);
        }}
        
        function onWindowResize() {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }}
        
        function animate() {{
            requestAnimationFrame(animate);
            
            const delta = clock.getDelta();
            mixers.forEach(mixer => mixer.update(delta));
            
            controls.update();
            renderer.render(scene, camera);
        }}
        
        function playAll() {{
            mixers.forEach(mixer => {{
                mixer._actions.forEach(action => {{
                    action.paused = false;
                    action.play();
                }});
            }});
        }}
        
        function pauseAll() {{
            mixers.forEach(mixer => {{
                mixer._actions.forEach(action => {{
                    action.paused = true;
                }});
            }});
        }}
        
        function resetCamera() {{
            camera.position.set(fbxDataList.length * 0.5, 1.2, 3.5);
            controls.target.set(fbxDataList.length * 0.5, 0.85, 0);
            controls.update();
        }}
    </script>
</body>
</html>
'''
        return html

    def _get_example_choices(self):
        """Get all example choices from all data sources"""
        choices = ["Custom Input"]
        for source_name in self.all_example_data:
            example_data = self.all_example_data[source_name]
            for text, _ in example_data:
                display_text = f"{text[:50]}..." if len(text) > 50 else text
                choices.append(display_text)
        return choices

    def _on_example_select(self, selected_example):
        """When selecting an example, the callback function"""
        if selected_example == "Custom Input":
            return "", self._generate_random_seeds(), gr.update()
        else:
            # find the corresponding example from all data sources
            for source_name in self.all_example_data:
                example_data = self.all_example_data[source_name]
                for text, duration in example_data:
                    display_text = f"{text[:50]}..." if len(text) > 50 else text
                    if display_text == selected_example:
                        return text, self._generate_random_seeds(), gr.update(value=duration)
            return "", self._generate_random_seeds(), gr.update()

    def build_ui(self):
        with gr.Blocks(css=APP_CSS) as demo:
            self.header_md = gr.Markdown(HEADER_BASE_MD, elem_classes=["main-header"])

            with gr.Row():
                # Left control panel
                with gr.Column(scale=2, elem_classes=["left-panel"]):
                    # Input textbox
                    self.text_input = gr.Textbox(
                        label="📝 Input Text",
                        placeholder="Enter text to generate motion, support Chinese and English text input.",
                    )
                    # Rewritten textbox
                    self.rewritten_text = gr.Textbox(
                        label="✏️ Rewritten Text",
                        placeholder="Rewritten text will be displayed here, you can further edit",
                        interactive=True,
                        visible=False,
                    )
                    # Duration slider
                    self.duration_slider = gr.Slider(
                        minimum=0.5,
                        maximum=12,
                        value=5.0,
                        step=0.1,
                        label="⏱️ Action Duration (seconds)",
                        info="Feel free to adjust the action duration",
                    )

                    # Execute buttons
                    with gr.Row():
                        if self.prompt_engineering_available:
                            self.rewrite_btn = gr.Button(
                                "🔄 Rewrite Text",
                                variant="secondary",
                                size="lg",
                                elem_classes=["rewrite-button"],
                            )
                        else:
                            # Create a hidden/disabled placeholder button
                            self.rewrite_btn = gr.Button(
                                "🔄 Rewrite Text (Unavailable)",
                                variant="secondary",
                                size="lg",
                                elem_classes=["rewrite-button"],
                                interactive=False,
                                visible=False,
                            )

                        self.generate_btn = gr.Button(
                            "🚀 Generate Motion",
                            variant="primary",
                            size="lg",
                            elem_classes=["generate-button"],
                            interactive=not self.prompt_engineering_available,  # Enable directly if rewrite not available
                        )

                    if not self.prompt_engineering_available:
                        gr.Markdown(
                            "> ⚠️ **Prompt engineering is not available.** Text rewriting and duration estimation are disabled. Your input text and duration will be used directly."
                        )

                    # Advanced settings
                    with gr.Accordion("🔧 Advanced Settings", open=False):
                        self._build_advanced_settings()

                    # Example selection dropdown
                    self.example_dropdown = gr.Dropdown(
                        choices=self._get_example_choices(),
                        value="Custom Input",
                        label="📚 Test Examples",
                        info="Select a preset example or input your own text above",
                        interactive=True,
                    )

                    # Status message depends on whether rewrite is available
                    if self.prompt_engineering_available:
                        status_msg = "Please click the [🔄 Rewrite Text] button to rewrite the text first"
                    else:
                        status_msg = "Enter your text and click [🚀 Generate Motion] directly."

                    self.status_output = gr.Textbox(
                        label="📊 Status Information",
                        value=status_msg,
                    )

                    # Custom FBX Upload section
                    with gr.Accordion("🎭 Custom FBX Skeleton (Optional)", open=False):
                        self.custom_fbx_upload = gr.File(
                            label="Upload Custom FBX Character",
                            file_types=[".fbx"],
                            type="filepath",
                        )
                        gr.Markdown(
                            "> **Upload a custom FBX character to retarget the generated motion (e.g., Mixamo characters)**\n\n"
                            "> If you upload a custom FBX, the motion will be retargeted to your character's skeleton. "
                            "This is useful for applying motions to game characters, avatars, or any rigged FBX model."
                        )

                    # FBX Download section
                    with gr.Row(visible=False) as self.fbx_download_row:
                        if getattr(self.runtime, "fbx_available", False):
                            self.fbx_files = gr.File(
                                label="📦 Download FBX Files",
                                file_count="multiple",
                                interactive=False,
                            )
                        else:
                            self.fbx_files = gr.State([])

                # Right display area
                with gr.Column(scale=3):
                    self.output_display = gr.HTML(
                        value=HTML_OUTPUT_PLACEHOLDER, show_label=False, elem_classes=["flask-display"]
                    )

            # Footer
            gr.Markdown(FOOTER_MD, elem_classes=["footer"])

            self._bind_events()
            demo.load(fn=self._get_header_text, outputs=[self.header_md])
            return demo

    def _build_advanced_settings(self):
        # Only show rewrite options if rewrite is available
        if self.prompt_engineering_available:
            with gr.Group():
                gr.Markdown("### 🔄 Text Rewriting Options")
                with gr.Row():
                    self.enable_rewrite = gr.Checkbox(
                        label="Enable Text Rewriting",
                        value=True,
                        info="Automatically optimize text prompt to get better motion generation",
                    )

            with gr.Group():
                gr.Markdown("### ⏱️ Duration Settings")
                self.enable_duration_est = gr.Checkbox(
                    label="Enable Duration Estimation",
                    value=True,
                    info="Automatically estimate the duration of the motion",
                )
        else:
            # Create hidden placeholders with default values (disabled)
            self.enable_rewrite = gr.Checkbox(
                label="Enable Text Rewriting",
                value=False,
                visible=False,
            )
            self.enable_duration_est = gr.Checkbox(
                label="Enable Duration Estimation",
                value=False,
                visible=False,
            )
            with gr.Group():
                gr.Markdown("### ⚠️ Prompt Engineering Unavailable")
                gr.Markdown(
                    "Text rewriting and duration estimation are not available. "
                    "Your input text and duration will be used directly."
                )

        with gr.Group():
            gr.Markdown("### ⚙️ Generation Parameters")
            with gr.Row():
                with gr.Column(scale=3):
                    self.seed_input = gr.Textbox(
                        label="🎯 Random Seed List (comma separated)",
                        value="0,1,2,3",
                        placeholder="Enter comma separated seed list (e.g.: 0,1,2,3)",
                        info="Random seeds control the diversity of generated motions",
                    )
                with gr.Column(scale=1, min_width=60, elem_classes=["dice-container"]):
                    self.dice_btn = gr.Button(
                        "🎲 Lucky Button",
                        variant="secondary",
                        size="sm",
                        elem_classes=["dice-button"],
                    )

            self.cfg_slider = gr.Slider(
                minimum=1,
                maximum=10,
                value=5.0,
                step=0.1,
                label="⚙️ CFG Strength",
                info="Text fidelity: higher = more faithful to the prompt",
            )

    def _bind_events(self):
        # Generate random seeds
        self.dice_btn.click(self._generate_random_seeds, outputs=[self.seed_input])

        # Bind example selection event
        self.example_dropdown.change(
            fn=self._on_example_select,
            inputs=[self.example_dropdown],
            outputs=[self.text_input, self.seed_input, self.duration_slider],
        )

        # Rewrite text logic (only bind when rewrite is available)
        if self.prompt_engineering_available:
            self.rewrite_btn.click(fn=lambda: "Rewriting text, please wait...", outputs=[self.status_output]).then(
                self._prompt_engineering,
                inputs=[
                    self.text_input,
                    self.duration_slider,
                    self.enable_rewrite,
                    self.enable_duration_est,
                ],
                outputs=[self.rewritten_text, self.generate_btn, self.duration_slider],
            ).then(
                fn=lambda: (
                    gr.update(visible=True),
                    "Text rewriting completed! Please check and edit the rewritten text, then click [🚀 Generate Motion]",
                ),
                outputs=[self.rewritten_text, self.status_output],
            )

        # Generate motion logic
        self.generate_btn.click(
            fn=lambda: "Generating motion, please wait... (It takes some extra time to start the renderer for the first generation)",
            outputs=[self.status_output],
        ).then(
            self._generate_motion,
            inputs=[
                self.text_input,
                self.rewritten_text,
                self.seed_input,
                self.duration_slider,
                self.cfg_slider,
                self.custom_fbx_upload,
            ],
            outputs=[self.output_display, self.fbx_files],
            concurrency_limit=NUM_WORKERS,
        ).then(
            fn=lambda fbx_list: (
                (
                    "🎉 Motion generation completed! You can view the motion visualization result on the right. FBX files are ready for download."
                    if fbx_list
                    else "🎉 Motion generation completed! You can view the motion visualization result on the right"
                ),
                gr.update(visible=bool(fbx_list)),
            ),
            inputs=[self.fbx_files],
            outputs=[self.status_output, self.fbx_download_row],
        )

        # Reset logic - different behavior based on rewrite availability
        if self.prompt_engineering_available:
            self.text_input.change(
                fn=lambda: (
                    gr.update(visible=False),
                    gr.update(interactive=False),
                    "Please click the [🔄 Rewrite Text] button to rewrite the text first",
                ),
                outputs=[self.rewritten_text, self.generate_btn, self.status_output],
            )
        else:
            # When rewrite is not available, enable generate button directly when text is entered
            self.text_input.change(
                fn=lambda text: (
                    gr.update(visible=False),
                    gr.update(interactive=bool(text.strip())),
                    (
                        "Ready to generate! Click [🚀 Generate Motion] to start."
                        if text.strip()
                        else "Enter your text and click [🚀 Generate Motion] directly."
                    ),
                ),
                inputs=[self.text_input],
                outputs=[self.rewritten_text, self.generate_btn, self.status_output],
            )
        # Only bind rewritten_text change when rewrite is available
        if self.prompt_engineering_available:
            self.rewritten_text.change(
                fn=lambda text: (
                    gr.update(interactive=bool(text.strip())),
                    (
                        "Rewritten text has been modified, you can click [🚀 Generate Motion]"
                        if text.strip()
                        else "Rewritten text cannot be empty, please enter valid text"
                    ),
                ),
                inputs=[self.rewritten_text],
                outputs=[self.generate_btn, self.status_output],
            )


def create_demo(final_model_path):
    """Create the Gradio demo with Zero GPU support."""
    global _global_runtime, _global_args

    class Args:
        model_path = final_model_path
        output_dir = "output/gradio"
        prompt_engineering_host = os.environ.get("PROMPT_HOST", None)
        prompt_engineering_model_path = os.environ.get("PROMPT_MODEL_PATH", None)
        disable_prompt_engineering = os.environ.get("DISABLE_PROMPT_ENGINEERING", False)

    args = Args()
    _global_args = args  # Set global args for lazy loading

    # Check required files:
    cfg = osp.join(args.model_path, "config.yml")
    ckpt = osp.join(args.model_path, "latest.ckpt")
    if not osp.exists(cfg):
        raise FileNotFoundError(f">>> Configuration file not found: {cfg}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # For Zero GPU: Don't load model at startup, use lazy loading
    # Create a minimal runtime for UI initialization (without model loading)
    if SPACES_AVAILABLE:
        print(">>> Hugging Face Spaces detected. Using Zero GPU lazy loading.")
        print(">>> Model will be loaded on first GPU request.")

        # Create a placeholder runtime with minimal initialization for UI
        class PlaceholderRuntime:
            def __init__(self):
                self.fbx_available = False
                self.prompt_engineering_host = args.prompt_engineering_host

            def rewrite_text_and_infer_time(self, text: str):
                # For prompt rewriting, we don't need GPU
                from hymotion.prompt_engineering.prompt_rewrite import PromptRewriter

                rewriter = PromptRewriter(
                    host=self.prompt_engineering_host, model_path=self.prompt_engineering_model_path
                )
                return rewriter.rewrite_prompt_and_infer_time(text)

        runtime = PlaceholderRuntime()
    else:
        # Local development: load model immediately
        print(">>> Local environment detected. Loading model at startup.")
        skip_model_loading = False
        if not os.path.exists(ckpt):
            print(f">>> [WARNING] Checkpoint file not found: {ckpt}")
            print(f">>> [WARNING] Model loading will be skipped. Motion generation will not be available.")
            skip_model_loading = True

        print(">>> Initializing T2MRuntime...")
        if "USE_HF_MODELS" not in os.environ:
            os.environ["USE_HF_MODELS"] = "1"

        skip_text = False
        runtime = T2MRuntime(
            config_path=cfg,
            ckpt_name=ckpt,
            skip_text=skip_text,
            device_ids=None,
            skip_model_loading=skip_model_loading,
            disable_prompt_engineering=args.disable_prompt_engineering,
            prompt_engineering_host=args.prompt_engineering_host,
            prompt_engineering_model_path=args.prompt_engineering_model_path,
        )
        _global_runtime = runtime  # Set global runtime for GPU function

    ui = T2MGradioUI(runtime=runtime, args=args)
    demo = ui.build_ui()
    return demo


if __name__ == "__main__":
    # Create demo at module level for Hugging Face Spaces
    final_model_path = try_to_download_model()
    demo = create_demo(final_model_path)
    demo.launch()

# t2m_runtime.py
import os
import threading
import time
import uuid
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import yaml

from ..prompt_engineering.prompt_rewrite import PromptRewriter
from .loaders import load_object
from .visualize_mesh_web import save_visualization_data, generate_static_html_content

try:
    import fbx

    FBX_AVAILABLE = True
    print(">>> FBX module found.")
except ImportError:
    FBX_AVAILABLE = False
    print(">>> FBX module not found.")


def _get_local_ip():
    import subprocess

    result = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        for ip in result.stdout.strip().split():
            if not ip.startswith("127.") and not ip.startswith("172.17."):
                return ip
    return "localhost"


def _now():
    t = time.time()
    ms = int((t - int(t)) * 1000)
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(t)) + f"{ms:03d}"


class T2MRuntime:
    def __init__(
        self,
        config_path: str,
        ckpt_name: str = "latest.ckpt",
        skip_text: bool = False,
        device_ids: Union[list[int], None] = None,
        skip_model_loading: bool = False,
        force_cpu: bool = False,
        disable_prompt_engineering: bool = False,
        prompt_engineering_host: Optional[str] = None,
        prompt_engineering_model_path: Optional[str] = None,
    ):
        self.config_path = config_path
        self.ckpt_name = ckpt_name
        self.skip_text = skip_text
        self.prompt_engineering_host = prompt_engineering_host
        self.prompt_engineering_model_path = prompt_engineering_model_path
        self.disable_prompt_engineering = disable_prompt_engineering
        self.skip_model_loading = skip_model_loading
        self.local_ip = _get_local_ip()

        if force_cpu:
            print(">>> [INFO] CPU mode enabled via HY_MOTION_DEVICE=cpu environment variable")
            self.device_ids = []
        elif torch.cuda.is_available():
            all_ids = list(range(torch.cuda.device_count()))
            self.device_ids = all_ids if device_ids is None else [i for i in device_ids if i in all_ids]
        else:
            self.device_ids = []

        self.pipelines = []
        self._gpu_load = []
        self._lock = threading.Lock()
        self._loaded = False

        if self.disable_prompt_engineering:
            self.prompt_rewriter = None
        else:
            self.prompt_rewriter = PromptRewriter(
                host=self.prompt_engineering_host, model_path=self.prompt_engineering_model_path
            )
        # Skip model loading if checkpoint not found
        if self.skip_model_loading:
            print(">>> [WARNING] Checkpoint not found, will use randomly initialized model weights")
        self.load()
        
        # Initialize body model for keypoints3d generation
        try:
            from ..pipeline.body_model import WoodenMesh
            self.body_model = WoodenMesh()
            print(">>> Body model (WoodenMesh) initialized")
        except Exception as e:
            print(f">>> Failed to initialize body model: {e}")
            self.body_model = None
        
        self.fbx_available = FBX_AVAILABLE
        if self.fbx_available:
            try:
                from .smplh2woodfbx import SMPLH2WoodFBX

                self.fbx_converter = SMPLH2WoodFBX()
            except Exception as e:
                print(f">>> Failed to initialize FBX converter: {e}")
                self.fbx_available = False
                self.fbx_converter = None
        else:
            self.fbx_converter = None
            print(">>> FBX module not found. FBX export will be disabled.")

        device_info = self.device_ids if self.device_ids else "cpu"
        if self.skip_model_loading:
            print(
                f">>> T2MRuntime initialized (using randomly initialized weights) in IP {self.local_ip}, devices={device_info}"
            )
        else:
            print(f">>> T2MRuntime loaded in IP {self.local_ip}, devices={device_info}")

    def load(self):
        if self._loaded:
            return
        print(f">>> Loading model from {self.config_path}...")

        with open(self.config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # Use allow_empty_ckpt=True when skip_model_loading is True
        allow_empty_ckpt = self.skip_model_loading

        if not self.device_ids:
            pipeline = load_object(
                config["train_pipeline"],
                config["train_pipeline_args"],
                network_module=config["network_module"],
                network_module_args=config["network_module_args"],
            )
            device = torch.device("cpu")
            pipeline.load_in_demo(
                self.ckpt_name,
                build_text_encoder=not self.skip_text,
                allow_empty_ckpt=allow_empty_ckpt,
            )
            pipeline.to(device)
            self.pipelines = [pipeline]
            self._gpu_load = [0]
        else:
            for gid in self.device_ids:
                p = load_object(
                    config["train_pipeline"],
                    config["train_pipeline_args"],
                    network_module=config["network_module"],
                    network_module_args=config["network_module_args"],
                )
                p.load_in_demo(
                    self.ckpt_name,
                    build_text_encoder=not self.skip_text,
                    allow_empty_ckpt=allow_empty_ckpt,
                )
                p.to(torch.device(f"cuda:{gid}"))
                self.pipelines.append(p)
            self._gpu_load = [0] * len(self.pipelines)

        self._loaded = True

    def _acquire_pipeline(self) -> int:
        while True:
            with self._lock:
                for i in range(len(self._gpu_load)):
                    if self._gpu_load[i] == 0:
                        self._gpu_load[i] = 1
                        return i
            time.sleep(0.01)

    def _release_pipeline(self, idx: int):
        with self._lock:
            self._gpu_load[idx] = 0

    def test_dit_inference(self, duration: float = 2.0, seed: int = 42) -> bool:
        """
        Test DiT model inference with unconditional/blank input.
        This method is used to verify the DiT model works before loading text encoder.

        Args:
            duration: Duration of the test motion in seconds
            seed: Random seed for reproducibility

        Returns:
            True if inference succeeds and produces valid output
        """
        if not self.pipelines:
            raise RuntimeError("No pipeline loaded. Call load() first.")

        pi = self._acquire_pipeline()
        try:
            pipeline = self.pipelines[pi]
            pipeline.eval()
            device = next(pipeline.parameters()).device

            # Calculate frame length from duration (assuming 30fps output, 20fps internal)
            length = int(duration * 20)
            length = min(length, pipeline.train_frames)

            # Use null features for unconditional generation
            batch_size = 1
            vtxt_input = pipeline.null_vtxt_feat.expand(batch_size, -1, -1).to(device)
            ctxt_input = pipeline.null_ctxt_input.expand(batch_size, -1, -1).to(device)
            ctxt_length = torch.tensor([1] * batch_size, device=device)

            # Create masks
            from ..pipeline.motion_diffusion import length_to_mask

            ctxt_mask_temporal = length_to_mask(ctxt_length, ctxt_input.shape[1])
            x_length = torch.LongTensor([length] * batch_size).to(device)
            x_mask_temporal = length_to_mask(x_length, pipeline.train_frames)

            # Run denoising inference
            print(f"\t>>> Running DiT inference test: length={length}, device={device}")

            # Create random noise
            generator = torch.Generator(device=device).manual_seed(seed)
            latent_shape = (batch_size, pipeline.train_frames, pipeline.mean.shape[-1])
            latents = torch.randn(latent_shape, generator=generator, device=device, dtype=vtxt_input.dtype)

            # Simple single-step denoising test (just forward pass)
            with torch.no_grad():
                # Get timestep
                timesteps = torch.tensor([0.5], device=device, dtype=vtxt_input.dtype).expand(batch_size)

                # Forward pass through DiT
                # Use correct parameter names for HunyuanMotionMMDiT.forward()
                _ = pipeline.motion_transformer(
                    x=latents,
                    ctxt_input=ctxt_input,
                    vtxt_input=vtxt_input,
                    timesteps=timesteps,
                    x_mask_temporal=x_mask_temporal,
                    ctxt_mask_temporal=ctxt_mask_temporal,
                )

            print(f"\t>>> DiT forward pass completed successfully!")
            return True

        except Exception as e:
            print(f"\t>>> DiT inference test failed: {e}")
            raise
        finally:
            self._release_pipeline(pi)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load_text_encoder(self) -> None:
        """
        Load text encoder for all pipelines.
        This is called after DiT model testing to complete the initialization.
        """
        if not self.pipelines:
            raise RuntimeError("No pipeline loaded. Call load() first.")

        print(">>> Loading text encoder for all pipelines...")
        for i, pipeline in enumerate(self.pipelines):
            if not hasattr(pipeline, "text_encoder") or pipeline.text_encoder is None:
                device = next(pipeline.parameters()).device
                pipeline.text_encoder = load_object(pipeline._text_encoder_module, pipeline._text_encoder_cfg)
                pipeline.text_encoder.to(device)
                print(f"\t>>> Text encoder loaded for pipeline {i} on {device}")

        # Update skip_text flag
        self.skip_text = False
        print(">>> Text encoder loading completed!")

    def rewrite_text_and_infer_time(self, text: str) -> Tuple[float, str]:
        print("Start rewriting text...")
        duration, rewritten_text = self.prompt_rewriter.rewrite_prompt_and_infer_time(f"{text}")
        print(f"\t>>> Rewritten text: {rewritten_text}, duration: {duration:.2f} seconds")
        return duration, rewritten_text

    def generate_motion(
        self,
        text: str,
        seeds_csv: str,
        duration: float,
        cfg_scale: float,
        output_format: str = "fbx",
        output_dir: Optional[str] = None,
        output_filename: Optional[str] = None,
        original_text: Optional[str] = None,
        use_special_game_feat: bool = False,
    ) -> Tuple[Union[str, list[str]], dict]:
        self.load()
        seeds = [int(s.strip()) for s in seeds_csv.split(",") if s.strip() != ""]
        pi = self._acquire_pipeline()
        try:
            pipeline = self.pipelines[pi]
            pipeline.eval()

            # When skip_text=True (debug mode), use blank text features
            if self.skip_text:
                print(">>> [Debug Mode] Using blank text features (skip_text=True)")
                device = next(pipeline.parameters()).device
                batch_size = len(seeds) if seeds else 1
                # Create blank hidden_state_dict using null features
                hidden_state_dict = {
                    "text_vec_raw": pipeline.null_vtxt_feat.expand(batch_size, -1, -1).to(device),
                    "text_ctxt_raw": pipeline.null_ctxt_input.expand(batch_size, -1, -1).to(device),
                    "text_ctxt_raw_length": torch.tensor([1] * batch_size, device=device),
                }
                # Disable CFG in debug mode (use cfg_scale=1.0)
                model_output = pipeline.generate(
                    text,
                    seeds,
                    duration,
                    cfg_scale=1.0,
                    use_special_game_feat=False,
                    hidden_state_dict=hidden_state_dict,
                )
            else:
                model_output = pipeline.generate(
                    text, seeds, duration, cfg_scale=cfg_scale, use_special_game_feat=use_special_game_feat
                )
        finally:
            self._release_pipeline(pi)

        ts = _now()
        save_data, base_filename = save_visualization_data(
            output=model_output,
            text=text if original_text is None else original_text,
            rewritten_text=text,
            timestamp=ts,
            output_dir=output_dir,
            output_filename=output_filename,
        )

        html_content = self._generate_html_content(
            timestamp=ts,
            file_path=base_filename,
            output_dir=output_dir,
        )

        if output_format == "fbx" and not self.fbx_available:
            print(">>> Warning: FBX export requested but FBX SDK is not available. Falling back to dict format.")
            output_format = "dict"

        if output_format == "fbx" and self.fbx_available:
            fbx_files = self._generate_fbx_files(
                visualization_data=save_data,
                output_dir=output_dir,
                fbx_filename=output_filename,
            )
            return html_content, fbx_files, model_output
        elif output_format == "dict":
            # Return HTML content and empty list for fbx_files when using dict format
            return html_content, [], model_output
        else:
            raise ValueError(f">>> Invalid output format: {output_format}")

    def _generate_html_content(
        self,
        timestamp: str,
        file_path: str,
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Generate static HTML content with embedded data for iframe srcdoc.
        All JavaScript code is embedded directly in the HTML, no external static resources needed.

        Args:
            timestamp: Timestamp string for logging
            file_path: Base filename (without extension)
            output_dir: Directory where NPZ/meta files are stored

        Returns:
            HTML content string (to be used in iframe srcdoc)
        """
        print(f">>> Generating static HTML content, timestamp: {timestamp}")
        gradio_dir = output_dir if output_dir is not None else "output/gradio"

        try:
            # Generate static HTML content with embedded data (all JS is embedded in template)
            html_content = generate_static_html_content(
                folder_name=gradio_dir,
                file_name=file_path,
                hide_captions=False,
            )

            print(f">>> Static HTML content generated for: {file_path}")
            return html_content

        except Exception as e:
            print(f">>> Failed to generate static HTML content: {e}")
            import traceback

            traceback.print_exc()
            # Return error HTML
            return f"<html><body><h1>Error generating visualization</h1><p>{str(e)}</p></body></html>"

    def _generate_fbx_files(
        self,
        visualization_data: dict,
        output_dir: Optional[str] = None,
        fbx_filename: Optional[str] = None,
    ) -> List[str]:
        assert "smpl_data" in visualization_data, "smpl_data not found in visualization_data"
        fbx_files = []
        if output_dir is None:
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            output_dir = os.path.join(root_dir, "output", "gradio")

        smpl_data_list = visualization_data["smpl_data"]

        unique_id = str(uuid.uuid4())[:8]
        text = visualization_data["text"]
        timestamp = visualization_data["timestamp"]
        
        # Get rot6d format data for NPZ saving (matching HYMotionSaveNPZ format)
        rot6d_data = visualization_data.get("rot6d")  # (batch_size, num_frames, 22, 6)
        transl_data = visualization_data.get("transl")  # (batch_size, num_frames, 3)
        
        for bb in range(len(smpl_data_list)):
            smpl_data = smpl_data_list[bb]
            if fbx_filename is None:
                fbx_filename_bb = f"{timestamp}_{unique_id}_{bb:03d}.fbx"
            else:
                fbx_filename_bb = f"{fbx_filename}_{bb:03d}.fbx"
            fbx_path = os.path.join(output_dir, fbx_filename_bb)
            success = self.fbx_converter.convert_npz_to_fbx(smpl_data, fbx_path)
            if success:
                fbx_files.append(fbx_path)
                print(f"\t>>> FBX file generated: {fbx_path}")
                
                # Save NPZ file in rot6d format (matching HYMotionSaveNPZ)
                if rot6d_data is not None and transl_data is not None:
                    npz_path = fbx_path.replace(".fbx", ".npz")
                    from ..utils.geometry import rot6d_to_rotation_matrix
                    
                    # Extract data for this batch sample
                    rot6d_bb = rot6d_data[bb] if hasattr(rot6d_data[bb], 'cpu') else torch.from_numpy(rot6d_data[bb])
                    transl_bb = transl_data[bb] if hasattr(transl_data[bb], 'cpu') else torch.from_numpy(transl_data[bb])
                    
                    # Compute root_rotations_mat and keypoints3d
                    root_rotations_mat = rot6d_to_rotation_matrix(rot6d_bb[:, 0])  # First joint is root
                    
                    # Generate keypoints3d using body model if available
                    keypoints3d = None
                    if hasattr(self, 'body_model') and self.body_model is not None:
                        with torch.no_grad():
                            body_out = self.body_model.forward({"rot6d": rot6d_bb.cpu(), "trans": transl_bb.cpu()})
                            keypoints3d = body_out["keypoints3d"]  # (num_frames, 52, 3)
                    else:
                        # Fallback: create placeholder keypoints3d
                        keypoints3d = torch.zeros(rot6d_bb.shape[0], 52, 3)
                    
                    # Save in rot6d format matching HYMotionSaveNPZ
                    npz_dict = {
                        "rot6d": rot6d_bb.cpu().numpy() if hasattr(rot6d_bb, 'cpu') else rot6d_bb,
                        "transl": transl_bb.cpu().numpy() if hasattr(transl_bb, 'cpu') else transl_bb,
                        "root_rotations_mat": root_rotations_mat.cpu().numpy() if hasattr(root_rotations_mat, 'cpu') else root_rotations_mat,
                        "keypoints3d": keypoints3d.cpu().numpy() if hasattr(keypoints3d, 'cpu') else keypoints3d,
                    }
                    np.savez_compressed(npz_path, **npz_dict)
                
                txt_path = fbx_path.replace(".fbx", ".txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text)
                fbx_files.append(txt_path)

        return fbx_files

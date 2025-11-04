import argparse
import os
import random
import secrets
from datetime import datetime
from enum import Enum
from typing import Optional

import torch


class GenerationStep(Enum):
    """Enum for tracking important steps in the image generation process"""

    INIT = "init"
    LOADING_MODEL = "loading_model"
    MODEL_LOADED = "model_loaded"
    LOADING_CUSTOM_LORA = "loading_custom_lora"
    LOADING_ULTRA_FAST_LORA = "loading_ultra_fast_lora"
    LOADING_FAST_LORA = "loading_fast_lora"
    LORA_LOADED = "lora_loaded"
    BATMAN_MODE_ACTIVATED = "batman_mode_activated"
    PREPARING_GENERATION = "preparing_generation"
    INFERENCE_START = "inference_start"
    INFERENCE_PROGRESS = "inference_progress"
    INFERENCE_COMPLETE = "inference_complete"
    SAVING_IMAGE = "saving_image"
    IMAGE_SAVED = "image_saved"
    COMPLETE = "complete"
    ERROR = "error"


def build_generate_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "generate",
        help="Generate a new image from text prompt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default="""A coffee shop entrance features a chalkboard sign reading "Apple Silicon Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "Generated with MPS on Apple Silicon". Next to it hangs a poster showing a beautiful woman, and beneath the poster is written "Just try it!".""",
        help="Prompt text to condition the image generation.",
    )
    parser.add_argument(
        "-np",
        "--negative-prompt",
        dest="negative_prompt",
        type=str,
        default=None,
        help=(
            "Text to discourage (negative prompt), e.g. 'blurry, watermark, text, low quality'. "
            "If omitted, an empty negative prompt is used."
        ),
    )
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=50,
        help="Number of inference steps for normal generation.",
    )
    parser.add_argument(
        "-f",
        "--fast",
        action="store_true",
        help="Use Lightning LoRA for fast generation (8 steps).",
    )
    parser.add_argument(
        "-uf",
        "--ultra-fast",
        action="store_true",
        help="Use Lightning LoRA for ultra-fast generation (4 steps).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Random seed for reproducible generation. If not provided, a random seed "
            "will be used for each image."
        ),
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="Number of images to generate.",
    )
    parser.add_argument(
        "--aspect",
        type=str,
        choices=["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"],
        default="16:9",
        help="Aspect ratio for the output image.",
    )
    parser.add_argument(
        "--lora",
        type=str,
        default=None,
        help="Path to local .safetensors file, Hugging Face model URL or repo ID for additional LoRA to load (e.g., '~/Downloads/lora.safetensors', 'flymy-ai/qwen-image-anime-irl-lora' or full HF URL).",
    )
    parser.add_argument(
        "--cfg-scale",
        dest="cfg_scale",
        type=float,
        default=None,
        help="Classifier-free guidance (CFG) scale. Overrides mode defaults (fast/ultra-fast use 1.0; normal uses 4.0).",
    )
    parser.add_argument(
        "--outdir",
        dest="output_dir",
        type=str,
        default=None,
        help="Directory to save generated images (default: ./output).",
    )
    parser.add_argument(
        "--batman",
        action="store_true",
        help="LEGO Batman photobombs your image! 🦇",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=[
            "Q2_K",
            "Q3_K_S",
            "Q3_K_M",
            "Q4_0",
            "Q4_1",
            "Q4_K_S",
            "Q4_K_M",
            "Q5_0",
            "Q5_1",
            "Q5_K_S",
            "Q5_K_M",
            "Q6_K",
            "Q8_0",
        ],
        help="Use GGUF quantized model (e.g., Q4_0 for ~3x memory reduction). Default uses standard model.",
    )
    return parser


def get_lora_path(ultra_fast=False, edit_mode=False):
    from huggingface_hub import hf_hub_download

    """Get the Lightning LoRA from Hugging Face Hub with a silent cache freshness check.

    The function will:
    - Look up any locally cached file for the target filename.
    - Then fetch the latest from the Hub (without forcing) which will reuse cache
      if up-to-date, or download a newer snapshot if the remote changed.
    - Return the final resolved local path.
    """

    if edit_mode and ultra_fast:
        # Use the ultra-fast Lightning LoRA for editing
        filename = "Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors"
        version = "Edit v1.0 (4-steps)"
    elif edit_mode:
        # Use the Lightning LoRA for standard fast editing
        filename = "Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors"
        version = "Edit v1.0 (8-steps)"
    elif ultra_fast:
        filename = "Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors"
        version = "v2.0 (4-steps)"
    else:
        filename = "Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors"
        version = "v2.0 (8-steps)"

    try:
        cached_path = None
        try:
            cached_path = hf_hub_download(
                repo_id="lightx2v/Qwen-Image-Lightning",
                filename=filename,
                repo_type="model",
                local_files_only=True,
            )
        except Exception:
            cached_path = None

        # Resolve latest from Hub; will reuse cache if fresh, or download newer
        latest_path = hf_hub_download(
            repo_id="lightx2v/Qwen-Image-Lightning",
            filename=filename,
            repo_type="model",
        )

        if cached_path and latest_path != cached_path:
            # A newer snapshot was fetched; keep output quiet per request
            pass

        print(f"Lightning LoRA {version} loaded from: {latest_path}")
        return latest_path
    except Exception as e:
        print(f"Failed to load Lightning LoRA {version}: {e}")
        return None


def get_custom_lora_path(lora_spec):
    """Get a custom LoRA from Hugging Face Hub or load from a local file.

    Args:
        lora_spec: Either a local file path to a safetensors file, a full HF URL,
                   or a repo ID (e.g., 'flymy-ai/qwen-image-anime-irl-lora')

    Returns:
        Path to the LoRA file (local or downloaded), or None if failed
    """
    import re
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    # Check if it's a local file path (handles both absolute and ~ paths)
    lora_path = Path(lora_spec).expanduser()
    if lora_path.exists() and lora_path.suffix == ".safetensors":
        print(f"Using local LoRA file: {lora_path}")
        return str(lora_path.absolute())

    # If not a local file, try HuggingFace
    # Extract repo_id from URL if it's a full HF URL
    if lora_spec.startswith("https://huggingface.co/"):
        # Extract repo_id from URL like https://huggingface.co/flymy-ai/qwen-image-anime-irl-lora
        match = re.match(r"https://huggingface\.co/([^/]+/[^/]+)", lora_spec)
        if match:
            repo_id = match.group(1)
        else:
            print(f"Invalid Hugging Face URL format: {lora_spec}")
            return None
    else:
        # Assume it's already a repo ID
        repo_id = lora_spec

    try:
        # First, try to list files to find the LoRA safetensors file
        from huggingface_hub import list_repo_files

        print(f"Looking for LoRA files in {repo_id}...")
        files = list_repo_files(repo_id, repo_type="model")

        # Find safetensors files that might be LoRAs
        safetensors_files = [f for f in files if f.endswith(".safetensors")]

        if not safetensors_files:
            print(f"No safetensors files found in {repo_id}")
            return None

        # Prefer files with 'lora' in the name, otherwise take the first one
        lora_files = [f for f in safetensors_files if "lora" in f.lower()]
        filename = lora_files[0] if lora_files else safetensors_files[0]

        print(f"Downloading LoRA file: {filename}")

        # Download the LoRA file
        lora_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
        )

        print(f"Custom LoRA loaded from: {lora_path}")
        return lora_path

    except Exception as e:
        print(f"Failed to load custom LoRA from {repo_id}: {e}")
        return None


def merge_lora_from_safetensors(pipe, lora_path):
    """Merge LoRA weights from safetensors file into the pipeline's transformer."""
    import safetensors.torch

    lora_state_dict = safetensors.torch.load_file(lora_path)

    transformer = pipe.transformer
    merged_count = 0

    lora_keys = list(lora_state_dict.keys())

    # Detect LoRA format
    uses_dot_lora_format = any(
        ".lora.down" in key or ".lora.up" in key for key in lora_keys
    )
    uses_diffusers_format = any(key.startswith("lora_unet_") for key in lora_keys)
    uses_lora_ab_format = any(".lora_A" in key or ".lora_B" in key for key in lora_keys)

    # Map diffusers-style keys to transformer parameter names
    def convert_diffusers_key_to_transformer_key(diffusers_key):
        """Convert diffusers-style LoRA keys to match transformer parameter names."""
        # Remove lora_unet_ prefix
        key = diffusers_key.replace("lora_unet_", "")

        # Replace underscores with dots for the transformer_blocks part
        # e.g., transformer_blocks_0 -> transformer_blocks.0
        import re

        key = re.sub(r"transformer_blocks_(\d+)", r"transformer_blocks.\1", key)

        # Map the naming conventions
        replacements = {
            "_attn_add_k_proj": ".attn.add_k_proj",
            "_attn_add_q_proj": ".attn.add_q_proj",
            "_attn_add_v_proj": ".attn.add_v_proj",
            "_attn_to_add_out": ".attn.to_add_out",
            "_ff_context_mlp_fc1": ".ff_context.net.0",
            "_ff_context_mlp_fc2": ".ff_context.net.2",
            "_ff_mlp_fc1": ".ff.net.0",
            "_ff_mlp_fc2": ".ff.net.2",
            "_attn_to_k": ".attn.to_k",
            "_attn_to_q": ".attn.to_q",
            "_attn_to_v": ".attn.to_v",
            "_attn_to_out_0": ".attn.to_out.0",
        }

        for old, new in replacements.items():
            key = key.replace(old, new)

        return key

    if uses_lora_ab_format:
        # Handle lora_A/lora_B format (e.g., diffusion_model.transformer_blocks.X.attn.Y.lora_A.weight)
        for name, param in transformer.named_parameters():
            base_name = (
                name.replace(".weight", "") if name.endswith(".weight") else name
            )

            # Try to find matching LoRA weights with different possible prefixes
            lora_a_key = None
            lora_b_key = None

            # Try with diffusion_model prefix
            test_key_a = f"diffusion_model.{base_name}.lora_A.weight"
            test_key_b = f"diffusion_model.{base_name}.lora_B.weight"

            if test_key_a in lora_state_dict and test_key_b in lora_state_dict:
                lora_a_key = test_key_a
                lora_b_key = test_key_b
            else:
                # Try without prefix
                test_key_a = f"{base_name}.lora_A.weight"
                test_key_b = f"{base_name}.lora_B.weight"

                if test_key_a in lora_state_dict and test_key_b in lora_state_dict:
                    lora_a_key = test_key_a
                    lora_b_key = test_key_b

            if lora_a_key and lora_b_key:
                lora_down = lora_state_dict[
                    lora_a_key
                ]  # lora_A is equivalent to lora_down
                lora_up = lora_state_dict[lora_b_key]  # lora_B is equivalent to lora_up

                # Default alpha to rank if not specified
                lora_alpha = lora_down.shape[0]
                rank = lora_down.shape[0]
                scaling_factor = lora_alpha / rank

                # Convert to float32 for computation
                lora_up = lora_up.float()
                lora_down = lora_down.float()

                # Apply LoRA: weight = weight + scaling_factor * (up @ down)
                delta_W = scaling_factor * torch.matmul(lora_up, lora_down)
                param.data = (param.data + delta_W.to(param.device)).type_as(param.data)
                merged_count += 1
    elif uses_diffusers_format:
        # Handle diffusers-style LoRA (like modern-anime)
        for name, param in transformer.named_parameters():
            base_name = (
                name.replace(".weight", "") if name.endswith(".weight") else name
            )

            # Try different naming patterns
            lora_down_key = None
            lora_up_key = None
            lora_alpha_key = None

            # Check for exact match first
            for key in lora_keys:
                if key.startswith("lora_unet_"):
                    converted_key = convert_diffusers_key_to_transformer_key(
                        key.replace(".lora_down.weight", "")
                        .replace(".lora_up.weight", "")
                        .replace(".alpha", "")
                    )
                    if converted_key == base_name:
                        if key.endswith(".lora_down.weight"):
                            lora_down_key = key
                        elif key.endswith(".lora_up.weight"):
                            lora_up_key = key
                        elif key.endswith(".alpha"):
                            lora_alpha_key = key

            if lora_down_key and lora_up_key:
                lora_down = lora_state_dict[lora_down_key]
                lora_up = lora_state_dict[lora_up_key]

                # Get alpha value if it exists, otherwise use rank
                if lora_alpha_key and lora_alpha_key in lora_state_dict:
                    lora_alpha = float(lora_state_dict[lora_alpha_key])
                else:
                    lora_alpha = lora_down.shape[0]  # Use rank as default

                rank = lora_down.shape[0]
                scaling_factor = lora_alpha / rank

                # Convert to float32 for computation
                lora_up = lora_up.float()
                lora_down = lora_down.float()

                # Apply LoRA: weight = weight + scaling_factor * (up @ down)
                delta_W = scaling_factor * torch.matmul(lora_up, lora_down)
                param.data = (param.data + delta_W.to(param.device)).type_as(param.data)
                merged_count += 1
    else:
        # Handle original format LoRAs
        for name, param in transformer.named_parameters():
            # Remove .weight suffix if present to get base parameter name
            base_name = (
                name.replace(".weight", "") if name.endswith(".weight") else name
            )

            if uses_dot_lora_format:
                lora_down_key = f"transformer.{base_name}.lora.down.weight"
                lora_up_key = f"transformer.{base_name}.lora.up.weight"
                lora_alpha_key = f"transformer.{base_name}.alpha"

                if lora_down_key not in lora_state_dict:
                    lora_down_key = f"{base_name}.lora.down.weight"
                    lora_up_key = f"{base_name}.lora.up.weight"
                    lora_alpha_key = f"{base_name}.alpha"
            else:
                lora_down_key = f"{base_name}.lora_down.weight"
                lora_up_key = f"{base_name}.lora_up.weight"
                lora_alpha_key = f"{base_name}.alpha"

            if lora_down_key in lora_state_dict and lora_up_key in lora_state_dict:
                lora_down = lora_state_dict[lora_down_key]
                lora_up = lora_state_dict[lora_up_key]

                # Get alpha value if it exists, otherwise use rank
                if lora_alpha_key in lora_state_dict:
                    lora_alpha = float(lora_state_dict[lora_alpha_key])
                else:
                    lora_alpha = lora_down.shape[0]  # Use rank as default

                rank = lora_down.shape[0]
                scaling_factor = lora_alpha / rank

                # Convert to float32 for computation
                lora_up = lora_up.float()
                lora_down = lora_down.float()

                # Apply LoRA: weight = weight + scaling_factor * (up @ down)
                delta_W = scaling_factor * torch.matmul(lora_up, lora_down)
                param.data = (param.data + delta_W.to(param.device)).type_as(param.data)
                merged_count += 1

    print(f"Merged {merged_count} LoRA weights into the model")
    return pipe


def build_edit_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "edit",
        help="Edit an existing image using text instructions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to the input image(s) to edit. Provide multiple paths to blend edits across images.",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        required=True,
        help="Editing instructions (e.g., 'Change the sky to sunset colors').",
    )
    parser.add_argument(
        "--negative-prompt",
        dest="negative_prompt",
        type=str,
        default=None,
        help=(
            "Text to discourage in the edit (negative prompt). If omitted, an empty negative prompt is used."
        ),
    )
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=40,
        help="Number of inference steps for normal editing.",
    )
    parser.add_argument(
        "-f",
        "--fast",
        action="store_true",
        help="Use Lightning LoRA for fast editing (8 steps).",
    )
    parser.add_argument(
        "-uf",
        "--ultra-fast",
        action="store_true",
        help="Use Lightning LoRA for ultra-fast editing (4 steps).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation. If not provided, a random seed will be used.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output filename (default: edited-<timestamp>.png).",
    )
    parser.add_argument(
        "--cfg-scale",
        dest="cfg_scale",
        type=float,
        default=None,
        help="Classifier-free guidance (CFG) scale. Overrides mode defaults (fast/ultra-fast use 1.0; normal uses 4.0).",
    )
    parser.add_argument(
        "--outdir",
        dest="output_dir",
        type=str,
        default=None,
        help="Directory to save edited images (default: ./output).",
    )
    parser.add_argument(
        "--lora",
        type=str,
        default=None,
        help="Path to local .safetensors file, Hugging Face model URL or repo ID for additional LoRA to load (e.g., '~/Downloads/lora.safetensors', 'flymy-ai/qwen-image-anime-irl-lora' or full HF URL).",
    )
    parser.add_argument(
        "--batman",
        action="store_true",
        help="LEGO Batman photobombs your image! 🦇",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=[
            "Q2_K",
            "Q3_K_S",
            "Q3_K_M",
            "Q4_0",
            "Q4_1",
            "Q4_K_S",
            "Q4_K_M",
            "Q5_0",
            "Q5_1",
            "Q5_K_S",
            "Q5_K_M",
            "Q6_K",
            "Q8_0",
        ],
        help="Use GGUF quantized model (e.g., Q4_0 for ~3x memory reduction). Default uses standard model.",
    )
    return parser


def get_device_and_dtype():
    """Get the optimal device and dtype for the current system."""

    if torch.backends.mps.is_available():
        print("Using MPS")
        return "mps", torch.bfloat16
    elif torch.cuda.is_available():
        print("Using CUDA")
        return "cuda", torch.bfloat16
    else:
        print("Using CPU")
        return "cpu", torch.float32


def get_gguf_model_path(quantization: str):
    """Download and return path to GGUF quantized model.

    Args:
        quantization: Quantization level (e.g., 'Q4_0', 'Q8_0')

    Returns:
        Path to the downloaded GGUF file
    """
    from huggingface_hub import hf_hub_download

    # Map quantization levels to filenames (lowercase 'qwen-image')
    gguf_files = {
        "Q2_K": "qwen-image-Q2_K.gguf",
        "Q3_K_S": "qwen-image-Q3_K_S.gguf",
        "Q3_K_M": "qwen-image-Q3_K_M.gguf",
        "Q4_0": "qwen-image-Q4_0.gguf",
        "Q4_1": "qwen-image-Q4_1.gguf",
        "Q4_K_S": "qwen-image-Q4_K_S.gguf",
        "Q4_K_M": "qwen-image-Q4_K_M.gguf",
        "Q5_0": "qwen-image-Q5_0.gguf",
        "Q5_1": "qwen-image-Q5_1.gguf",
        "Q5_K_S": "qwen-image-Q5_K_S.gguf",
        "Q5_K_M": "qwen-image-Q5_K_M.gguf",
        "Q6_K": "qwen-image-Q6_K.gguf",
        "Q8_0": "qwen-image-Q8_0.gguf",
    }

    if quantization not in gguf_files:
        raise ValueError(f"Unsupported quantization level: {quantization}")

    filename = gguf_files[quantization]
    print(f"Downloading GGUF model with {quantization} quantization...")

    try:
        gguf_path = hf_hub_download(
            repo_id="city96/Qwen-Image-gguf",
            filename=filename,
            repo_type="model",
        )
        print(f"GGUF model downloaded: {gguf_path}")
        return gguf_path
    except Exception as e:
        print(f"Failed to download GGUF model: {e}")
        return None


def load_gguf_pipeline(quantization: str, device, torch_dtype, edit_mode=False):
    """Load a GGUF quantized model pipeline.

    Args:
        quantization: Quantization level (e.g., 'Q4_0')
        device: Device to load model on
        torch_dtype: Data type for computation
        edit_mode: Whether to load edit pipeline

    Returns:
        Loaded pipeline or None if failed

    Note:
        Currently only quantizes the transformer/diffusion model (main component).
        Text encoder (Qwen2.5-VL-7B, ~16.6GB) and VAE remain at full precision.

        Text encoder quantization is not yet supported because:
        - The text encoder is a full Qwen2.5-VL model, not a simple CLIP model
        - GGUF files from unsloth are for the full VLM, not optimized for diffusion use
        - Diffusers doesn't yet support loading GGUF text encoders via from_single_file

        Future versions may support quantized text encoders for additional memory savings.
    """

    try:
        from diffusers import GGUFQuantizationConfig
    except ImportError:
        print(
            "Error: GGUFQuantizationConfig not found. Please update diffusers to version >=0.35.0"
        )
        return None

    # Get GGUF model path
    gguf_path = get_gguf_model_path(quantization)
    if not gguf_path:
        return None

    # Create quantization config
    quantization_config = GGUFQuantizationConfig(compute_dtype=torch_dtype)

    if edit_mode:
        print("Note: GGUF quantized models for editing are not yet supported.")
        print("The GGUF models from city96 are for the base Qwen-Image model only.")
        return None
    else:
        # Load GGUF model for generation
        from diffusers import DiffusionPipeline

        print(f"Loading GGUF quantized model ({quantization})...")
        mem = get_total_memory_estimate(quantization)
        if mem is not None:
            print(
                "Estimated on-device memory (transformer + text encoder + VAE): "
                f"{mem['formatted']}"
            )
        else:
            print(
                f"Estimated transformer memory: {get_model_size(quantization)}; "
                "text encoder (~16.6 GB) and VAE (~1.2 GB) additionally."
            )

        try:
            # Try to load using the QwenImageTransformer2DModel class
            from diffusers.models import QwenImageTransformer2DModel

            print("Loading transformer from GGUF file...")
            # Try loading with different approaches
            try:
                # First try: Load with quantization config
                transformer = QwenImageTransformer2DModel.from_single_file(
                    gguf_path,
                    quantization_config=quantization_config,
                    torch_dtype=torch_dtype,
                    config="Qwen/Qwen-Image",
                    subfolder="transformer",
                )
            except Exception as e1:
                print(f"First attempt failed: {e1}")
                try:
                    # Second try: Load without quantization config
                    transformer = QwenImageTransformer2DModel.from_single_file(
                        gguf_path,
                        torch_dtype=torch_dtype,
                        config="Qwen/Qwen-Image",
                        subfolder="transformer",
                    )
                except Exception as e2:
                    print(f"Second attempt failed: {e2}")
                    # Third try: Load with minimal config
                    transformer = QwenImageTransformer2DModel.from_single_file(
                        gguf_path,
                        torch_dtype=torch_dtype,
                    )

            print("Creating pipeline with quantized transformer...")

            pipeline = DiffusionPipeline.from_pretrained(
                "Qwen/Qwen-Image",
                transformer=transformer,
                torch_dtype=torch_dtype,
            )

            pipeline = pipeline.to(device)

            # Enable memory optimizations
            # pipeline.enable_attention_slicing(slice_size=1)
            # pipeline.enable_vae_slicing()

            print(f"Successfully loaded GGUF model with {quantization} quantization")
            return pipeline

        except (ImportError, AttributeError) as e:
            print(f"Error: Could not import QwenImageTransformer2DModel: {e}")
            print(
                "This might be because your diffusers version doesn't support GGUF for Qwen-Image yet."
            )
            print("Falling back to standard transformer...")

            # Fallback: Use standard transformer and standard text encoder
            pipeline = DiffusionPipeline.from_pretrained(
                "Qwen/Qwen-Image",
                torch_dtype=torch_dtype,
            )
            pipeline = pipeline.to(device)
            print("Successfully loaded standard model")
            return pipeline

        except Exception as e:
            print(f"Error loading GGUF model: {e}")
            print(
                "The GGUF file format might not be compatible with the current diffusers implementation."
            )
            print("Falling back to standard transformer...")

            # Fallback: Use standard transformer and standard text encoder
            pipeline = DiffusionPipeline.from_pretrained(
                "Qwen/Qwen-Image",
                torch_dtype=torch_dtype,
            )
            pipeline = pipeline.to(device)
            print("Successfully loaded standard model")
            return pipeline


def load_quantized_text_encoder(quantization: str, device, torch_dtype):
    """Load a quantized text encoder using transformers native quantization.

    Args:
        quantization: Quantization level (e.g., '4bit', '8bit')
        device: Device to load on
        torch_dtype: Data type

    Returns:
        Quantized text encoder or None
    """

    try:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        if quantization == "4bit":
            # 4-bit quantization using bitsandbytes
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
        elif quantization == "8bit":
            # 8-bit quantization
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            print(f"Unsupported quantization type: {quantization}")
            return None

        print(f"Loading text encoder with {quantization} quantization...")
        return None

    except ImportError:
        print("bitsandbytes is required for text encoder quantization")
        print("Install with: pip install bitsandbytes")
        return None


def get_text_encoder_gguf_path(quantization: str):
    """Download and return path to GGUF quantized text encoder.

    Args:
        quantization: Quantization level (e.g., 'Q4_0', 'Q8_0')

    Returns:
        Path to the downloaded GGUF file or None if failed

    Note:
        Downloads from unsloth/Qwen2.5-VL-7B-Instruct-GGUF which contains
        quantized versions of the Qwen2.5-VL-7B model that can be used as
        text encoders with custom implementation.
    """
    from huggingface_hub import hf_hub_download

    # Map quantization levels to filenames for Qwen2.5-VL-7B
    # Using unsloth repository with correct case-sensitive filenames
    gguf_files = {
        "Q2_K": "Qwen2.5-VL-7B-Instruct-Q2_K.gguf",
        "Q3_K_S": "Qwen2.5-VL-7B-Instruct-Q3_K_S.gguf",
        "Q3_K_M": "Qwen2.5-VL-7B-Instruct-Q3_K_M.gguf",
        "Q4_0": "Qwen2.5-VL-7B-Instruct-Q4_0.gguf",
        "Q4_1": "Qwen2.5-VL-7B-Instruct-Q4_1.gguf",
        "Q4_K_S": "Qwen2.5-VL-7B-Instruct-Q4_K_S.gguf",
        "Q4_K_M": "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
        "Q5_0": "Qwen2.5-VL-7B-Instruct-Q5_0.gguf",
        "Q5_1": "Qwen2.5-VL-7B-Instruct-Q5_1.gguf",
        "Q5_K_S": "Qwen2.5-VL-7B-Instruct-Q5_K_S.gguf",
        "Q5_K_M": "Qwen2.5-VL-7B-Instruct-Q5_K_M.gguf",
        "Q6_K": "Qwen2.5-VL-7B-Instruct-Q6_K.gguf",
        "Q8_0": "Qwen2.5-VL-7B-Instruct-Q8_0.gguf",
    }

    if quantization not in gguf_files:
        print(f"Unsupported text encoder quantization level: {quantization}")
        return None

    filename = gguf_files[quantization]
    print(f"Downloading GGUF text encoder with {quantization} quantization...")

    try:
        gguf_path = hf_hub_download(
            repo_id="unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
            filename=filename,
            repo_type="model",
        )
        print(f"GGUF text encoder downloaded: {gguf_path}")
        return gguf_path
    except Exception as e:
        print(f"Error downloading GGUF text encoder: {e}")
        return None


class QwenTextEncoderGGUF:
    """Custom text encoder that loads from GGUF files.

    This class provides a wrapper around the Qwen2.5-VL model loaded from GGUF
    to be used as a text encoder in diffusion pipelines.
    """

    def __init__(self, gguf_path: str, device, torch_dtype):
        """Initialize the GGUF text encoder.

        Args:
            gguf_path: Path to the GGUF file
            device: Device to load on
            torch_dtype: Data type for computation
        """
        self.device = device
        self.torch_dtype = torch_dtype
        self.gguf_path = gguf_path

        # Load the model using transformers
        self._load_model()

    def _load_model(self):
        """Load the GGUF model using transformers."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"Loading GGUF text encoder from {self.gguf_path}...")

            # Load the model and tokenizer
            self.model = AutoModelForCausalLM.from_pretrained(
                self.gguf_path,
                torch_dtype=self.torch_dtype,
                device_map="auto" if self.device != "cpu" else None,
                trust_remote_code=True,
            )

            # Load tokenizer from the original model
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True
            )

            if self.device == "cpu":
                self.model = self.model.to(self.device)

            print("GGUF text encoder loaded successfully")

        except Exception as e:
            print(f"Error loading GGUF text encoder: {e}")
            raise

    def encode(self, text, return_tensors="pt"):
        """Encode text to embeddings.

        Args:
            text: Input text to encode
            return_tensors: Format to return tensors in

        Returns:
            Encoded text embeddings
        """
        try:
            # Tokenize the input text
            inputs = self.tokenizer(
                text,
                return_tensors=return_tensors,
                padding=True,
                truncation=True,
                max_length=512,  # Reasonable limit for text encoder
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get embeddings from the model
            import torch

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # Use the last hidden state as embeddings
                embeddings = outputs.hidden_states[-1]

            return embeddings

        except Exception as e:
            print(f"Error encoding text: {e}")
            raise

    def __call__(self, text, return_tensors="pt"):
        """Make the encoder callable."""
        return self.encode(text, return_tensors)


def load_gguf_text_encoder(quantization: str, device, torch_dtype):
    """Load a GGUF quantized text encoder.

    Args:
        quantization: Quantization level (e.g., 'Q4_0')
        device: Device to load on
        torch_dtype: Data type for computation

    Returns:
        QwenTextEncoderGGUF instance or None if failed
    """
    try:
        # Get the GGUF file path
        gguf_path = get_text_encoder_gguf_path(quantization)
        if not gguf_path:
            return None

        # Create the custom text encoder
        text_encoder = QwenTextEncoderGGUF(gguf_path, device, torch_dtype)
        return text_encoder

    except Exception as e:
        print(f"Error loading GGUF text encoder: {e}")
        return None


def get_model_size(quantization: str) -> str:
    """Get approximate model size for a quantization level."""
    sizes = {
        "Q2_K": "7.06 GB",
        "Q3_K_S": "8.95 GB",
        "Q3_K_M": "9.68 GB",
        "Q4_0": "11.9 GB",
        "Q4_1": "12.8 GB",
        "Q4_K_S": "12.1 GB",
        "Q4_K_M": "13.1 GB",
        "Q5_0": "14.4 GB",
        "Q5_1": "15.4 GB",
        "Q5_K_S": "14.1 GB",
        "Q5_K_M": "14.9 GB",
        "Q6_K": "16.8 GB",
        "Q8_0": "21.8 GB",
    }
    return sizes.get(quantization, "Unknown")


def get_text_encoder_size(quantization: str) -> str:
    """Get approximate text encoder size.

    Text encoder GGUF is not supported in this pipeline; assume full precision.

    Historical/future notes (kept for reference):
    - When GGUF text encoders become supported, approximate sizes could be:
      Q2_K ~2.8 GB; Q3_K_S ~3.5 GB; Q3_K_M ~3.8 GB; Q4_0 ~4.4 GB; Q4_1 ~4.6 GB;
      Q4_K_S ~4.4 GB; Q4_K_M ~4.6 GB; Q5_0 ~5.2 GB; Q5_1 ~5.3 GB; Q5_K_S ~5.2 GB;
      Q5_K_M ~5.3 GB; Q6_K ~6.1 GB; Q8_0 ~8.1 GB.
    - See `get_text_encoder_gguf_path` docstring for context on unsloth GGUF files
      and why integration requires custom support similar to ComfyUI-GGUF.

    Currently returning full-precision default (16.6 GB) for accuracy.
    """
    return "16.6 GB"


def get_total_memory_estimate(quantization: str):
    """Estimate end-to-end on-device memory usage for GGUF mode.

    Includes:
    - Quantized transformer (per `get_model_size`) loaded via GGUF
    - Quantized text encoder (per `get_text_encoder_size`) loaded via GGUF
    - Full-precision VAE and overhead (approx 1.2 GB)

    Returns a dict with numeric GB values and a formatted string.
    If the quantization value is unknown, returns None.
    """
    transformer_str = get_model_size(quantization)
    text_encoder_str = get_text_encoder_size(quantization)

    if transformer_str == "Unknown":
        return None

    try:
        transformer_gb = float(transformer_str.replace(" GB", "").strip())
        text_encoder_gb = float(text_encoder_str.replace(" GB", "").strip())
    except Exception:
        return None

    vae_and_overhead_gb = 1.2
    total_gb = round(transformer_gb + text_encoder_gb + vae_and_overhead_gb, 1)

    return {
        "transformer_gb": transformer_gb,
        "text_encoder_gb": text_encoder_gb,
        "vae_gb": vae_and_overhead_gb,
        "total_gb": total_gb,
        "breakdown": (
            f"~{transformer_gb} (transformer) + ~{text_encoder_gb} (text encoder) + "
            f"~{vae_and_overhead_gb} (VAE/overhead)"
        ),
        "formatted": (
            f"~{total_gb} GB (transformer ~{transformer_gb} GB + "
            f"text encoder ~{text_encoder_gb} GB + "
            f"VAE/overhead ~{vae_and_overhead_gb} GB)"
        ),
    }


def create_generator(device, seed):
    """Create a torch.Generator with the appropriate device."""

    generator_device = "cpu" if device == "mps" else device
    return torch.Generator(device=generator_device).manual_seed(seed)


def generate_image(args):
    from diffusers import DiffusionPipeline

    # Get the event callback if provided
    event_callback = getattr(args, "event_callback", None)

    # Helper function to yield events directly
    def emit_event(step: GenerationStep):
        if event_callback:
            try:
                event_callback(step)
            except Exception as e:
                print(f"Warning: Event callback error: {e}")
        # Always yield the step for generator consumers
        return step

    try:
        yield emit_event(GenerationStep.INIT)

        model_name = "Qwen/Qwen-Image"
        device, torch_dtype = get_device_and_dtype()

        yield emit_event(GenerationStep.LOADING_MODEL)

        # Check if quantization is requested
        quantization = getattr(args, "quantization", None)

        if quantization:
            # Load GGUF quantized model
            pipe = load_gguf_pipeline(
                quantization, device, torch_dtype, edit_mode=False
            )
            if pipe is None:
                print("Failed to load GGUF model, falling back to standard model...")
                pipe = DiffusionPipeline.from_pretrained(
                    model_name, torch_dtype=torch_dtype
                )
                pipe = pipe.to(device)
        else:
            # Load standard model
            pipe = DiffusionPipeline.from_pretrained(
                model_name, torch_dtype=torch_dtype
            )
            pipe = pipe.to(device)

        # pipe.enable_attention_slicing(slice_size=1)
        # pipe.enable_vae_slicing()
        yield emit_event(GenerationStep.MODEL_LOADED)

        # Check if using GGUF model
        using_gguf = quantization is not None

        # Apply custom LoRA if specified (skip if using GGUF due to incompatibility)
        if args.lora:
            if using_gguf:
                print(
                    "Warning: LoRA loading is not supported with GGUF quantized models."
                )
                print(
                    "The internal structure of GGUF models differs from standard models."
                )
                print("Continuing without LoRA...")
            else:
                yield emit_event(GenerationStep.LOADING_CUSTOM_LORA)
                print(f"Loading custom LoRA: {args.lora}")
                custom_lora_path = get_custom_lora_path(args.lora)
                if custom_lora_path:
                    pipe = merge_lora_from_safetensors(pipe, custom_lora_path)
                    yield emit_event(GenerationStep.LORA_LOADED)
                else:
                    print(
                        "Warning: Could not load custom LoRA, continuing without it..."
                    )

        # Apply Lightning LoRA if fast or ultra-fast mode is enabled (skip if using GGUF)
        if args.ultra_fast:
            if using_gguf:
                print(
                    "Warning: Lightning LoRA is not compatible with GGUF quantized models."
                )
                print("Using GGUF model with standard inference settings...")
                num_steps = 4  # Still use fewer steps for speed
                cfg_scale = 1.0
            else:
                yield emit_event(GenerationStep.LOADING_ULTRA_FAST_LORA)
                print("Loading Lightning LoRA v1.0 for ultra-fast generation...")
                lora_path = get_lora_path(ultra_fast=True)
                if lora_path:
                    pipe = merge_lora_from_safetensors(pipe, lora_path)
                    # Use fixed 4 steps for Ultra Lightning mode
                    num_steps = 4
                    cfg_scale = 1.0
                    yield emit_event(GenerationStep.LORA_LOADED)
                    print(
                        f"Ultra-fast mode enabled: {num_steps} steps, CFG scale {cfg_scale}"
                    )
                else:
                    print("Warning: Could not load Lightning LoRA v1.0")
                    print("Falling back to normal generation...")
                    num_steps = args.steps
                    cfg_scale = 4.0
        elif args.fast:
            if using_gguf:
                print(
                    "Warning: Lightning LoRA is not compatible with GGUF quantized models."
                )
                print("Using GGUF model with reduced steps for faster generation...")
                num_steps = 8  # Still use fewer steps for speed
                cfg_scale = 1.0
            else:
                yield emit_event(GenerationStep.LOADING_FAST_LORA)
                print("Loading Lightning LoRA for fast generation...")
                lora_path = get_lora_path(ultra_fast=False)
                if lora_path:
                    pipe = merge_lora_from_safetensors(pipe, lora_path)
                    # Use fixed 8 steps for Lightning mode
                    num_steps = 8
                    cfg_scale = 1.0
                    yield emit_event(GenerationStep.LORA_LOADED)
                    print(
                        f"Fast mode enabled: {num_steps} steps, CFG scale {cfg_scale}"
                    )
                else:
                    print("Warning: Could not load Lightning LoRA v2.0")
                    print("Falling back to normal generation...")
                    num_steps = args.steps
                    cfg_scale = 4.0
        else:
            num_steps = args.steps
            cfg_scale = 4.0

        # Override CFG scale if provided by user
        if getattr(args, "cfg_scale", None) is not None:
            try:
                cfg_scale = float(args.cfg_scale)
                if cfg_scale < 0:
                    cfg_scale = 0.0
            except Exception:
                pass

        # LEGO Batman photobomb mode!
        if args.batman:
            yield emit_event(GenerationStep.BATMAN_MODE_ACTIVATED)

            batman_additions = [
                ", with a tiny LEGO Batman minifigure photobombing in the corner doing a dramatic cape pose",
                ", featuring a small LEGO Batman minifigure sneaking into the frame from the side",
                ", and a miniature LEGO Batman figure peeking from behind something",
                ", with a tiny LEGO Batman minifigure in the background striking a heroic pose",
                ", including a small LEGO Batman figure hanging upside down from the top of the frame",
                ", with a tiny LEGO Batman minifigure doing the Batusi dance in the corner",
                ", and a small LEGO Batman figure photobombing with jazz hands",
                ", featuring a miniature LEGO Batman popping up from the bottom like 'I'm Batman!'",
                ", with a tiny LEGO Batman minifigure sliding into frame on a grappling hook",
                ", and a small LEGO Batman figure in the distance shouting 'WHERE ARE THEY?!'",
            ]
            print("\n🦇 BATMAN MODE ACTIVATED: Adding surprise LEGO Batman photobomb!")

        yield emit_event(GenerationStep.PREPARING_GENERATION)

        # Negative prompt: allow CLI override; default to empty when not provided
        neg_from_args = getattr(args, "negative_prompt", None)
        negative_prompt = " " if neg_from_args is None else neg_from_args

        aspect_ratios = {
            "1:1": (1328, 1328),
            "16:9": (1664, 928),
            "9:16": (928, 1664),
            "4:3": (1472, 1104),
            "3:4": (1104, 1472),
            "3:2": (1584, 1056),
            "2:3": (1056, 1584),
        }

        width, height = aspect_ratios[args.aspect]

        # Ensure we generate at least one image
        num_images = max(1, int(args.num_images))

        # Shared timestamp for this generation batch
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        saved_paths = []
        saved_seeds = []

        for image_index in range(num_images):
            if args.seed is not None:
                # Deterministic: increment seed per image starting from the provided seed
                per_image_seed = int(args.seed) + image_index
            else:
                # Random seed for each image when no seed is provided
                # Use 63-bit to keep it positive and well within torch's expected range
                per_image_seed = secrets.randbits(63)

            # Choose a random Batman prompt for each image when in Batman mode
            current_prompt = args.prompt
            if args.batman:
                batman_action = random.choice(batman_additions)
                current_prompt = current_prompt + batman_action
                if num_images > 1:
                    print(
                        f"  Image {image_index + 1}: Using Batman variant - {batman_action[2:50]}..."
                    )

            generator = create_generator(device, per_image_seed)

            yield emit_event(GenerationStep.INFERENCE_START)
            with torch.inference_mode():
                image = pipe(
                    prompt=current_prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_steps,
                    true_cfg_scale=cfg_scale,
                    generator=generator,
                ).images[0]
            yield emit_event(GenerationStep.INFERENCE_COMPLETE)

            # Save with timestamp to avoid overwriting previous generations
            yield emit_event(GenerationStep.SAVING_IMAGE)
            suffix = f"-{image_index+1}" if num_images > 1 else ""

            default_output_dir = getattr(args, "output_dir", None) or "output"

            # Check if we have custom output path from args
            if hasattr(args, "output_path") and args.output_path:
                output_filename = args.output_path
            else:
                output_filename = os.path.join(
                    default_output_dir, f"image-{timestamp}{suffix}.png"
                )

            os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)

            image.save(output_filename)
            abs_path = os.path.abspath(output_filename)
            saved_paths.append(abs_path)
            saved_seeds.append(per_image_seed)
            yield emit_event(GenerationStep.IMAGE_SAVED)

        # Print full path(s) of saved image(s)
        if len(saved_paths) == 1:
            print(f"\nImage saved to: {saved_paths[0]} (seed: {saved_seeds[0]})")
        else:
            print("\nImages saved:")
            for path, seed_val in zip(saved_paths, saved_seeds):
                print(f"- {path} (seed: {seed_val})")

        yield emit_event(GenerationStep.COMPLETE)

        # Yield the final result so pipeline can catch it
        yield saved_paths

    except Exception as e:
        yield emit_event(GenerationStep.ERROR)
        print(f"Error during image generation: {e}")
        raise


def edit_image(args) -> None:
    from PIL import Image

    try:
        from diffusers import QwenImageEditPlusPipeline as EditPipeline
    except ImportError:
        from diffusers import QwenImageEditPipeline as EditPipeline

    device, torch_dtype = get_device_and_dtype()

    # Check if quantization is requested
    quantization = getattr(args, "quantization", None)
    if quantization:
        # Load GGUF quantized model for editing
        print(f"Loading GGUF quantized model ({quantization}) for image editing...")
        pipeline = load_gguf_pipeline(quantization, device, torch_dtype, edit_mode=True)
        if pipeline is None:
            print("GGUF models for editing may not be available yet.")
            print("Falling back to standard edit model...")
            pipeline = EditPipeline.from_pretrained(
                "Qwen/Qwen-Image-Edit-2509", torch_dtype=torch_dtype
            )
            pipeline = pipeline.to(device)
    else:
        # Load the standard image editing pipeline
        print("Loading Qwen-Image-Edit model for image editing...")
        pipeline = EditPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2509", torch_dtype=torch_dtype
        )
        pipeline = pipeline.to(device)

    pipeline.set_progress_bar_config(disable=None)

    # Apply custom LoRA if specified
    if args.lora:
        print(f"Loading custom LoRA: {args.lora}")
        custom_lora_path = get_custom_lora_path(args.lora)
        if custom_lora_path:
            pipeline = merge_lora_from_safetensors(pipeline, custom_lora_path)
        else:
            print("Warning: Could not load custom LoRA, continuing without it...")

    # Apply Lightning LoRA if fast or ultra-fast mode is enabled
    if args.ultra_fast:
        print("Loading Lightning Edit LoRA v1.0 (4 steps) for ultra-fast editing...")
        lora_path = get_lora_path(ultra_fast=True, edit_mode=True)
        if lora_path:
            # Use manual LoRA merging for edit pipeline
            pipeline = merge_lora_from_safetensors(pipeline, lora_path)
            # Use fixed 4 steps for Ultra Lightning mode
            num_steps = 4
            cfg_scale = 1.0
            print(f"Ultra-fast mode enabled: {num_steps} steps, CFG scale {cfg_scale}")
        else:
            print("Warning: Could not load Lightning Edit LoRA v1.0 (4 steps)")
            print("Falling back to normal editing...")
            num_steps = args.steps
            cfg_scale = 4.0
    elif args.fast:
        print("Loading Lightning Edit LoRA v1.0 for fast editing...")
        lora_path = get_lora_path(edit_mode=True)
        if lora_path:
            # Use manual LoRA merging for edit pipeline
            pipeline = merge_lora_from_safetensors(pipeline, lora_path)
            # Use fixed 8 steps for Lightning Edit mode
            num_steps = 8
            cfg_scale = 1.0
            print(f"Fast edit mode enabled: {num_steps} steps, CFG scale {cfg_scale}")
        else:
            print("Warning: Could not load Lightning Edit LoRA v1.0")
            print("Falling back to normal editing...")
            num_steps = args.steps
            cfg_scale = 4.0
    else:
        num_steps = args.steps
        cfg_scale = 4.0

    # Override CFG scale if provided by user
    if getattr(args, "cfg_scale", None) is not None:
        try:
            cfg_scale = float(args.cfg_scale)
            if cfg_scale < 0:
                cfg_scale = 0.0
        except Exception:
            pass

    # Load input image
    input_paths = args.input if isinstance(args.input, (list, tuple)) else [args.input]

    images = []
    try:
        for path in input_paths:
            image = Image.open(path).convert("RGB")
            print(f"Loaded input image: {path} ({image.size[0]}x{image.size[1]})")
            images.append(image)
    except Exception as e:
        print(f"Error loading input image: {e}")
        return

    # Set up generation parameters
    seed = args.seed if args.seed is not None else secrets.randbits(63)
    generator = create_generator(device, seed)

    # Modify prompt for Batman photobomb mode
    edit_prompt = args.prompt
    if args.batman:
        import random

        batman_edits = [
            " Also add a tiny LEGO Batman minifigure photobombing somewhere unexpected.",
            " Include a small LEGO Batman figure sneaking into the scene.",
            " Add a miniature LEGO Batman peeking from an edge.",
            " Put a tiny LEGO Batman minifigure doing something heroic in the background.",
            " Add a small LEGO Batman figure photobombing with a dramatic pose.",
            " Include a tiny LEGO Batman minifigure who looks like he's saying 'I'm Batman!'",
            " Add a miniature LEGO Batman swinging on a tiny grappling hook.",
            " Include a small LEGO Batman figure doing the Batusi dance.",
            " Add a tiny LEGO Batman minifigure brooding mysteriously in a corner.",
            " Put a small LEGO Batman photobombing like he's protecting Gotham.",
        ]
        batman_edit = random.choice(batman_edits)
        edit_prompt = args.prompt + batman_edit
        print("\n🦇 BATMAN MODE ACTIVATED: LEGO Batman will photobomb this edit!")

    # Prepare negative prompt (allow CLI override; default to empty)
    edit_negative_prompt = (
        " " if getattr(args, "negative_prompt", None) is None else args.negative_prompt
    )

    # Perform image editing
    print(f"Editing image with prompt: {edit_prompt}")
    print(f"Using {num_steps} inference steps...")

    # Qwen image edit pipeline for image editing
    with torch.inference_mode():
        pipeline_inputs = images if len(images) > 1 else images[0]
        output = pipeline(
            image=pipeline_inputs,
            prompt=edit_prompt,
            negative_prompt=edit_negative_prompt,
            num_inference_steps=num_steps,
            generator=generator,
            guidance_scale=cfg_scale,
        )
        edited_image = output.images[0]

    # Save the edited image
    default_output_dir = getattr(args, "output_dir", None) or "output"

    if args.output:
        output_filename = args.output
        if os.path.basename(output_filename) == output_filename:
            output_filename = os.path.join(default_output_dir, output_filename)
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_filename = os.path.join(default_output_dir, f"edited-{timestamp}.png")

    os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)

    edited_image.save(output_filename)
    print(f"\nEdited image saved to: {os.path.abspath(output_filename)} (seed: {seed})")


def main() -> None:
    try:
        from . import __version__
    except ImportError:
        # Fallback when module is loaded without package context
        __version__ = "0.4.5"

    parser = argparse.ArgumentParser(
        description="Qwen-Image MPS - Generate and edit images with Qwen models on Apple Silicon",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"qwen-image-mps {__version__}",
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Add generate and edit subcommands
    build_generate_parser(subparsers)
    build_edit_parser(subparsers)

    args = parser.parse_args()

    # Handle the command
    if args.command == "generate":
        # Consume the generator to execute the image generation
        list(generate_image(args))
    elif args.command == "edit":
        edit_image(args)
    else:
        # Default to generate for backward compatibility if no subcommand
        # This allows the old style invocation to still work
        import sys

        if len(sys.argv) > 1 and sys.argv[1] not in [
            "generate",
            "edit",
            "-h",
            "--help",
        ]:
            # Parse as generate command for backward compatibility
            sys.argv.insert(1, "generate")
            args = parser.parse_args()
            # Consume the generator to execute the image generation
            list(generate_image(args))
        else:
            parser.print_help()


if __name__ == "__main__":
    main()

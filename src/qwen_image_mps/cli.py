import argparse
import os
import secrets
import sys
from datetime import datetime


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
        default="""A coffee shop entrance features a chalkboard sign reading "Apple Silicon Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "Generated with MPS on Apple Silicon". Next to it hangs a poster showing a beautiful Italian woman, and beneath the poster is written "Just try it!". Ultra HD, 4K, cinematic composition""",
        help="Prompt text to condition the image generation.",
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
        help="Use Lightning LoRA v1.1 for fast generation (8 steps).",
    )
    parser.add_argument(
        "-uf",
        "--ultra-fast",
        action="store_true",
        help="Use Lightning LoRA v1.0 for ultra-fast generation (4 steps).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for reproducible generation (default: 42). If not explicitly "
            "provided and generating multiple images, a new random seed is used for each image."
        ),
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="Number of images to generate.",
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
    return parser


def get_lora_path(ultra_fast=False):
    from huggingface_hub import hf_hub_download

    """Get the Lightning LoRA from Hugging Face Hub with a silent cache freshness check.

    The function will:
    - Look up any locally cached file for the target filename.
    - Then fetch the latest from the Hub (without forcing) which will reuse cache
      if up-to-date, or download a newer snapshot if the remote changed.
    - Return the final resolved local path.
    """

    if ultra_fast:
        filename = "Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors"
        version = "v1.0 (4-steps)"
    else:
        filename = "Qwen-Image-Lightning-8steps-V1.1.safetensors"
        version = "v1.1 (8-steps)"

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
    import torch

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
        required=True,
        help="Path to the input image to edit.",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        required=True,
        help="Editing instructions (e.g., 'Change the sky to sunset colors').",
    )
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=50,
        help="Number of inference steps for normal editing.",
    )
    parser.add_argument(
        "-f",
        "--fast",
        action="store_true",
        help="Use Lightning LoRA v1.1 for fast editing (8 steps).",
    )
    parser.add_argument(
        "-uf",
        "--ultra-fast",
        action="store_true",
        help="Use Lightning LoRA v1.0 for ultra-fast editing (4 steps).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output filename (default: edited-<timestamp>.png).",
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
    return parser


def get_device_and_dtype():
    """Get the optimal device and dtype for the current system."""
    import torch

    if torch.backends.mps.is_available():
        print("Using MPS")
        return "mps", torch.bfloat16
    elif torch.cuda.is_available():
        print("Using CUDA")
        return "cuda", torch.bfloat16
    else:
        print("Using CPU")
        return "cpu", torch.float32


def create_generator(device, seed):
    """Create a torch.Generator with the appropriate device."""
    import torch

    generator_device = "cpu" if device == "mps" else device
    return torch.Generator(device=generator_device).manual_seed(seed)


def generate_image(args) -> None:
    from diffusers import DiffusionPipeline

    model_name = "Qwen/Qwen-Image"
    device, torch_dtype = get_device_and_dtype()

    pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
    pipe = pipe.to(device)

    # Apply custom LoRA if specified
    if args.lora:
        print(f"Loading custom LoRA: {args.lora}")
        custom_lora_path = get_custom_lora_path(args.lora)
        if custom_lora_path:
            pipe = merge_lora_from_safetensors(pipe, custom_lora_path)
        else:
            print("Warning: Could not load custom LoRA, continuing without it...")

    # Apply Lightning LoRA if fast or ultra-fast mode is enabled
    if args.ultra_fast:
        print("Loading Lightning LoRA v1.0 for ultra-fast generation...")
        lora_path = get_lora_path(ultra_fast=True)
        if lora_path:
            pipe = merge_lora_from_safetensors(pipe, lora_path)
            # Use fixed 4 steps for Ultra Lightning mode
            num_steps = 4
            cfg_scale = 1.0
            print(f"Ultra-fast mode enabled: {num_steps} steps, CFG scale {cfg_scale}")
        else:
            print("Warning: Could not load Lightning LoRA v1.0")
            print("Falling back to normal generation...")
            num_steps = args.steps
            cfg_scale = 4.0
    elif args.fast:
        print("Loading Lightning LoRA v1.1 for fast generation...")
        lora_path = get_lora_path(ultra_fast=False)
        if lora_path:
            pipe = merge_lora_from_safetensors(pipe, lora_path)
            # Use fixed 8 steps for Lightning mode
            num_steps = 8
            cfg_scale = 1.0
            print(f"Fast mode enabled: {num_steps} steps, CFG scale {cfg_scale}")
        else:
            print("Warning: Could not load Lightning LoRA v1.1")
            print("Falling back to normal generation...")
            num_steps = args.steps
            cfg_scale = 4.0
    else:
        num_steps = args.steps
        cfg_scale = 4.0

    prompt = args.prompt

    # LEGO Batman photobomb mode!
    if args.batman:
        import random

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
        batman_action = random.choice(batman_additions)
        prompt = prompt + batman_action
        print("\n🦇 BATMAN MODE ACTIVATED: Adding surprise LEGO Batman photobomb!")

    negative_prompt = (
        " "  # using an empty string if you do not have specific concept to remove
    )

    aspect_ratios = {
        "1:1": (1328, 1328),
        "16:9": (1664, 928),
        "9:16": (928, 1664),
        "4:3": (1472, 1140),
        "3:4": (1140, 1472),
        "3:2": (1584, 1056),
        "2:3": (1056, 1584),
    }

    width, height = aspect_ratios["16:9"]

    # Ensure we generate at least one image
    num_images = max(1, int(args.num_images))

    # Shared timestamp for this generation batch
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    saved_paths = []

    # Detect whether the user explicitly provided --seed on the command line
    argv = sys.argv[1:]
    seed_provided = ("--seed" in argv) or any(arg.startswith("--seed=") for arg in argv)
    for image_index in range(num_images):
        if seed_provided:
            # Deterministic: increment seed per image starting from the provided seed
            per_image_seed = int(args.seed) + image_index
        elif num_images > 1:
            # Non-deterministic for multi-image when no seed explicitly provided
            # Use 63-bit to keep it positive and well within torch's expected range
            per_image_seed = secrets.randbits(63)
        else:
            # Single image without explicit seed: use default (42)
            per_image_seed = int(args.seed)

        generator = create_generator(device, per_image_seed)
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_steps,
            true_cfg_scale=cfg_scale,
            generator=generator,
        ).images[0]

        # Save with timestamp to avoid overwriting previous generations
        suffix = f"-{image_index+1}" if num_images > 1 else ""
        output_filename = f"image-{timestamp}{suffix}.png"
        image.save(output_filename)
        saved_paths.append(os.path.abspath(output_filename))

    # Print full path(s) of saved image(s)
    if len(saved_paths) == 1:
        print(f"\nImage saved to: {saved_paths[0]}")
    else:
        print("\nImages saved:")
        for path in saved_paths:
            print(f"- {path}")


def edit_image(args) -> None:
    import torch
    from diffusers import QwenImageEditPipeline
    from PIL import Image

    device, torch_dtype = get_device_and_dtype()

    # Load the image editing pipeline
    print("Loading Qwen-Image-Edit model for image editing...")
    pipeline = QwenImageEditPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit", torch_dtype=torch_dtype
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
        print("Loading Lightning LoRA v1.0 for ultra-fast editing...")
        lora_path = get_lora_path(ultra_fast=True)
        if lora_path:
            # Use manual LoRA merging for edit pipeline
            pipeline = merge_lora_from_safetensors(pipeline, lora_path)
            # Use fixed 4 steps for Ultra Lightning mode
            num_steps = 4
            cfg_scale = 1.0
            print(f"Ultra-fast mode enabled: {num_steps} steps, CFG scale {cfg_scale}")
        else:
            print("Warning: Could not load Lightning LoRA v1.0")
            print("Falling back to normal editing...")
            num_steps = args.steps
            cfg_scale = 4.0
    elif args.fast:
        print("Loading Lightning LoRA v1.1 for fast editing...")
        lora_path = get_lora_path(ultra_fast=False)
        if lora_path:
            # Use manual LoRA merging for edit pipeline
            pipeline = merge_lora_from_safetensors(pipeline, lora_path)
            # Use fixed 8 steps for Lightning mode
            num_steps = 8
            cfg_scale = 1.0
            print(f"Fast mode enabled: {num_steps} steps, CFG scale {cfg_scale}")
        else:
            print("Warning: Could not load Lightning LoRA v1.1")
            print("Falling back to normal editing...")
            num_steps = args.steps
            cfg_scale = 4.0
    else:
        num_steps = args.steps
        cfg_scale = 4.0

    # Load input image
    try:
        image = Image.open(args.input).convert("RGB")
        print(f"Loaded input image: {args.input} ({image.size[0]}x{image.size[1]})")
    except Exception as e:
        print(f"Error loading input image: {e}")
        return

    # Set up generation parameters
    generator = create_generator(device, args.seed)

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

    # Perform image editing
    print(f"Editing image with prompt: {edit_prompt}")
    print(f"Using {num_steps} inference steps...")

    # QwenImageEditPipeline for image editing
    with torch.inference_mode():
        output = pipeline(
            image=image,
            prompt=edit_prompt,
            negative_prompt=" ",
            num_inference_steps=num_steps,
            generator=generator,
            guidance_scale=cfg_scale,
        )
        edited_image = output.images[0]

    # Save the edited image
    if args.output:
        output_filename = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_filename = f"edited-{timestamp}.png"

    edited_image.save(output_filename)
    print(f"\nEdited image saved to: {os.path.abspath(output_filename)}")


def main() -> None:
    try:
        from . import __version__
    except ImportError:
        # Fallback when module is loaded without package context
        __version__ = "0.4.0"

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
        generate_image(args)
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
            generate_image(args)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()

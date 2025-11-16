## Qwen Image (MPS/CUDA/CPU)

Generate and edit images from text prompts using the Hugging Face Diffusers pipeline for `Qwen/Qwen-Image`, with automatic device selection for Apple Silicon (MPS), NVIDIA CUDA, or CPU fallback.

### Features
- **Auto device selection**: prefers MPS (Apple Silicon), then CUDA, else CPU
- **Simple CLI**: provide a prompt and number of steps
- **Image generation**: create new images from text prompts
- **Image editing**: modify existing images using text instructions
- **Fast editing**: 8-step and 4-step editing using Lightning LoRA
- **Timestamped outputs**: avoids overwriting previous generations
- **Clean outputs**: saves images to `output/` by default (configurable with `--outdir`)
- **Fast mode**: 8-step generation using Lightning LoRA (auto-downloads if needed)
- **Ultra-fast mode**: 4-step generation using Lightning LoRA (auto-downloads if needed)
- **Multi-image generation**: generate multiple images in one run with `--num-images`
- **Multi-image editing**: blend multiple input images in a single edit command
- **Batman mode**: Add a LEGO Batman minifigure photobombing your images with `--batman` 🦇

### Examples

Example generation result:

![Example image](https://raw.githubusercontent.com/ivanfioravanti/qwen-image-mps/main/example.png)

Example edit results showing winter transformation:

![Edit example](https://raw.githubusercontent.com/ivanfioravanti/qwen-image-mps/main/editexample.jpg)

Anime photo-to-anime example using `--anime`:

Input (`einstein.jpg`) vs. anime output (`anime-einstein.png`):

![Einstein photo](einstein.jpg)
![Anime Einstein](anime-einstein.png)

Photo-to-anime mode is powered by Choomba’s (X: [@begina​non](https://x.com/beginanon)) LoRA, available on Hugging Face as [`autoweeb/Qwen-Image-Edit-2509-Photo-to-Anime`](https://huggingface.co/autoweeb/Qwen-Image-Edit-2509-Photo-to-Anime).

## Installation

### Option 1: Install from PyPI (Recommended)

Install the package using pip:
```bash
pip install qwen-image-mps
```

Then run it directly from the command line:
```bash
qwen-image-mps --help
qwen-image-mps --version        # Show version
qwen-image-mps generate --help  # For image generation
qwen-image-mps edit --help      # For image editing
```

### Option 2: Direct script execution with uv

You can run this script directly using `uv run` without installation - it will install all dependencies automatically in an isolated environment:
```bash
uv run https://raw.githubusercontent.com/ivanfioravanti/qwen-image-mps/refs/heads/main/qwen-image-mps.py --help
```

Or download the file first:
```bash
curl -O https://raw.githubusercontent.com/ivanfioravanti/qwen-image-mps/refs/heads/main/qwen-image-mps.py
uv run qwen-image-mps.py --help
```

### Option 3: Install from source

Clone the repository and install in development mode:
```bash
git clone https://github.com/ivanfioravanti/qwen-image-mps.git
cd qwen-image-mps
pip install -e .
```

**Note:** The first time you run the tool, it will download the 57.7GB model from Hugging Face and store it in your `~/.cache/huggingface/hub/models--Qwen--Qwen-Image` directory.

## Usage

After installation, use the `qwen-image-mps` command with either `generate` or `edit` subcommands:

```bash
qwen-image-mps --help
qwen-image-mps --version        # Show version
qwen-image-mps generate --help  # For image generation
qwen-image-mps edit --help      # For image editing
```

### Image Generation Examples:

```bash
# Default prompt and steps
qwen-image-mps generate

# Custom prompt and fewer steps
qwen-image-mps generate -p "A serene alpine lake at sunrise, ultra detailed, cinematic" -s 30

# Fast mode with Lightning LoRA (8 steps)
qwen-image-mps generate -f -p "A magical forest with glowing mushrooms"

# Ultra-fast mode with Lightning LoRA (4 steps)
qwen-image-mps generate --ultra-fast -p "A magical forest with glowing mushrooms"
# Or use the short form
qwen-image-mps generate -uf -p "A magical forest with glowing mushrooms"


# Custom seed for reproducible generation
qwen-image-mps generate --seed 42 -p "A vintage coffee shop"

# Generate multiple images (incrementing seed per image when seed is provided)
qwen-image-mps generate -p "Retro sci-fi city skyline at night" --num-images 3 --seed 100

# Generate multiple images with a fresh random seed for each image (omit --seed)
qwen-image-mps generate -p "Retro sci-fi city skyline at night" --num-images 3

# Generate with a custom LoRA for anime style
qwen-image-mps generate -p "A magical forest" --lora flymy-ai/qwen-image-anime-irl-lora

# Generate with custom LoRA and fast mode combined
qwen-image-mps generate -p "A futuristic city" --lora your-username/your-lora-model --fast

# Use a negative prompt to discourage artifacts
qwen-image-mps generate -p "Portrait photo" --negative-prompt "blurry, watermark, text, low quality"
qwen-image-mps generate -p "Portrait photo" -np "blurry, watermark, text, low quality"

# Batman mode: LEGO Batman photobombs your image!
qwen-image-mps generate -p "A magical forest with elves" --batman

# Combine Batman mode with ultra-fast generation
qwen-image-mps generate -p "A serene mountain lake" --batman --ultra-fast

# Specify aspect ratio (default is 16:9)
qwen-image-mps generate -f -p "Cozy reading nook, soft morning light" --aspect 1:1
qwen-image-mps generate -f -p "Tall cyberpunk city street, neon rain" --aspect 9:16

# Save images into a custom directory
qwen-image-mps generate -p "A cozy cabin in the woods" --outdir my-outputs

# Override CFG scale (default 4.0 normal, 1.0 fast/ultra-fast)
qwen-image-mps generate -p "Portrait photo" --cfg-scale 2.5
```

### Image Editing Examples:

```bash
# Basic image editing (uses Rapid-AIO transformer, 4 steps by default)
qwen-image-mps edit -i input.jpg -p "Change the sky to sunset colors"

# Fast mode with Rapid-AIO transformer (8 steps)
qwen-image-mps edit -i photo.png -p "Add snow to the mountains" --fast

# Ultra-fast mode with Rapid-AIO transformer (4 steps)
qwen-image-mps edit -i landscape.jpg -p "Make it autumn colors" --ultra-fast
# Or use the short form
qwen-image-mps edit -i landscape.jpg -p "Make it autumn colors" -uf

# Edit with custom output filename
qwen-image-mps edit -i portrait.jpg -p "Change hair color to blonde" -o blonde_portrait.png

# Edit with custom seed and steps
qwen-image-mps edit -i scene.jpg -p "Add dramatic lighting" --seed 123 -s 30

# Edit with a custom LoRA for specific style
qwen-image-mps edit -i photo.jpg -p "Make it anime style" --lora flymy-ai/qwen-image-anime-irl-lora

# Edit with custom LoRA and ultra-fast mode combined
qwen-image-mps edit -i landscape.jpg -p "Add cyberpunk elements" --lora your-username/your-lora-model --ultra-fast

# Combine multiple input images in a single edit
qwen-image-mps edit -i input1.png input2.png -p "Merge the two scenes into a city skyline"

# Use a negative prompt during editing
qwen-image-mps edit -i photo.jpg -p "Studio portrait" --negative-prompt "blurry, watermark, text, low quality"

# Batman mode for editing: LEGO Batman photobombs your edited image!
qwen-image-mps edit -i photo.jpg -p "Change to sunset lighting" --batman

# Combine Batman mode with fast editing
qwen-image-mps edit -i portrait.jpg -p "Add dramatic shadows" --batman --fast

# Transform photo to anime style (prompt is optional with --anime) fast mode with Rapid-AIO transformer recommended 
qwen-image-mps edit -i photo.jpg --anime --fast

# Anime transformation with additional instructions
qwen-image-mps edit -i photo.jpg -p "make it colorful" --anime --fast

# Save edited image into a custom directory
qwen-image-mps edit -i photo.jpg -p "Add autumn colors" --outdir edits

# Override CFG scale for editing (default 4.0 normal, 1.0 fast/ultra-fast)
qwen-image-mps edit -i input.jpg -p "Studio portrait" --cfg-scale 2.0
```

If using the direct script with uv, replace `qwen-image-mps` with `uv run qwen-image-mps.py` in the examples above.

### Command Arguments

#### Generate Command Arguments
- `-p, --prompt` (str): Prompt text for image generation.
- `--negative-prompt` (str): Text to discourage (negative prompt), e.g. `"blurry, watermark, text, low quality"`.
- `-s, --steps` (int): Number of inference steps (default: 50).
- `-f, --fast`: Enable fast mode using Lightning LoRA for 8-step generation.
- `-uf, --ultra-fast`: Enable ultra-fast mode using Lightning LoRA v2.0 for 4-step generation.
- `--seed` (int): Random seed for reproducible generation (default: 42). If not
  explicitly provided and generating multiple images, a new random seed is used
  for each image.
- `--num-images` (int): Number of images to generate (default: 1).
- `--lora` (str): Hugging Face model URL or repo ID for additional LoRA to load
  (e.g., 'flymy-ai/qwen-image-anime-irl-lora' or full HF URL).
- `--batman`: Add a LEGO Batman minifigure photobombing your image in unexpected ways!
- `--outdir` (str): Directory to save generated images (default: `./output`).
- `--cfg-scale` (float): Classifier-free guidance scale (overrides mode defaults).

#### Edit Command Arguments
- `-i, --input` (str): Path(s) to the input image(s) to edit (required). Provide multiple paths to blend results.
- `-p, --prompt` (str): Editing instructions. Optional when `--anime` is used (defaults to anime transformation prompt).
- `--negative-prompt` (str): Text to discourage in the edit (negative prompt).
- `-s, --steps` (int): Number of inference steps for normal editing (default: 50).
- `-f, --fast`: Enable fast mode using Rapid-AIO transformer for 4-step editing.
- `-uf, --ultra-fast`: Enable ultra-fast mode using Rapid-AIO transformer for 4-step editing.
- `--seed` (int): Random seed for reproducible generation (default: 42).
- `-o, --output` (str): Output filename (default: edited-<timestamp>.png).
- `--outdir` (str): Directory to save edited images (default: `./output`). If `--output` is a basename, it is saved under this directory.
- `--cfg-scale` (float): Classifier-free guidance scale (overrides mode defaults).
- `--lora` (str): Hugging Face model URL or repo ID for additional LoRA to load
  (e.g., 'flymy-ai/qwen-image-anime-irl-lora' or full HF URL).
- `--batman`: Add a LEGO Batman minifigure photobombing your edited image!
- `--anime`: Transform photo to anime style using Photo-to-Anime LoRA. Can be combined with `--fast` or `--ultra-fast` for faster processing.

## What the script does

### Image Generation
- Loads `Qwen/Qwen-Image` via `diffusers.DiffusionPipeline`
- Selects device and dtype:
  - MPS: `bfloat16`
  - CUDA: `bfloat16`
  - CPU: `float32`
- Uses a light positive conditioning suffix for quality
- Generates at a 16:9 resolution (default `1664x928`)
- Saves images under `output/` by default. Filenames are `image-YYYYMMDD-HHMMSS.png` for a single image,
  or `image-YYYYMMDD-HHMMSS-1.png`, `image-YYYYMMDD-HHMMSS-2.png`, ... when using `--num-images`. Use `--outdir` to change the directory.
- Prints the full path of the saved image

### Image Editing
- Loads `Qwen/Qwen-Image-Edit-2509` via `QwenImageEditPlusPipeline` (falling back to `QwenImageEditPipeline` when needed)
- Uses Rapid-AIO transformer ([`linoyts/Qwen-Image-Edit-Rapid-AIO`](https://huggingface.co/linoyts/Qwen-Image-Edit-Rapid-AIO)) for optimized fast inference
- Takes an existing image and editing instructions as input
- Applies transformations while preserving the original structure
- Uses 4 inference steps by default (optimized for Rapid-AIO transformer)
- Saves the edited image under `output/` by default as `edited-YYYYMMDD-HHMMSS.png`, or to a custom filename.
- Prints the full path of the edited image

### Fast Mode & Ultra-Fast Mode

#### Image Editing (Rapid-AIO Transformer)
All image editing operations use the Rapid-AIO transformer ([`linoyts/Qwen-Image-Edit-Rapid-AIO`](https://huggingface.co/linoyts/Qwen-Image-Edit-Rapid-AIO)) for optimized fast inference:
- Automatically loads the Rapid-AIO transformer (cached in `~/.cache/huggingface/hub/`)
- Uses 4 inference steps by default with CFG scale 1.0
- Provides fast, high-quality image editing results

#### Fast Mode (`-f/--fast`) for Editing
When using the `-f/--fast` flag with edit command:
- Uses Rapid-AIO transformer (already loaded by default)
- Uses 4 inference steps with CFG scale 1.0 (Rapid-AIO optimized)

#### Ultra-Fast Mode (`-uf/--ultra-fast`) for Editing
When using the `-uf/--ultra-fast` flag with edit command:
- Uses Rapid-AIO transformer (already loaded by default)
- Uses 4 inference steps with CFG scale 1.0 (Rapid-AIO optimized)
- Both fast and ultra-fast modes use the same optimized 4-step inference

#### Image Generation (Lightning LoRA)
For image generation, the tool uses Lightning LoRA models:
- Fast mode (`-f/--fast`): Uses Lightning LoRA v2.0 for 8-step generation
- Ultra-fast mode (`-uf/--ultra-fast`): Uses Lightning LoRA v2.0 for 4-step generation
- The fast implementation is based on [Qwen-Image-Lightning](https://github.com/ModelTC/Qwen-Image-Lightning)
- Lightning LoRA models are available on HuggingFace at [lightx2v/Qwen-Image-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Lightning)

### Batman Mode 🦇

The `--batman` flag adds a fun twist to your image generation and editing by having a LEGO Batman minifigure photobomb your images! This feature works with both `generate` and `edit` commands.

When enabled, the tool randomly selects from various photobombing styles:
- LEGO Batman doing dramatic cape poses
- Sneaking into frame from the sides
- Peeking from behind objects
- Hanging upside down from the top
- Doing the Batusi dance
- Striking heroic poses
- Shouting his famous catchphrases

This feature adds a playful element to your images while keeping the main subject intact. The LEGO Batman appears small but noticeable, creating unexpected and humorous compositions.

#### Loading Additional LoRAs

##### Command Line Usage

The `--lora` argument allows you to load custom LoRA models from Hugging Face Hub:

```bash
# Using a repo ID
qwen-image-mps generate -p "Your prompt" --lora flymy-ai/qwen-image-anime-irl-lora

# Using a full Hugging Face URL
qwen-image-mps generate -p "Your prompt" --lora https://huggingface.co/flymy-ai/qwen-image-anime-irl-lora

# Combine with Lightning LoRA for both speed and style
qwen-image-mps generate -p "Your prompt" --lora your-username/style-lora --fast
```

The tool will automatically:
- Download the LoRA from Hugging Face Hub (cached locally)
- Find the appropriate safetensors file in the repository
- Merge the LoRA weights into the model
- Apply any Lightning LoRA if `--fast` or `--ultra-fast` is also specified


## Notes and tweaks
- **Aspect ratio / resolution**: Use `--aspect` to select output size. Available choices: `1:1`, `16:9`, `9:16`, `4:3`, `3:4`, `3:2`, `2:3`. Default is `16:9`.
- **Determinism**: Use the `--seed` parameter to control the random generator for reproducible results. On MPS, the random generator runs on CPU for improved stability.
- **Performance**: If you hit memory or speed issues, try reducing `--steps`.

## Troubleshooting
- If you see "Using CPU" in the console on Apple Silicon, ensure your PyTorch build includes MPS and you are running on Apple Silicon Python (not under Rosetta).
- If model download fails or is unauthorized, log in with `huggingface-cli login` or accept the model terms on the Hugging Face model page.

## Development

To contribute or modify the tool:

1. Clone the repository:
```bash
git clone https://github.com/ivanfioravanti/qwen-image-mps.git
cd qwen-image-mps
```

2. Install in development mode with dev dependencies:
```bash
pip install -e ".[dev]"
```

3. Install pre-commit hooks:
```bash
pre-commit install
```

The project uses:
- `black` for code formatting
- `isort` for import sorting
- `ruff` for linting
- Pre-commit hooks for code quality

### Running Tests

The project includes integration tests that verify the image generation functionality:

```bash
# Run all tests
pytest tests/

# Run only fast tests (skip integration tests)
pytest -m "not slow"

# Run integration tests (generates real images with minimal steps)
pytest -m slow -v

# Run a specific test with verbose output and print statements
pytest tests/integration/test_generate_function.py::TestGenerateImageIntegration::test_generator_yields_expected_steps -v -s

# Run a specific test class
pytest tests/integration/test_generate_function.py::TestGenerateImageIntegration -v
```

Integration tests generate actual images (by default under `output/`) using ultra-fast mode (4 steps) to minimize execution time while ensuring the pipeline works correctly. Use `-v` for verbose output and `-s` to see print statements during test execution.

## Repository contents
- `src/qwen_image_mps/`: Main package source code
- `qwen-image-mps.py`: Script wrapper for direct URL execution
- `pyproject.toml`: Package configuration and dependencies
- `uv.lock`: Locked dependencies for reproducible builds
- `.github/workflows/`: CI/CD pipelines for testing and publishing
- `example.png`: Sample generated image

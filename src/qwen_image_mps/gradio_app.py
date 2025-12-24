from __future__ import annotations

import glob
import os
import sys
import time
from datetime import datetime
from types import SimpleNamespace
from typing import Generator, List, Sequence

import gradio as gr
from PIL import Image

from .cli import GenerationStep, edit_image, generate_image, layer_image


class LoggingStream:
    """A stream that captures output and appends to a log list."""

    def __init__(self, original_stream, log_list: List[str], prefix: str = ""):
        self.original_stream = original_stream
        self.log_list = log_list
        self.prefix = prefix
        self.buffer = ""

    def write(self, text: str):
        """Write to both original stream and log list."""
        if text:
            self.original_stream.write(text)
            self.original_stream.flush()  # Flush immediately for real-time output
            self.buffer += text
            # Process complete lines
            while "\n" in self.buffer:
                line, self.buffer = self.buffer.split("\n", 1)
                if line.strip():
                    log_line = (
                        f"{self.prefix}{line.strip()}" if self.prefix else line.strip()
                    )
                    if log_line not in self.log_list:
                        self.log_list.append(log_line)

    def flush(self):
        """Flush both streams."""
        self.original_stream.flush()
        # Process any remaining buffer content
        if self.buffer.strip():
            log_line = (
                f"{self.prefix}{self.buffer.strip()}"
                if self.prefix
                else self.buffer.strip()
            )
            if log_line not in self.log_list:
                self.log_list.append(log_line)
            self.buffer = ""

    def __getattr__(self, name):
        """Delegate other attributes to original stream."""
        return getattr(self.original_stream, name)


THEME_TOGGLE_JS = """
function toggleTheme() {
    const url = new URL(window.location);
    const currentTheme = url.searchParams.get("__theme") || "dark";
    const newTheme = currentTheme === "dark" ? "light" : "dark";
    url.searchParams.set("__theme", newTheme);
    window.location.href = url.toString();
}
"""


# Note: Theme switching via URL parameter may not work in Gradio 6.x
# Users can toggle theme using the theme button in the UI
SET_DARK_THEME_JS = """
function ensureDarkTheme() {
    // Check current theme and switch to dark if needed
    const body = document.body;
    if (body && !body.classList.contains('dark')) {
        // Try to find and click the theme toggle button
        const themeBtn = document.querySelector('[data-testid="theme-switch"]') ||
                         document.querySelector('.theme-toggle-btn');
        if (themeBtn) {
            themeBtn.click();
        }
    }
}
// Wait for page to load before applying theme
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', ensureDarkTheme);
} else {
    setTimeout(ensureDarkTheme, 100);
}
"""

SCROLL_GALLERY_TO_TOP_JS = """
(function() {
    function scrollGalleryToTop() {
        const galleryElements = document.querySelectorAll('.generated-images-gallery');
        galleryElements.forEach(gallery => {
            const scrollContainer = gallery.querySelector('.gallery-container') || gallery;
            if (scrollContainer) {
                scrollContainer.scrollTop = 0;
            }
        });
    }
    
    function setupGalleryClickHandlers() {
        const galleries = document.querySelectorAll('.generated-images-gallery');
        galleries.forEach(gallery => {
            const container = gallery.querySelector('.gallery-container, .grid-container') || gallery.querySelector('div');
            if (!container) return;
            
            // Find all image items
            const items = Array.from(container.querySelectorAll('.gallery-item, .grid-item, .thumbnail, [data-testid="gallery-item"]'));
            if (items.length === 0) return;
            
            // Ensure first item is displayed large on top
            const firstItem = items[0];
            if (firstItem) {
                firstItem.style.width = '100%';
                firstItem.style.maxWidth = '100%';
            }
            
            // Setup click handlers for all items
            items.forEach((item, index) => {
                // Remove any existing click handlers
                const newItem = item.cloneNode(true);
                item.parentNode.replaceChild(newItem, item);
                
                // Prevent default link behavior
                const link = newItem.querySelector('a');
                if (link) {
                    link.addEventListener('click', function(e) {
                        e.preventDefault();
                        e.stopPropagation();
                    });
                }
                
                // Add click handler to move clicked image to top
                newItem.addEventListener('click', function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    
                    // Prevent lightbox
                    if (link) {
                        e.preventDefault();
                    }
                    
                    // Move clicked item to first position
                    const allItems = Array.from(container.querySelectorAll('.gallery-item, .grid-item, .thumbnail, [data-testid="gallery-item"]'));
                    const clickedIndex = allItems.indexOf(newItem);
                    
                    if (clickedIndex > 0) {
                        // Remove from current position
                        newItem.remove();
                        // Insert at the beginning
                        container.insertBefore(newItem, container.firstChild);
                        // Re-setup handlers after reordering
                        setTimeout(setupGalleryClickHandlers, 50);
                        // Scroll to top
                        scrollGalleryToTop();
                    }
                });
                
                // Style thumbnails (not first item)
                if (index > 0) {
                    newItem.style.cursor = 'pointer';
                }
            });
        });
    }
    
    // Scroll to top when gallery updates
    document.addEventListener('DOMContentLoaded', function() {
        const observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.addedNodes.length > 0) {
                    scrollGalleryToTop();
                    // Re-setup click handlers when gallery updates
                    setTimeout(setupGalleryClickHandlers, 100);
                }
            });
        });
        
        // Observe gallery containers
        setTimeout(() => {
            const galleries = document.querySelectorAll('.generated-images-gallery');
            galleries.forEach(gallery => {
                observer.observe(gallery, { childList: true, subtree: true });
                setupGalleryClickHandlers();
            });
        }, 1000);
    });
    
    window.scrollGalleryToTop = scrollGalleryToTop;
    window.setupGalleryClickHandlers = setupGalleryClickHandlers;
})();
"""

DEFAULT_GENERATE_STEPS = 50
FAST_MODE_STEPS = 8
ULTRA_FAST_MODE_STEPS = 4

DEFAULT_EDIT_STEPS = 40
EDIT_FAST_STEPS = 8  # Lightning LoRA for non-anime, Rapid-AIO uses 4 for anime
EDIT_ULTRA_FAST_STEPS = 4  # Lightning LoRA for non-anime, Rapid-AIO uses 4 for anime

GENERATE_STEPS_MIN = 4
GENERATE_STEPS_MAX = 60
NUM_IMAGES_MIN = 1
NUM_IMAGES_MAX = 4
EDIT_STEPS_MIN = 4
EDIT_STEPS_MAX = 60

DEFAULT_CFG_SCALE = 4.0
FAST_CFG_SCALE = 1.0
ULTRA_FAST_CFG_SCALE = 1.0


def _sanitize_slider_value(value, default, minimum, maximum):
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, number))


def _sync_slider_and_state(value, default, minimum, maximum):
    sanitized = _sanitize_slider_value(value, default, minimum, maximum)
    return sanitized, sanitized


def _handle_fast_toggle(
    fast_selected: bool,
    ultra_selected: bool,
    default_steps: int,
    fast_steps: int,
    ultra_steps: int,
    min_steps: int,
    max_steps: int,
):
    """Handle Fast checkbox interactions (mutually exclusive with Ultra-fast)."""
    # Ensure all values are valid (not None)
    fast_selected = bool(fast_selected) if fast_selected is not None else False
    ultra_selected = bool(ultra_selected) if ultra_selected is not None else False
    default_steps = _sanitize_slider_value(
        default_steps, default_steps, min_steps, max_steps
    )
    fast_steps = _sanitize_slider_value(fast_steps, fast_steps, min_steps, max_steps)
    ultra_steps = _sanitize_slider_value(ultra_steps, ultra_steps, min_steps, max_steps)

    if fast_selected:
        return True, False, fast_steps, FAST_CFG_SCALE, fast_steps
    if ultra_selected:
        return False, True, ultra_steps, ULTRA_FAST_CFG_SCALE, ultra_steps
    return False, False, default_steps, DEFAULT_CFG_SCALE, default_steps


def _handle_ultra_toggle(
    fast_selected: bool,
    ultra_selected: bool,
    default_steps: int,
    fast_steps: int,
    ultra_steps: int,
    min_steps: int,
    max_steps: int,
):
    """Handle Ultra-fast checkbox interactions (mutually exclusive with Fast)."""
    # Ensure all values are valid (not None)
    fast_selected = bool(fast_selected) if fast_selected is not None else False
    ultra_selected = bool(ultra_selected) if ultra_selected is not None else False
    default_steps = _sanitize_slider_value(
        default_steps, default_steps, min_steps, max_steps
    )
    fast_steps = _sanitize_slider_value(fast_steps, fast_steps, min_steps, max_steps)
    ultra_steps = _sanitize_slider_value(ultra_steps, ultra_steps, min_steps, max_steps)

    if ultra_selected:
        return False, True, ultra_steps, ULTRA_FAST_CFG_SCALE, ultra_steps
    if fast_selected:
        return True, False, fast_steps, FAST_CFG_SCALE, fast_steps
    return False, False, default_steps, DEFAULT_CFG_SCALE, default_steps


DEFAULT_PROMPT = (
    'A coffee shop entrance features a chalkboard sign reading "Apple Silicon '
    'Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "Generated '
    'with MPS on Apple Silicon". Next to it hangs a poster showing a beautiful '
    'woman, and beneath the poster is written "Just try it!".'
)

QUANTIZATION_LEVELS = [
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
]

GENERATION_STEP_DESCRIPTIONS = {
    GenerationStep.INIT: "Initializing generation request…",
    GenerationStep.LOADING_MODEL: "Loading model weights…",
    GenerationStep.MODEL_LOADED: "Model ready.",
    GenerationStep.LOADING_CUSTOM_LORA: "Loading custom LoRA…",
    GenerationStep.LOADING_ULTRA_FAST_LORA: "Loading Lightning LoRA (4-step)…",
    GenerationStep.LOADING_FAST_LORA: "Loading Lightning LoRA (8-step)…",
    GenerationStep.LORA_LOADED: "LoRA merged successfully.",
    GenerationStep.BATMAN_MODE_ACTIVATED: "🦇 Batman mode enabled!",
    GenerationStep.PREPARING_GENERATION: "Preparing inference call…",
    GenerationStep.INFERENCE_START: "Running inference…",
    GenerationStep.INFERENCE_PROGRESS: "Inference progressing…",
    GenerationStep.INFERENCE_COMPLETE: "Inference complete.",
    GenerationStep.SAVING_IMAGE: "Saving image…",
    GenerationStep.IMAGE_SAVED: "Image saved.",
    GenerationStep.COMPLETE: "Generation completed.",
    GenerationStep.ERROR: "Generation failed.",
}


def _normalize_string(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _normalize_optional_int(value) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise gr.Error("Value must be an integer.") from exc


def _normalize_optional_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise gr.Error("Value must be a number.") from exc


def _load_images(paths: Sequence[str]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for path in paths:
        with Image.open(path) as img:
            images.append(img.copy())
    return images


def _coerce_output_dir(path: str | None) -> str:
    normalized = (path or "output").strip() or "output"
    os.makedirs(normalized, exist_ok=True)
    return normalized


def run_generate(  # pragma: no cover - exercised via manual UI usage
    prompt: str,
    negative_prompt: str,
    steps: float,
    fast: bool,
    ultra_fast: bool,
    seed,
    num_images: float,
    aspect: str,
    lora: str,
    cfg_scale,
    batman: bool,
    quantization: str | None,
    output_dir: str,
) -> Generator[tuple, None, None]:
    logs: List[str] = []
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    def _on_event(step: GenerationStep):
        message = GENERATION_STEP_DESCRIPTIONS.get(step, step.value)
        if message not in logs:
            logs.append(message)

    # Ensure steps and num_images are never None
    steps_value = int(steps) if steps is not None else DEFAULT_GENERATE_STEPS
    num_images_value = int(num_images) if num_images is not None else 1

    args = SimpleNamespace(
        prompt=prompt or DEFAULT_PROMPT,
        negative_prompt=_normalize_string(negative_prompt),
        steps=steps_value,
        fast=bool(fast),
        ultra_fast=bool(ultra_fast),
        seed=_normalize_optional_int(seed),
        num_images=num_images_value,
        aspect=aspect,
        lora=_normalize_string(lora),
        cfg_scale=_normalize_optional_float(cfg_scale),
        output_dir=_coerce_output_dir(output_dir),
        batman=bool(batman),
        quantization=_normalize_string(quantization),
        event_callback=_on_event,
        output_path=None,
    )

    saved_paths: List[str] = []
    output_directory = _coerce_output_dir(output_dir)
    generation_start_time = time.time()  # Track when generation started

    try:
        # Replace stdout/stderr with logging streams
        logging_stdout = LoggingStream(original_stdout, logs)
        logging_stderr = LoggingStream(original_stderr, logs, prefix="[stderr] ")
        sys.stdout = logging_stdout
        sys.stderr = logging_stderr

        try:
            # Use Gradio's built-in progress tracking
            progress = gr.Progress()
            last_log_count = 0
            images_generated = 0
            progress_started = False
            current_gallery_images: List[str] = []  # Track images shown in gallery
            last_seen_files = set()  # Track files we've already shown
            last_gallery_state = (
                None  # Track last gallery state to avoid redundant yields
            )

            for output in generate_image(args):
                # Check if log was updated (dedicated event listener for status)
                log_updated = len(logs) > last_log_count
                if log_updated:
                    last_log_count = len(logs)

                if isinstance(output, list):
                    # This is the final output with saved paths
                    saved_paths = output
                    images_generated = len(saved_paths)
                    # Complete progress when images are ready
                    progress(1.0, desc=f"Generated {images_generated} image(s)")
                    # Yield final state
                    log_text = "\n".join(logs) if logs else ""
                    try:
                        images = _load_images(saved_paths)
                        yield images, log_text
                    except Exception:
                        yield [], log_text
                elif isinstance(output, GenerationStep):
                    # Start progress once and keep it running (don't restart)
                    if not progress_started:
                        progress(None, desc="Generating images...")
                        progress_started = True

                    # Check if an image was just saved
                    image_just_saved = False
                    if output == GenerationStep.IMAGE_SAVED:
                        # Look for newly saved images in the output directory
                        # Give a moment for the file to be written
                        time.sleep(0.1)
                        # Find all image files in output directory created after generation started
                        pattern = os.path.join(output_directory, "image-*.png")
                        all_files = glob.glob(pattern)
                        # Filter files created after generation started
                        recent_files = {
                            f
                            for f in all_files
                            if os.path.getmtime(f) >= generation_start_time
                        }
                        new_files = recent_files - last_seen_files

                        if new_files:
                            # Sort by modification time to get the latest
                            new_files_sorted = sorted(new_files, key=os.path.getmtime)
                            for new_file in new_files_sorted:
                                if new_file not in current_gallery_images:
                                    current_gallery_images.append(new_file)
                                    last_seen_files.add(new_file)
                            image_just_saved = True

                    # Always yield log updates when logs change (dedicated event listener)
                    # This ensures status updates happen independently of image updates
                    log_text = "\n".join(logs) if logs else ""

                    # Yield updates: log always updates when changed, gallery only when images change
                    if log_updated:
                        # Log updated - always yield to update status
                        if current_gallery_images:
                            try:
                                images = _load_images(current_gallery_images)
                                images_generated = len(images)
                                if image_just_saved:
                                    progress(
                                        (images_generated / num_images_value),
                                        desc=f"Generated {images_generated}/{num_images_value} image(s)",
                                    )
                                    last_gallery_state = tuple(current_gallery_images)
                                yield images, log_text
                            except Exception:
                                yield [], log_text
                        else:
                            # No images yet, but log updated - yield to update status log
                            yield [], log_text
                    elif image_just_saved:
                        # Image saved but log unchanged - still yield to update gallery
                        try:
                            images = _load_images(current_gallery_images)
                            images_generated = len(images)
                            progress(
                                (images_generated / num_images_value),
                                desc=f"Generated {images_generated}/{num_images_value} image(s)",
                            )
                            last_gallery_state = tuple(current_gallery_images)
                            yield images, log_text
                        except Exception:
                            yield [], log_text
        finally:
            # Restore original streams
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            logging_stdout.flush()
            logging_stderr.flush()
    except Exception as exc:  # pragma: no cover - gradio reports the error
        logs.append(f"Error: {exc}")
        log_text = "\n".join(logs)
        yield [], log_text
        raise gr.Error(str(exc)) from exc

    if not saved_paths:
        raise gr.Error("No images were generated. Check the logs for details.")

    images = _load_images(saved_paths)
    log_text = "\n".join(dict.fromkeys(logs)) or "Generation complete."
    yield images, log_text


def run_edit(  # pragma: no cover - exercised via manual UI usage
    input_images,
    prompt: str,
    negative_prompt: str,
    steps: float,
    fast: bool,
    ultra_fast: bool,
    seed,
    lora: str,
    cfg_scale,
    batman: bool,
    anime: bool,
    quantization: str | None,
    output_dir: str,
):
    if not input_images:
        raise gr.Error("Please upload at least one image to edit.")

    if isinstance(input_images, str):
        input_paths = [input_images]
    else:
        input_paths = [
            item["name"] if isinstance(item, dict) else item
            for item in input_images
            if item
        ]

    if not input_paths:
        raise gr.Error("Could not read uploaded image paths.")

    normalized_prompt = _normalize_string(prompt)
    if not anime and normalized_prompt is None:
        raise gr.Error("Prompt is required unless Anime mode is enabled.")

    output_directory = _coerce_output_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_filename = os.path.join(output_directory, f"gradio-edit-{timestamp}.png")

    # Ensure steps is never None
    steps_value = int(steps) if steps is not None else DEFAULT_EDIT_STEPS

    args = SimpleNamespace(
        input=input_paths,
        prompt=normalized_prompt,
        negative_prompt=_normalize_string(negative_prompt),
        steps=steps_value,
        fast=bool(fast),
        ultra_fast=bool(ultra_fast),
        seed=_normalize_optional_int(seed),
        output=output_filename,
        cfg_scale=_normalize_optional_float(cfg_scale),
        output_dir=output_directory,
        lora=_normalize_string(lora),
        batman=bool(batman),
        anime=bool(anime),
        quantization=_normalize_string(quantization),
    )

    logs: List[str] = []
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    saved_path = None

    try:
        # Replace stdout/stderr with logging streams
        logging_stdout = LoggingStream(original_stdout, logs)
        logging_stderr = LoggingStream(original_stderr, logs, prefix="[stderr] ")
        sys.stdout = logging_stdout
        sys.stderr = logging_stderr

        try:
            saved_path = edit_image(args)

            # Yield log updates
            if logs:
                log_text = "\n".join(logs)
                yield None, log_text
        finally:
            # Restore original streams
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            logging_stdout.flush()
            logging_stderr.flush()
    except Exception as exc:  # pragma: no cover
        logs.append(f"Error: {exc}")
        log_text = "\n".join(logs) if logs else str(exc)
        yield None, log_text
        raise gr.Error(str(exc)) from exc

    if not saved_path:
        raise gr.Error("Image editing failed.")

    edited_image = _load_images([saved_path])[0]
    final_log = "\n".join(logs) if logs else f"Edited image saved to: {saved_path}"
    yield edited_image, final_log


def run_layer(
    input_image,
    prompt: str,
    negative_prompt: str,
    steps: float,
    seed,
    num_layers: float,
    cfg_scale,
    output_dir: str,
):
    """Decompose an image into multiple transparent layers."""
    if not input_image:
        raise gr.Error("Please upload an image to decompose into layers.")

    # Get the file path from the input
    if isinstance(input_image, dict):
        input_path = input_image.get("name")
    else:
        input_path = input_image

    if not input_path:
        raise gr.Error("Could not read uploaded image path.")

    # Ensure steps is never None
    steps_value = int(steps) if steps is not None else 50
    num_layers_value = int(num_layers) if num_layers is not None else 4

    args = SimpleNamespace(
        input=input_path,
        prompt=_normalize_string(prompt),
        negative_prompt=_normalize_string(negative_prompt) or " ",
        steps=steps_value,
        seed=_normalize_optional_int(seed),
        layers=num_layers_value,
        cfg_scale=_normalize_optional_float(cfg_scale) or 4.0,
        cfg_norm=True,
        use_en_prompt=True,
        output_dir=_coerce_output_dir(output_dir),
        zip=False,
    )

    logs: List[str] = []
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    saved_paths = []

    try:
        # Replace stdout/stderr with logging streams
        logging_stdout = LoggingStream(original_stdout, logs)
        logging_stderr = LoggingStream(original_stderr, logs, prefix="[stderr] ")
        sys.stdout = logging_stdout
        sys.stderr = logging_stderr

        try:
            saved_paths = layer_image(args)

            # Yield log updates
            if logs:
                log_text = "\n".join(logs)
                yield [], log_text
        finally:
            # Restore original streams
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            logging_stdout.flush()
            logging_stderr.flush()
    except Exception as exc:  # pragma: no cover
        logs.append(f"Error: {exc}")
        log_text = "\n".join(logs) if logs else str(exc)
        yield [], log_text
        raise gr.Error(str(exc)) from exc

    if not saved_paths:
        raise gr.Error("Layer decomposition failed.")

    # Load and yield the layer images
    try:
        layer_images = _load_images(saved_paths)
        final_log = "\n".join(logs) if logs else f"Layers saved to: {saved_paths[0]}"
        yield layer_images, final_log
    except Exception as e:
        yield [], f"Error loading layers: {e}"


# Custom CSS for the Gradio interface
CUSTOM_CSS = """
.theme-toggle-btn {
    min-width: 40px !important;
    width: 40px !important;
    padding: 8px !important;
    position: relative !important;
}
.theme-toggle-btn::before {
    content: "Toggle light / dark theme";
    position: absolute;
    bottom: calc(100% + 8px);
    left: 50%;
    transform: translateX(-50%);
    background: rgba(0, 0, 0, 0.85);
    color: white;
    padding: 6px 10px;
    border-radius: 4px;
    font-size: 12px;
    white-space: nowrap;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.2s;
    z-index: 1000;
}
.theme-toggle-btn:hover::before {
    opacity: 1;
}
/* Gallery layout: selected image on top, thumbnails below */
.generated-images-gallery {
    display: flex !important;
    flex-direction: column !important;
}
.generated-images-gallery .grid-container,
.generated-images-gallery .gallery-container,
.generated-images-gallery > div {
    display: flex !important;
    flex-direction: column !important;
    width: 100% !important;
    gap: 12px !important;
}
/* Selected/main image display on top - full width */
.generated-images-gallery .grid-item:first-child,
.generated-images-gallery .gallery-item:first-child,
.generated-images-gallery .thumbnail:first-child {
    width: 100% !important;
    max-width: 100% !important;
    order: 1 !important;
}
.generated-images-gallery .grid-item:first-child img,
.generated-images-gallery .gallery-item:first-child img,
.generated-images-gallery .thumbnail:first-child img {
    width: 100% !important;
    max-width: 100% !important;
    height: auto !important;
    object-fit: contain !important;
    display: block !important;
    max-height: 500px !important;
}
/* Thumbnails row below selected image */
.generated-images-gallery .grid-item:not(:first-child),
.generated-images-gallery .gallery-item:not(:first-child),
.generated-images-gallery .thumbnail:not(:first-child) {
    order: 2 !important;
    width: calc(25% - 9px) !important;
    max-width: calc(25% - 9px) !important;
    cursor: pointer !important;
    border: 2px solid transparent !important;
    border-radius: 4px !important;
    transition: border-color 0.2s !important;
    padding: 4px !important;
    flex-shrink: 0 !important;
}
/* Thumbnail row wrapper */
.generated-images-gallery .thumbnail-row {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
    width: 100% !important;
    margin-top: 8px !important;
}
.generated-images-gallery .grid-item:not(:first-child) img,
.generated-images-gallery .gallery-item:not(:first-child) img,
.generated-images-gallery .thumbnail:not(:first-child) img {
    width: 100% !important;
    height: auto !important;
    object-fit: cover !important;
    border-radius: 2px !important;
    aspect-ratio: 1 !important;
}
.generated-images-gallery .grid-item:not(:first-child):hover,
.generated-images-gallery .gallery-item:not(:first-child):hover,
.generated-images-gallery .thumbnail:not(:first-child):hover {
    border-color: #4CAF50 !important;
}
/* Disable lightbox/fullscreen */
.generated-images-gallery .lightbox,
.generated-images-gallery .gallery-lightbox,
.generated-images-gallery .lightbox-overlay {
    display: none !important;
}
.generated-images-gallery a {
    pointer-events: none !important;
    cursor: default !important;
}
.generated-images-gallery .gallery-item a,
.generated-images-gallery .grid-item a {
    pointer-events: none !important;
}
.generated-images-gallery img {
    pointer-events: auto !important;
    cursor: pointer !important;
}
.generated-images-gallery .gallery-item:first-child img {
    cursor: default !important;
}
.generate-button-full-width {
    width: 100% !important;
    max-width: 100% !important;
}
"""


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Qwen-Image Studio") as demo:
        gr.HTML(f"<script>{SCROLL_GALLERY_TO_TOP_JS}</script>", visible=False)
        with gr.Row():
            with gr.Column(scale=10):
                gr.Markdown(
                    "## Qwen-Image Studio\n"
                    "Generate and edit images using Qwen models with Apple Silicon (MPS), "
                    "CUDA, or CPU backends. Generate text-to-image or edit images with prompts. "
                    "Fast/Ultra-fast modes use Lightning LoRA for regular edits or Rapid-AIO for anime; "
                    "Batman mode adds LEGO Batman photobombs.",
                )
            with gr.Column(scale=0, min_width=50):
                theme_toggle = gr.Button(
                    "🌙",
                    variant="secondary",
                    size="sm",
                    elem_classes=["theme-toggle-btn"],
                )
                theme_toggle.click(
                    fn=None,
                    inputs=None,
                    outputs=None,
                    js=THEME_TOGGLE_JS,
                )

        with gr.Tab("Generate"):
            with gr.Row():
                with gr.Column(scale=1):
                    prompt = gr.Textbox(
                        label="Prompt",
                        value=DEFAULT_PROMPT,
                        lines=4,
                    )
                    negative_prompt = gr.Textbox(
                        label="Negative prompt",
                        placeholder="blurry, watermark, text, low quality",
                        lines=2,
                    )
                    with gr.Row():
                        steps = gr.Slider(
                            label="Steps",
                            minimum=4,
                            maximum=60,
                            step=1,
                            value=DEFAULT_GENERATE_STEPS,
                        )
                        num_images = gr.Slider(
                            label="Number of images",
                            minimum=1,
                            maximum=5,
                            step=1,
                            value=1,
                        )
                        aspect = gr.Dropdown(
                            label="Aspect ratio",
                            choices=["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"],
                            value="16:9",
                        )
                    with gr.Row():
                        fast = gr.Checkbox(label="Fast (Lightning 8-step)")
                        ultra_fast = gr.Checkbox(label="Ultra-fast (Lightning 4-step)")
                        batman = gr.Checkbox(label="Batman photobomb 🦇")
                    seed = gr.Number(label="Seed (optional)", precision=0)
                    cfg_scale = gr.Number(
                        label="CFG scale override (optional)",
                        precision=1,
                        value=DEFAULT_CFG_SCALE,
                    )
                    lora = gr.Textbox(
                        label="Custom LoRA (path or HF repo)",
                        placeholder="e.g., autoweeb/Qwen-Image-Edit-2509-Photo-to-Anime, flymy-ai/qwen-image-anime-irl-lora",
                    )
                    quantization = gr.Dropdown(
                        label="Quantization (GGUF)",
                        choices=[None] + QUANTIZATION_LEVELS,
                        value=None,
                    )
                    output_dir = gr.Textbox(
                        label="Output directory",
                        value="output",
                    )
                    steps_state = gr.State(DEFAULT_GENERATE_STEPS)
                    num_images_state = gr.State(1)
                with gr.Column(scale=1):
                    generate_button = gr.Button(
                        "Generate",
                        variant="primary",
                        elem_classes=["generate-button-full-width"],
                    )
                    gallery = gr.Gallery(
                        label="Generated images",
                        columns=1,
                        rows=4,
                        height=600,
                        preview=True,
                        show_label=True,
                        elem_classes=["generated-images-gallery"],
                    )
                    log = gr.Textbox(label="Event log", lines=12)

            steps.change(
                fn=lambda value: _sync_slider_and_state(
                    value,
                    DEFAULT_GENERATE_STEPS,
                    GENERATE_STEPS_MIN,
                    GENERATE_STEPS_MAX,
                ),
                inputs=steps,
                outputs=[steps, steps_state],
            )

            num_images.change(
                fn=lambda value: _sync_slider_and_state(
                    value, NUM_IMAGES_MIN, NUM_IMAGES_MIN, NUM_IMAGES_MAX
                ),
                inputs=num_images,
                outputs=[num_images, num_images_state],
            )

            def _safe_fast_toggle(fast_val, ultra_val):
                """Wrapper to ensure values are never None."""
                try:
                    return _handle_fast_toggle(
                        fast_val,
                        ultra_val,
                        DEFAULT_GENERATE_STEPS,
                        FAST_MODE_STEPS,
                        ULTRA_FAST_MODE_STEPS,
                        GENERATE_STEPS_MIN,
                        GENERATE_STEPS_MAX,
                    )
                except (TypeError, ValueError):
                    # Fallback to defaults if any value is invalid
                    return (
                        False,
                        False,
                        DEFAULT_GENERATE_STEPS,
                        DEFAULT_CFG_SCALE,
                        DEFAULT_GENERATE_STEPS,
                    )

            def _safe_ultra_toggle(fast_val, ultra_val):
                """Wrapper to ensure values are never None."""
                try:
                    return _handle_ultra_toggle(
                        fast_val,
                        ultra_val,
                        DEFAULT_GENERATE_STEPS,
                        FAST_MODE_STEPS,
                        ULTRA_FAST_MODE_STEPS,
                        GENERATE_STEPS_MIN,
                        GENERATE_STEPS_MAX,
                    )
                except (TypeError, ValueError):
                    # Fallback to defaults if any value is invalid
                    return (
                        False,
                        False,
                        DEFAULT_GENERATE_STEPS,
                        DEFAULT_CFG_SCALE,
                        DEFAULT_GENERATE_STEPS,
                    )

            fast.change(
                fn=_safe_fast_toggle,
                inputs=[fast, ultra_fast],
                outputs=[fast, ultra_fast, steps, cfg_scale, steps_state],
            )

            ultra_fast.change(
                fn=_safe_ultra_toggle,
                inputs=[fast, ultra_fast],
                outputs=[fast, ultra_fast, steps, cfg_scale, steps_state],
            )

            generate_button.click(
                fn=run_generate,
                inputs=[
                    prompt,
                    negative_prompt,
                    steps_state,
                    fast,
                    ultra_fast,
                    seed,
                    num_images_state,
                    aspect,
                    lora,
                    cfg_scale,
                    batman,
                    quantization,
                    output_dir,
                ],
                outputs=[gallery, log],
            )

        with gr.Tab("Edit"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_images = gr.File(
                        label="Upload image(s)",
                        file_types=["image"],
                        file_count="multiple",
                    )
                    prompt_edit = gr.Textbox(
                        label="Edit prompt (leave empty for anime mode)",
                        placeholder="Add snow to the mountains",
                        lines=4,
                    )
                    negative_prompt_edit = gr.Textbox(
                        label="Negative prompt",
                        placeholder="blurry, watermark, text, low quality",
                        lines=2,
                    )
                    with gr.Row():
                        steps_edit = gr.Slider(
                            label="Steps",
                            minimum=4,
                            maximum=60,
                            step=1,
                            value=DEFAULT_EDIT_STEPS,
                        )
                        fast_edit = gr.Checkbox(
                            label="Fast (Lightning 8-step / Rapid-AIO for anime)"
                        )
                        ultra_fast_edit = gr.Checkbox(
                            label="Ultra-fast (Lightning 4-step / Rapid-AIO for anime)"
                        )
                    with gr.Row():
                        batman_edit = gr.Checkbox(label="Batman photobomb 🦇")
                        anime_edit = gr.Checkbox(label="Anime mode (photo ➜ anime)")
                    seed_edit = gr.Number(label="Seed (optional)", precision=0)
                    cfg_scale_edit = gr.Number(
                        label="CFG scale override (optional)",
                        precision=1,
                        value=DEFAULT_CFG_SCALE,
                    )
                    lora_edit = gr.Textbox(
                        label="Custom LoRA (path or HF repo)",
                        placeholder="e.g., autoweeb/Qwen-Image-Edit-2509-Photo-to-Anime, flymy-ai/qwen-image-anime-irl-lora",
                    )
                    quantization_edit = gr.Dropdown(
                        label="Quantization (GGUF)",
                        choices=[None] + QUANTIZATION_LEVELS,
                        value=None,
                    )
                    output_dir_edit = gr.Textbox(
                        label="Output directory",
                        value="output",
                    )
                    steps_edit_state = gr.State(DEFAULT_EDIT_STEPS)
                    uploaded_images_state = gr.State([])
                with gr.Column(scale=1):
                    edit_button = gr.Button(
                        "Edit Image",
                        variant="primary",
                        elem_classes=["generate-button-full-width"],
                    )
                    uploaded_gallery = gr.Gallery(
                        label="Uploaded images (click to remove)",
                        columns=2,
                        rows=2,
                        height=300,
                        preview=True,
                        show_label=True,
                        interactive=True,
                    )
                    edited_preview = gr.Image(
                        label="Edited image preview", interactive=False
                    )
                    edit_log = gr.Textbox(label="Status", lines=6)

            def _handle_file_upload(files, current_state):
                """Handle file uploads and update gallery."""
                # Start with current state
                existing_files = list(current_state) if current_state else []

                if not files:
                    # If no new files, return current state
                    if existing_files:
                        try:
                            images = _load_images(existing_files)
                            return images, existing_files
                        except Exception:
                            return [], existing_files
                    return [], []

                # Convert new files to list format
                new_file_list = []
                if isinstance(files, str):
                    new_file_list = [files]
                else:
                    new_file_list = [
                        item["name"] if isinstance(item, dict) else item
                        for item in files
                        if item
                    ]

                # Combine with existing files (avoid duplicates)
                combined_files = existing_files.copy()
                for new_file in new_file_list:
                    if new_file not in combined_files:
                        combined_files.append(new_file)

                # Load images for gallery display
                if combined_files:
                    try:
                        images = _load_images(combined_files)
                        return images, combined_files
                    except Exception:
                        return [], combined_files
                return [], []

            def _handle_gallery_select(evt: gr.SelectData, current_state):
                """Handle image removal when clicked in gallery."""
                if not current_state or evt.index >= len(current_state):
                    # Return current state if invalid index
                    if current_state:
                        try:
                            images = _load_images(current_state)
                            return images, current_state
                        except Exception:
                            return [], current_state
                    return [], []

                # Remove the selected image
                new_state = [
                    img for i, img in enumerate(current_state) if i != evt.index
                ]

                # Reload images for gallery
                if new_state:
                    try:
                        new_images = _load_images(new_state)
                        return new_images, new_state
                    except Exception:
                        return [], new_state
                else:
                    return [], []

            steps_edit.change(
                fn=lambda value: _sync_slider_and_state(
                    value, DEFAULT_EDIT_STEPS, EDIT_STEPS_MIN, EDIT_STEPS_MAX
                ),
                inputs=steps_edit,
                outputs=[steps_edit, steps_edit_state],
            )

            input_images.upload(
                fn=_handle_file_upload,
                inputs=[input_images, uploaded_images_state],
                outputs=[uploaded_gallery, uploaded_images_state],
            )

            uploaded_gallery.select(
                fn=_handle_gallery_select,
                inputs=[uploaded_images_state],
                outputs=[uploaded_gallery, uploaded_images_state],
            )

            def _safe_fast_edit_toggle(fast_val, ultra_val):
                """Wrapper to ensure values are never None."""
                try:
                    return _handle_fast_toggle(
                        fast_val,
                        ultra_val,
                        DEFAULT_EDIT_STEPS,
                        EDIT_FAST_STEPS,
                        EDIT_ULTRA_FAST_STEPS,
                        EDIT_STEPS_MIN,
                        EDIT_STEPS_MAX,
                    )
                except (TypeError, ValueError):
                    # Fallback to defaults if any value is invalid
                    return (
                        False,
                        False,
                        DEFAULT_EDIT_STEPS,
                        DEFAULT_CFG_SCALE,
                        DEFAULT_EDIT_STEPS,
                    )

            def _safe_ultra_edit_toggle(fast_val, ultra_val):
                """Wrapper to ensure values are never None."""
                try:
                    return _handle_ultra_toggle(
                        fast_val,
                        ultra_val,
                        DEFAULT_EDIT_STEPS,
                        EDIT_FAST_STEPS,
                        EDIT_ULTRA_FAST_STEPS,
                        EDIT_STEPS_MIN,
                        EDIT_STEPS_MAX,
                    )
                except (TypeError, ValueError):
                    # Fallback to defaults if any value is invalid
                    return (
                        False,
                        False,
                        DEFAULT_EDIT_STEPS,
                        DEFAULT_CFG_SCALE,
                        DEFAULT_EDIT_STEPS,
                    )

            fast_edit.change(
                fn=_safe_fast_edit_toggle,
                inputs=[fast_edit, ultra_fast_edit],
                outputs=[
                    fast_edit,
                    ultra_fast_edit,
                    steps_edit,
                    cfg_scale_edit,
                    steps_edit_state,
                ],
            )

            ultra_fast_edit.change(
                fn=_safe_ultra_edit_toggle,
                inputs=[fast_edit, ultra_fast_edit],
                outputs=[
                    fast_edit,
                    ultra_fast_edit,
                    steps_edit,
                    cfg_scale_edit,
                    steps_edit_state,
                ],
            )

            edit_button.click(
                fn=run_edit,
                inputs=[
                    uploaded_images_state,
                    prompt_edit,
                    negative_prompt_edit,
                    steps_edit_state,
                    fast_edit,
                    ultra_fast_edit,
                    seed_edit,
                    lora_edit,
                    cfg_scale_edit,
                    batman_edit,
                    anime_edit,
                    quantization_edit,
                    output_dir_edit,
                ],
                outputs=[edited_preview, edit_log],
            )

        with gr.Tab("Layer"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_layer_image = gr.File(
                        label="Upload image",
                        file_types=["image"],
                        file_count="single",
                    )
                    layer_preview = gr.Image(
                        label="Uploaded image preview",
                        interactive=False,
                    )
                    prompt_layer = gr.Textbox(
                        label="Prompt (optional)",
                        placeholder="Describe the image content (auto-generated if empty)",
                        lines=2,
                    )
                    negative_prompt_layer = gr.Textbox(
                        label="Negative prompt",
                        placeholder="Elements to exclude",
                        lines=2,
                    )
                    with gr.Row():
                        steps_layer = gr.Slider(
                            label="Steps",
                            minimum=4,
                            maximum=60,
                            step=1,
                            value=50,
                        )
                        num_layers = gr.Slider(
                            label="Number of layers",
                            minimum=2,
                            maximum=10,
                            step=1,
                            value=4,
                        )
                    seed_layer = gr.Number(label="Seed (optional)", precision=0)
                    cfg_scale_layer = gr.Number(
                        label="CFG scale",
                        precision=1,
                        value=4.0,
                    )
                    output_dir_layer = gr.Textbox(
                        label="Output directory",
                        value="output",
                    )
                    steps_layer_state = gr.State(50)
                with gr.Column(scale=1):
                    layer_button = gr.Button(
                        "Decompose into Layers",
                        variant="primary",
                        elem_classes=["generate-button-full-width"],
                    )
                    layer_gallery = gr.Gallery(
                        label="Generated layers",
                        columns=2,
                        rows=2,
                        height=500,
                        preview=True,
                        show_label=True,
                        elem_classes=["generated-images-gallery"],
                    )
                    layer_log = gr.Textbox(label="Status", lines=8)

            def _handle_layer_file_upload(file):
                """Handle file upload for layer tab."""
                if not file:
                    return None, []

                if isinstance(file, dict):
                    file_path = file.get("name")
                else:
                    file_path = file

                if not file_path:
                    return None, []

                try:
                    images = _load_images([file_path])
                    return images[0], [file_path]
                except Exception:
                    return None, []

            def _sync_layer_steps(value):
                """Sync steps value."""
                try:
                    return int(value) if value is not None else 50
                except (TypeError, ValueError):
                    return 50

            input_layer_image.upload(
                fn=_handle_layer_file_upload,
                inputs=[input_layer_image],
                outputs=[layer_preview, gr.State([])],
            )

            steps_layer.change(
                fn=_sync_layer_steps,
                inputs=[steps_layer],
                outputs=[steps_layer_state],
            )

            layer_button.click(
                fn=run_layer,
                inputs=[
                    input_layer_image,
                    prompt_layer,
                    negative_prompt_layer,
                    steps_layer_state,
                    seed_layer,
                    num_layers,
                    cfg_scale_layer,
                    output_dir_layer,
                ],
                outputs=[layer_gallery, layer_log],
            )

        # Note: In Gradio 6.x, automatic theme setting on load may not work
        # Users can use the theme toggle button (🌙) in the UI
        # demo.load(fn=lambda: None, inputs=None, outputs=None, js=SET_DARK_THEME_JS)

    return demo


def launch(
    server_name: str | None = None,
    server_port: int | None = None,
    share: bool = False,
):
    demo = build_interface()
    demo = demo.queue(max_size=8)
    # In Gradio 6.x, CSS must be passed to launch(), not Blocks()
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        css=CUSTOM_CSS,
    )


def main():
    launch()


__all__ = ["build_interface", "launch", "main"]

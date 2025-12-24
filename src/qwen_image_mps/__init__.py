"""Qwen-Image MPS - Generate images with Qwen-Image on Apple Silicon and other devices."""

__version__ = "0.7.0"

from .cli import (  # noqa: E402
    build_edit_parser,
    build_generate_parser,
    build_layer_parser,
    edit_image,
    generate_image,
    get_lora_path,
    layer_image,
    main,
    merge_lora_from_safetensors,
)


def launch_gradio(
    *,
    server_name: str | None = None,
    server_port: int | None = None,
    share: bool = False,
):
    """Launch the Gradio interface.

    This local import avoids importing Gradio unless the UI is requested.
    """

    from .gradio_app import launch as _launch  # pylint: disable=import-outside-toplevel

    return _launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
    )


__all__ = [
    "main",
    "build_generate_parser",
    "build_edit_parser",
    "build_layer_parser",
    "generate_image",
    "edit_image",
    "layer_image",
    "get_lora_path",
    "merge_lora_from_safetensors",
    "launch_gradio",
]

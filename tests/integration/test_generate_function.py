"""
Integration tests for image generation and editing functions.
Tests the actual functions without mocking to ensure everything works end-to-end.
Covers all modes: normal, fast, and ultra-fast for generation, editing, and anime transformation.
"""

import os
import sys
from argparse import Namespace
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qwen_image_mps.cli import GenerationStep, edit_image, generate_image


@pytest.mark.slow
class TestGenerateImageIntegration:
    """Integration tests for the refactored generate_image function."""

    def test_cli_backward_compatibility_simple_generation(self):
        """Test: CLI can still generate images with basic arguments (small test)."""
        args = Namespace(
            prompt="Sunset in New York City",
            steps=1,  # Minimal steps for speed
            seed=123,
            num_images=1,
            lora=None,
            batman=True,
            ultra_fast=True,
            fast=False,
            output_path="image-test-compatibility.png",
            aspect="16:9",
        )

        # Save to project root instead of temp directory
        original_dir = os.getcwd()
        try:
            # Act: Consume the generator
            results = list(generate_image(args))

            # Assert: Check that function completes and creates files
            assert len(results) > 0, "Generator should yield results"

            # Check for generated image files
            png_files = list(Path(".").glob("*.png"))
            assert len(png_files) >= 1, "Should generate at least one PNG file"

            # Verify file is not empty
            for png_file in png_files:
                assert (
                    png_file.stat().st_size > 0
                ), f"Generated file {png_file} should not be empty"

        finally:
            # Restore original directory
            os.chdir(original_dir)

    def test_generator_yields_expected_steps(self):
        """Test: Generator yields all expected GenerationStep events."""
        args = Namespace(
            prompt="Sunset in New York City",
            steps=1,  # Minimal steps for speed
            seed=123,
            num_images=1,
            lora=None,
            batman=True,
            ultra_fast=True,
            fast=False,
            output_path="image-test-output.png",
            aspect="16:9",
        )

        # Save to project root instead of temp directory
        original_dir = os.getcwd()
        try:
            # Act: Collect yielded steps
            yielded_steps = []
            final_result = None

            for result in generate_image(args):
                if isinstance(result, GenerationStep):
                    yielded_steps.append(result)
                    print(f"DEBUG: Yielded step: {result}")
                else:
                    final_result = result
                    print(f"DEBUG: Final result type: {type(result)} - {result}")

            # Assert: Verify expected steps are yielded
            # Note: The function may yield additional steps based on internal logic
            required_steps = [
                GenerationStep.INIT,
                GenerationStep.LOADING_MODEL,
                GenerationStep.MODEL_LOADED,
                GenerationStep.PREPARING_GENERATION,
                GenerationStep.INFERENCE_START,
                GenerationStep.INFERENCE_COMPLETE,
                GenerationStep.SAVING_IMAGE,
                GenerationStep.IMAGE_SAVED,
                GenerationStep.COMPLETE,
            ]

            # Verify required steps are all present
            for step in required_steps:
                assert (
                    step in yielded_steps
                ), f"Required step {step} not found in yielded steps"

            # Verify the steps are in the right general order (INIT first, COMPLETE last)
            assert yielded_steps[0] == GenerationStep.INIT, "First step should be INIT"
            assert (
                yielded_steps[-1] == GenerationStep.COMPLETE
            ), "Last step should be COMPLETE"

            # Verify we have a reasonable number of steps (at least the required ones)
            assert len(yielded_steps) >= len(
                required_steps
            ), "Should have at least the required number of steps"
            assert final_result is not None, "Should yield final result"
            assert isinstance(
                final_result, list
            ), "Final result should be a list of paths"

        finally:
            # Restore original directory (though not needed for this test)
            os.chdir(original_dir)

    def test_multiple_images_generation(self):
        """Test: Can generate multiple images."""
        args = Namespace(
            prompt="Sunset in New York City",
            steps=1,  # Minimal steps for speed
            seed=123,
            num_images=2,  # Generate 2 images
            lora=None,
            batman=True,
            ultra_fast=True,
            fast=False,
            aspect="16:9",
            # Note: No output_path for multiple images - will use auto-generated names
        )

        # Save to project root instead of temp directory
        original_dir = os.getcwd()
        try:
            # Act: Get final result
            final_result = None
            for result in generate_image(args):
                if not isinstance(result, GenerationStep):
                    final_result = result

            # Assert: Multiple files created
            assert final_result is not None, "Should yield final result"
            assert isinstance(final_result, list), "Final result should be a list"
            assert len(final_result) == 2, "Should have paths for 2 generated images"

            # Check actual files exist under default output directory
            png_files = list(Path("output").glob("*.png"))
            assert len(png_files) >= 2, "Should create at least 2 PNG files in output/"

            # Verify paths are absolute and files exist
            for path in final_result:
                assert os.path.isabs(path), f"Path {path} should be absolute"
                assert os.path.exists(path), f"File {path} should exist"

        finally:
            # Restore original directory
            os.chdir(original_dir)

    def test_generate_normal_mode(self):
        """Test: Generate image in normal mode (no fast/ultra-fast flags)."""
        args = Namespace(
            prompt="A serene mountain landscape",
            steps=2,  # Minimal steps for speed
            seed=456,
            num_images=1,
            lora=None,
            batman=False,
            ultra_fast=False,
            fast=False,
            aspect="16:9",
        )

        original_dir = os.getcwd()
        try:
            final_result = None
            for result in generate_image(args):
                if not isinstance(result, GenerationStep):
                    final_result = result

            assert final_result is not None, "Should yield final result"
            assert isinstance(final_result, list), "Final result should be a list"
            assert len(final_result) == 1, "Should have path for 1 generated image"
            assert os.path.exists(
                final_result[0]
            ), f"File {final_result[0]} should exist"
            assert (
                os.path.getsize(final_result[0]) > 0
            ), "Generated file should not be empty"

        finally:
            os.chdir(original_dir)

    def test_generate_fast_mode(self):
        """Test: Generate image in fast mode (8 steps with Lightning LoRA)."""
        args = Namespace(
            prompt="A futuristic cityscape",
            steps=50,  # Will be overridden by fast mode
            seed=789,
            num_images=1,
            lora=None,
            batman=False,
            ultra_fast=False,
            fast=True,  # Fast mode enabled
            aspect="16:9",
        )

        original_dir = os.getcwd()
        try:
            final_result = None
            for result in generate_image(args):
                if not isinstance(result, GenerationStep):
                    final_result = result

            assert final_result is not None, "Should yield final result"
            assert isinstance(final_result, list), "Final result should be a list"
            assert len(final_result) == 1, "Should have path for 1 generated image"
            assert os.path.exists(
                final_result[0]
            ), f"File {final_result[0]} should exist"
            assert (
                os.path.getsize(final_result[0]) > 0
            ), "Generated file should not be empty"

        finally:
            os.chdir(original_dir)

    def test_generate_ultra_fast_mode(self):
        """Test: Generate image in ultra-fast mode (4 steps with Lightning LoRA)."""
        args = Namespace(
            prompt="A peaceful forest scene",
            steps=50,  # Will be overridden by ultra-fast mode
            seed=321,
            num_images=1,
            lora=None,
            batman=False,
            ultra_fast=True,  # Ultra-fast mode enabled
            fast=False,
            aspect="16:9",
        )

        original_dir = os.getcwd()
        try:
            final_result = None
            for result in generate_image(args):
                if not isinstance(result, GenerationStep):
                    final_result = result

            assert final_result is not None, "Should yield final result"
            assert isinstance(final_result, list), "Final result should be a list"
            assert len(final_result) == 1, "Should have path for 1 generated image"
            assert os.path.exists(
                final_result[0]
            ), f"File {final_result[0]} should exist"
            assert (
                os.path.getsize(final_result[0]) > 0
            ), "Generated file should not be empty"

        finally:
            os.chdir(original_dir)


@pytest.mark.slow
class TestEditImageIntegration:
    """Integration tests for the edit_image function."""

    @pytest.fixture
    def test_image_path(self):
        """Provide path to test image for editing."""
        # Use example.png from project root
        image_path = os.path.join(os.path.dirname(__file__), "..", "..", "example.png")
        abs_path = os.path.abspath(image_path)
        if not os.path.exists(abs_path):
            pytest.skip(f"Test image not found at {abs_path}")
        return abs_path

    def test_edit_normal_mode(self, test_image_path):
        """Test: Edit image in normal mode (with custom steps)."""
        args = Namespace(
            input=[test_image_path],
            prompt="Add dramatic lighting",
            steps=3,  # Custom steps (minimal for speed)
            seed=111,
            fast=False,
            ultra_fast=False,
            output=None,
            output_dir="output",
            lora=None,
            batman=False,
            anime=False,
            negative_prompt=None,
            cfg_scale=None,
            quantization=None,
        )

        original_dir = os.getcwd()
        try:
            edit_image(args)

            # Check that edited image was created
            edited_files = list(Path("output").glob("edited-*.png"))
            assert len(edited_files) >= 1, "Should create at least one edited image"
            assert edited_files[0].stat().st_size > 0, "Edited file should not be empty"

        finally:
            os.chdir(original_dir)

    def test_edit_fast_mode(self, test_image_path):
        """Test: Edit image in fast mode (4 steps with Rapid-AIO)."""
        args = Namespace(
            input=[test_image_path],
            prompt="Add vibrant colors",
            steps=40,  # Default, will be overridden by fast mode
            seed=222,
            fast=True,  # Fast mode enabled
            ultra_fast=False,
            output=None,
            output_dir="output",
            lora=None,
            batman=False,
            anime=False,
            negative_prompt=None,
            cfg_scale=None,
            quantization=None,
        )

        original_dir = os.getcwd()
        try:
            edit_image(args)

            # Check that edited image was created
            edited_files = list(Path("output").glob("edited-*.png"))
            assert len(edited_files) >= 1, "Should create at least one edited image"
            assert edited_files[0].stat().st_size > 0, "Edited file should not be empty"

        finally:
            os.chdir(original_dir)

    def test_edit_ultra_fast_mode(self, test_image_path):
        """Test: Edit image in ultra-fast mode (same as fast mode for editing)."""
        args = Namespace(
            input=[test_image_path],
            prompt="Add sunset colors",
            steps=40,  # Default, will be overridden by ultra-fast mode
            seed=333,
            fast=False,
            ultra_fast=True,  # Ultra-fast mode enabled
            output=None,
            output_dir="output",
            lora=None,
            batman=False,
            anime=False,
            negative_prompt=None,
            cfg_scale=None,
            quantization=None,
        )

        original_dir = os.getcwd()
        try:
            edit_image(args)

            # Check that edited image was created
            edited_files = list(Path("output").glob("edited-*.png"))
            assert len(edited_files) >= 1, "Should create at least one edited image"
            assert edited_files[0].stat().st_size > 0, "Edited file should not be empty"

        finally:
            os.chdir(original_dir)

    def test_anime_normal_mode(self, test_image_path):
        """Test: Anime transformation in normal mode (with custom steps)."""
        args = Namespace(
            input=[test_image_path],
            prompt="Make it colorful",  # Optional with anime
            steps=3,  # Custom steps (minimal for speed)
            seed=444,
            fast=False,
            ultra_fast=False,
            output=None,
            output_dir="output",
            lora=None,
            batman=False,
            anime=True,  # Anime mode enabled
            negative_prompt=None,
            cfg_scale=None,
            quantization=None,
        )

        original_dir = os.getcwd()
        try:
            edit_image(args)

            # Check that edited image was created
            edited_files = list(Path("output").glob("edited-*.png"))
            assert len(edited_files) >= 1, "Should create at least one edited image"
            assert edited_files[0].stat().st_size > 0, "Edited file should not be empty"

        finally:
            os.chdir(original_dir)

    def test_anime_fast_mode(self, test_image_path):
        """Test: Anime transformation in fast mode (4 steps with Rapid-AIO)."""
        args = Namespace(
            input=[test_image_path],
            prompt="Add dramatic lighting",  # Optional with anime
            steps=40,  # Default, will be overridden by fast mode
            seed=555,
            fast=True,  # Fast mode enabled
            ultra_fast=False,
            output=None,
            output_dir="output",
            lora=None,
            batman=False,
            anime=True,  # Anime mode enabled
            negative_prompt=None,
            cfg_scale=None,
            quantization=None,
        )

        original_dir = os.getcwd()
        try:
            edit_image(args)

            # Check that edited image was created
            edited_files = list(Path("output").glob("edited-*.png"))
            assert len(edited_files) >= 1, "Should create at least one edited image"
            assert edited_files[0].stat().st_size > 0, "Edited file should not be empty"

        finally:
            os.chdir(original_dir)

    def test_anime_ultra_fast_mode(self, test_image_path):
        """Test: Anime transformation in ultra-fast mode (same as fast for editing)."""
        args = Namespace(
            input=[test_image_path],
            prompt=None,  # Prompt is optional with anime
            steps=40,  # Default, will be overridden by ultra-fast mode
            seed=666,
            fast=False,
            ultra_fast=True,  # Ultra-fast mode enabled
            output=None,
            output_dir="output",
            lora=None,
            batman=False,
            anime=True,  # Anime mode enabled
            negative_prompt=None,
            cfg_scale=None,
            quantization=None,
        )

        original_dir = os.getcwd()
        try:
            edit_image(args)

            # Check that edited image was created
            edited_files = list(Path("output").glob("edited-*.png"))
            assert len(edited_files) >= 1, "Should create at least one edited image"
            assert edited_files[0].stat().st_size > 0, "Edited file should not be empty"

        finally:
            os.chdir(original_dir)

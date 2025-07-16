import random
import string
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from DocTest.VisualTest import VisualTest


# Helper functions to create images and add text with boundary checking
def create_image(image_size=(500, 300), color=(255, 255, 255)):
    """
    Creates a blank image of the specified size and color.

    Args:
    - image_size: Tuple representing the size of the image (width, height).
    - color: Background color of the image (default is white).

    Returns:
    - img: A blank image of the given size and color.
    """
    img = np.ones((image_size[1], image_size[0], 3), np.uint8)
    img[:] = color
    return img


def add_text_to_image(img, text, text_position=(10, 50), font_scale=1, thickness=2):
    """
    Adds text to the given image at the specified position, ensuring the text does not go out of bounds.

    Args:
    - img: The image where text is to be added.
    - text: The string to add to the image.
    - text_position: The (x, y) coordinates where the text should be placed.
    - font_scale: The scale factor for the text size.
    - thickness: The thickness of the text.

    Returns:
    - img_with_text: The image with the text added.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Get the size of the text
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    # Calculate the boundaries of the image
    image_height, image_width = img.shape[:2]

    # Ensure the text does not go out of the image boundaries
    x, y = text_position
    if x + text_size[0] > image_width:  # Text is too wide, adjust x position
        x = image_width - text_size[0] - 10  # Add a small padding to ensure it fits

    if y - text_size[1] < 0:  # Text is too high, adjust y position
        y = text_size[1] + 10  # Add padding below the text

    if y > image_height:  # Text is too low, adjust y position
        y = image_height - 10  # Align text at the bottom with a small padding

    # Add text to the image at the adjusted position
    cv2.putText(img, text, (x, y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    return img


# Helper function to generate random text
def random_text(length=10):
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


# First Parametrized Test: Single Text Shift
@pytest.mark.parametrize(
    "text, move_distance",
    [
        ("Test 01", 10),
        ("Test 02", 5),
        ("Test 03", 15),
        ("Test 04", 20),
        ("Test 05", 0),
        ("Test 06", 30),
        ("Test 07", 45),
        ("Test 08", 2),
        ("Test 09", 25),
        ("Test 10", 1),
    ],
)
def test_find_existing_partial_image_with_sift(text, move_distance):
    """
    Unit test for detecting text shift between two images using SIFT.
    The text is shifted by the specified move_distance between the reference and candidate images.

    Args:
    - text: The text to be placed on the images.
    - move_distance: The number of pixels to shift the text vertically.
    """
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        # Create VisualTest object
        visual_tester = VisualTest(movement_detection="sift")

        # Create reference image with text
        ref_image = create_image((200, 100), (255, 255, 255))
        ref_image = add_text_to_image(ref_image, text)
        ref_image_path = temp_dir_path / f"ref_image_{text}.png"
        cv2.imwrite(str(ref_image_path), ref_image)

        # Create candidate image with the same text, but shifted vertically
        cand_image = create_image((200, 100), (255, 255, 255))
        cand_image = add_text_to_image(cand_image, text, (10, 50 + move_distance))
        cand_image_path = temp_dir_path / f"cand_image_{text}.png"
        cv2.imwrite(str(cand_image_path), cand_image)

        # Compare images and assert within move tolerance
        visual_tester.compare_images(
            ref_image_path, cand_image_path, move_tolerance=move_distance + 1
        )
        if move_distance > 0:
            with pytest.raises(AssertionError):
                visual_tester.compare_images(
                    ref_image_path, cand_image_path, move_tolerance=move_distance - 2
                )


# Second Parametrized Test: Multiple Texts with Shifts
@pytest.mark.parametrize("move_distance, tolerance", [(5, 6), (10, 12), (20, 22)])
def test_multiple_texts_with_sift(move_distance, tolerance):
    """
    Test for multiple texts added to larger images with various positions.
    Ensures no overlap and checks movement within tolerance for some texts.

    Args:
    - move_distance: The number of pixels to shift Text 2 and Text 3.
    - tolerance: The allowed movement tolerance.
    """
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        # Create VisualTest object
        visual_tester = VisualTest(movement_detection="sift")

        # Create a large blank image
        ref_image = create_image((500, 300), (255, 255, 255))
        cand_image = create_image((500, 300), (255, 255, 255))

        # Text 1: Same position on both reference and candidate image
        text1 = random_text(10)
        ref_image = add_text_to_image(ref_image, text1, (50, 50), font_scale=1)
        cand_image = add_text_to_image(cand_image, text1, (50, 50), font_scale=1)

        # Text 2: Shifted within tolerance between reference and candidate image
        text2 = random_text(10)
        ref_image = add_text_to_image(ref_image, text2, (200, 150), font_scale=1)
        cand_image = add_text_to_image(
            cand_image, text2, (200, 150 + move_distance), font_scale=1
        )

        # Text 3: Multiline text shifted within tolerance between reference and candidate image
        text3 = f"{random_text(8)}\n{random_text(8)}\n{random_text(8)}"
        ref_image = add_text_to_image(ref_image, text3, (300, 50), font_scale=0.8)
        cand_image = add_text_to_image(
            cand_image, text3, (300, 50 + move_distance), font_scale=0.8
        )

        # Save the images
        ref_image_path = temp_dir_path / "ref_image.png"
        cand_image_path = temp_dir_path / "cand_image.png"
        cv2.imwrite(str(ref_image_path), ref_image)
        cv2.imwrite(str(cand_image_path), cand_image)

        # Compare images with the provided tolerance
        visual_tester.compare_images(
            ref_image_path, cand_image_path, move_tolerance=tolerance
        )


# First Parametrized Test: Single Text Shift
@pytest.mark.parametrize(
    "text, move_distance",
    [
        ("Test 01", 10),
        ("Test 02", 5),
        ("Test 03", 15),
        ("Test 04", 20),
        ("Test 05", 0),
        ("Test 06", 30),
        ("Test 07", 45),
        ("Test 08", 2),
        ("Test 09", 25),
    ],
)
def test_find_existing_partial_image_with_template(text, move_distance):
    """
    Unit test for detecting text shift between two images using SIFT.
    The text is shifted by the specified move_distance between the reference and candidate images.

    Args:
    - text: The text to be placed on the images.
    - move_distance: The number of pixels to shift the text vertically.
    """
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        # Create VisualTest object
        visual_tester = VisualTest(movement_detection="template")

        # Create reference image with text
        ref_image = create_image((200, 100), (255, 255, 255))
        ref_image = add_text_to_image(ref_image, text)
        ref_image_path = temp_dir_path / f"ref_image_{text}.png"
        cv2.imwrite(str(ref_image_path), ref_image)

        # Create candidate image with the same text, but shifted vertically
        cand_image = create_image((200, 100), (255, 255, 255))
        cand_image = add_text_to_image(cand_image, text, (10, 50 + move_distance))
        cand_image_path = temp_dir_path / f"cand_image_{text}.png"
        cv2.imwrite(str(cand_image_path), cand_image)

        # Compare images and assert within move tolerance
        visual_tester.compare_images(
            ref_image_path, cand_image_path, move_tolerance=move_distance + 1
        )
        if move_distance > 0:
            with pytest.raises(AssertionError):
                visual_tester.compare_images(
                    ref_image_path, cand_image_path, move_tolerance=move_distance - 2
                )


# Second Parametrized Test: Multiple Texts with Shifts
@pytest.mark.parametrize("move_distance, tolerance", [(5, 6), (10, 12), (20, 22)])
def test_multiple_texts_with_template(move_distance, tolerance):
    """
    Test for multiple texts added to larger images with various positions.
    Ensures no overlap and checks movement within tolerance for some texts.

    Args:
    - move_distance: The number of pixels to shift Text 2 and Text 3.
    - tolerance: The allowed movement tolerance.
    """
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        # Create VisualTest object
        visual_tester = VisualTest(movement_detection="template")

        # Create a large blank image
        ref_image = create_image((500, 300), (255, 255, 255))
        cand_image = create_image((500, 300), (255, 255, 255))

        # Text 1: Same position on both reference and candidate image
        text1 = random_text(10)
        ref_image = add_text_to_image(ref_image, text1, (50, 50), font_scale=1)
        cand_image = add_text_to_image(cand_image, text1, (50, 50), font_scale=1)

        # Text 2: Shifted within tolerance between reference and candidate image
        text2 = random_text(10)
        ref_image = add_text_to_image(ref_image, text2, (200, 150), font_scale=1)
        cand_image = add_text_to_image(
            cand_image, text2, (200, 150 + move_distance), font_scale=1
        )

        # Text 3: Multiline text shifted within tolerance between reference and candidate image
        text3 = f"{random_text(8)}\n{random_text(8)}\n{random_text(8)}"
        ref_image = add_text_to_image(ref_image, text3, (300, 50), font_scale=0.8)
        cand_image = add_text_to_image(
            cand_image, text3, (300, 50 + move_distance), font_scale=0.8
        )

        # Save the images
        ref_image_path = temp_dir_path / "ref_image.png"
        cand_image_path = temp_dir_path / "cand_image.png"
        cv2.imwrite(str(ref_image_path), ref_image)
        cv2.imwrite(str(cand_image_path), cand_image)

        # Compare images with the provided tolerance
        visual_tester.compare_images(
            ref_image_path, cand_image_path, move_tolerance=tolerance
        )


@pytest.mark.parametrize(
    "move_distance, tolerance",
    [
        (0, 0),  # Exact match - no movement
        (1, 1),  # Minimal movement
        (10, 10),  # Medium Movement
        (49, 50),  # Minimal movement within tolerance
    ],
)
def test_exact_tolerance_boundary(move_distance, tolerance):
    """
    Test behavior when the movement is exactly at the tolerance boundary.

    Args:
    - move_distance: The number of pixels to shift the text
    - tolerance: The movement tolerance that should be just below or exactly at the move distance
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        # Test with both SIFT and template methods
        for detection_method in ["template", "sift"]:
            visual_tester = VisualTest(movement_detection=detection_method)

            # Create test images
            ref_image = create_image((300, 200), (255, 255, 255))
            cand_image = create_image((300, 200), (255, 255, 255))

            text = f"Exact Tolerance {detection_method}"
            ref_image = add_text_to_image(ref_image, text, (50, 50))
            cand_image = add_text_to_image(cand_image, text, (50, 50 + move_distance))

            ref_path = temp_dir_path / f"ref_exact_{detection_method}.png"
            cand_path = temp_dir_path / f"cand_exact_{detection_method}.png"
            cv2.imwrite(str(ref_path), ref_image)
            cv2.imwrite(str(cand_path), cand_image)

            # Should pass when tolerance >= movement
            if move_distance <= tolerance:
                visual_tester.compare_images(
                    ref_path, cand_path, move_tolerance=tolerance
                )

            # Should fail when tolerance < movement
            if move_distance > tolerance:
                with pytest.raises(AssertionError):
                    visual_tester.compare_images(
                        ref_path, cand_path, move_tolerance=tolerance
                    )


@pytest.mark.parametrize(
    "direction, coords",
    [
        ("up", (0, -15)),
        ("down", (0, 15)),
        ("left", (-15, 0)),
        ("right", (15, 0)),
        ("diagonal", (10, 10)),
    ],
)
def test_movement_directions(direction, coords):
    """
    Test detection of movement in different directions.

    Args:
    - direction: String representing the movement direction (for test naming)
    - coords: Tuple of (x, y) pixel shifts to apply
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        # Test with both SIFT and template methods
        for detection_method in ["template", "sift"]:
            visual_tester = VisualTest(movement_detection=detection_method)

            # Calculate move distance for tolerance
            x_shift, y_shift = coords
            move_distance = int(np.sqrt(x_shift**2 + y_shift**2))

            # Create test images
            ref_image = create_image((600, 200), (255, 255, 255))
            cand_image = create_image((600, 200), (255, 255, 255))

            text = f"Direction {direction}"
            ref_image = add_text_to_image(ref_image, text, (100, 100))
            cand_image = add_text_to_image(
                cand_image, text, (100 + x_shift, 100 + y_shift)
            )

            ref_path = temp_dir_path / f"ref_{direction}_{detection_method}.png"
            cand_path = temp_dir_path / f"cand_{direction}_{detection_method}.png"
            cv2.imwrite(str(ref_path), ref_image)
            cv2.imwrite(str(cand_path), cand_image)

            # Test with exact tolerance
            visual_tester.compare_images(
                ref_path, cand_path, move_tolerance=move_distance
            )

            # Test with insufficient tolerance
            with pytest.raises(AssertionError):
                visual_tester.compare_images(
                    ref_path, cand_path, move_tolerance=move_distance - 3
                )


def test_multiple_independent_movements():
    """
    Test detection of multiple elements moving in different directions with different distances.
    Verifies that each movement is detected independently.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        # Test with both detection methods
        for detection_method in ["sift", "template"]:
            visual_tester = VisualTest(movement_detection=detection_method)

            # Create large images
            ref_image = create_image((600, 400), (255, 255, 255))
            cand_image = create_image((600, 400), (255, 255, 255))

            # Element 1: Text moving down by 5 pixels
            text1 = random_text(10)
            ref_image = add_text_to_image(ref_image, text1, (100, 100))
            cand_image = add_text_to_image(cand_image, text1, (100, 105))

            # Element 2: Text moving right by 10 pixels
            text2 = random_text(10)
            ref_image = add_text_to_image(ref_image, text2, (300, 100))
            cand_image = add_text_to_image(cand_image, text2, (310, 100))

            # Element 3: Text moving diagonally by 10 pixels (√(7²+7²) = ~10)
            text3 = random_text(10)
            ref_image = add_text_to_image(ref_image, text3, (100, 300))
            cand_image = add_text_to_image(cand_image, text3, (107, 307))

            # Element 4: Text not moving
            text4 = random_text(10)
            ref_image = add_text_to_image(ref_image, text4, (300, 300))
            cand_image = add_text_to_image(cand_image, text4, (300, 300))

            ref_path = temp_dir_path / f"ref_multi_movement_{detection_method}.png"
            cand_path = temp_dir_path / f"cand_multi_movement_{detection_method}.png"
            cv2.imwrite(str(ref_path), ref_image)
            cv2.imwrite(str(cand_path), cand_image)

            # Test with tolerance that accepts all movements
            visual_tester.compare_images(ref_path, cand_path, move_tolerance=10)

            # Test with tolerance that rejects the 10px movements
            with pytest.raises(AssertionError):
                visual_tester.compare_images(ref_path, cand_path, move_tolerance=7)


@pytest.mark.parametrize("font_scale_change", [0.8, 0.9, 1.1, 1.2])
def test_text_size_changes_with_movement(font_scale_change):
    """
    Test behavior when text not only moves but also changes size.

    Args:
    - font_scale_change: Factor to change the font size by in the candidate image
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        move_distance = 15

        # Test with both detection methods
        for detection_method in ["sift", "template"]:
            visual_tester = VisualTest(movement_detection=detection_method)

            # Create test images
            ref_image = create_image((300, 200), (255, 255, 255))
            cand_image = create_image((300, 200), (255, 255, 255))

            text = f"Size Change {font_scale_change}"
            ref_image = add_text_to_image(ref_image, text, (50, 100), font_scale=1.0)
            cand_image = add_text_to_image(
                cand_image,
                text,
                (50, 100 + move_distance),
                font_scale=font_scale_change,
            )

            ref_path = (
                temp_dir_path / f"ref_size_{detection_method}_{font_scale_change}.png"
            )
            cand_path = (
                temp_dir_path / f"cand_size_{detection_method}_{font_scale_change}.png"
            )
            cv2.imwrite(str(ref_path), ref_image)
            cv2.imwrite(str(cand_path), cand_image)

            # Test with sufficient tolerance - may fail depending on algorithm sensitivity to size changes
            try:
                visual_tester.compare_images(
                    ref_path, cand_path, move_tolerance=move_distance + 5
                )
                size_sensitive = False
            except AssertionError:
                size_sensitive = True

            # Document the behavior
            print(
                f"Method {detection_method} is {'sensitive' if size_sensitive else 'not sensitive'} "
                f"to font scale change of {font_scale_change}"
            )


@pytest.mark.parametrize("quality", [20, 50, 80, 100])
def test_image_quality_impact(quality):
    """
    Test how JPEG compression quality affects movement detection.

    Args:
    - quality: JPEG compression quality (0-100)
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        move_distance = 15

        # Test with both detection methods
        for detection_method in ["sift", "template"]:
            visual_tester = VisualTest(movement_detection=detection_method)

            # Create test images
            ref_image = create_image((300, 200), (255, 255, 255))
            cand_image = create_image((300, 200), (255, 255, 255))

            text = "Quality Test"
            ref_image = add_text_to_image(ref_image, text, (50, 100))
            cand_image = add_text_to_image(cand_image, text, (50, 100 + move_distance))

            # Save as JPEG with specified quality
            ref_path = temp_dir_path / f"ref_quality_{detection_method}_{quality}.jpg"
            cand_path = temp_dir_path / f"cand_quality_{detection_method}_{quality}.jpg"
            cv2.imwrite(str(ref_path), ref_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            cv2.imwrite(str(cand_path), cand_image, [cv2.IMWRITE_JPEG_QUALITY, quality])

            # Test with exact tolerance
            visual_tester.compare_images(
                ref_path, cand_path, move_tolerance=move_distance
            )

            # Test with insufficient tolerance
            with pytest.raises(AssertionError):
                visual_tester.compare_images(
                    ref_path, cand_path, move_tolerance=move_distance - 5
                )

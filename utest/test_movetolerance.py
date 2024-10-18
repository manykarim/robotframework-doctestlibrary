from DocTest.VisualTest import VisualTest
import pytest
import cv2
import numpy as np
from pathlib import Path
import tempfile
import random
import string

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
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


# First Parametrized Test: Single Text Shift
@pytest.mark.parametrize("text, move_distance", [
    ("Test 1", 10),
    ("Test 2", 5),
    ("Test 3", 15),
    ("Test 4", 20),
    ("Test 5", 0),
    ("Test 6", 30),
    ("Test 7", 45),
    ("Test 8", 2),
    ("Test 9", 25),
    ("Test 10", 1)
])
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
        visual_tester = VisualTest()

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
        visual_tester.compare_images(ref_image_path, cand_image_path, move_tolerance=move_distance + 1)
        if move_distance > 0:
            with pytest.raises(AssertionError):
                visual_tester.compare_images(ref_image_path, cand_image_path, move_tolerance=move_distance - 2)

# Second Parametrized Test: Multiple Texts with Shifts
@pytest.mark.parametrize("move_distance, tolerance", [
    (5, 6),
    (10, 12),
    (20, 22)
])
def test_multiple_texts_with_shifts(move_distance, tolerance):
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
        visual_tester = VisualTest()

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
        cand_image = add_text_to_image(cand_image, text2, (200, 150 + move_distance), font_scale=1)

        # Text 3: Multiline text shifted within tolerance between reference and candidate image
        text3 = f"{random_text(8)}\n{random_text(8)}\n{random_text(8)}"
        ref_image = add_text_to_image(ref_image, text3, (300, 50), font_scale=0.8)
        cand_image = add_text_to_image(cand_image, text3, (300, 50 + move_distance), font_scale=0.8)

        # Save the images
        ref_image_path = temp_dir_path / "ref_image.png"
        cand_image_path = temp_dir_path / "cand_image.png"
        cv2.imwrite(str(ref_image_path), ref_image)
        cv2.imwrite(str(cand_image_path), cand_image)

        # Compare images with the provided tolerance
        visual_tester.compare_images(ref_image_path, cand_image_path, move_tolerance=tolerance)


# If you are using pytest in the terminal, this code would be run using:
# pytest <name_of_test_file>.py

import logging
from PIL import Image, ImageChops

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_new_dimensions(original_width: int, original_height: int, scale_factor: float) -> tuple[int, int]:
    """
    Calculates new dimensions based on a scale factor, preserving aspect ratio.

    Args:
        original_width: The original width of the image.
        original_height: The original height of the image.
        scale_factor: The factor by which to scale the dimensions.

    Returns:
        A tuple (new_width, new_height).
    """
    if original_width <= 0 or original_height <= 0:
        logging.warning("Original dimensions must be positive.")
        return original_width, original_height
    if scale_factor <= 0:
        logging.warning("Scale factor must be positive.")
        return original_width, original_height

    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    logging.info(f"Calculated new dimensions: ({new_width}, {new_height}) for scale factor {scale_factor}")
    return new_width, new_height

def get_image_dpi(image_path: str) -> tuple[float | None, float | None] | None:
    """
    Extracts DPI (dots per inch) from image metadata using Pillow.

    Args:
        image_path: Path to the image file.

    Returns:
        A tuple (dpi_x, dpi_y) if DPI information is found, otherwise None.
        DPI values can be None if only one direction is specified.
    """
    try:
        with Image.open(image_path) as img:
            dpi = img.info.get('dpi')
            if dpi:
                if isinstance(dpi, tuple) and len(dpi) == 2:
                    logging.info(f"Extracted DPI: {dpi} from {image_path}")
                    return float(dpi[0]), float(dpi[1])
                elif isinstance(dpi, (int, float)): # Some images might store a single DPI value
                    logging.info(f"Extracted single DPI value: {dpi} from {image_path}. Assuming X and Y are the same.")
                    return float(dpi), float(dpi)
                else:
                    logging.warning(f"Unexpected DPI format in {image_path}: {dpi}")
                    return None
            # Attempt to read resolution from other tags if 'dpi' is not present
            x_resolution = img.info.get('x_resolution')
            y_resolution = img.info.get('y_resolution')
            if x_resolution and y_resolution:
                 # Exif might store resolution as a tuple (numerator, denominator)
                if isinstance(x_resolution, tuple):
                    x_res = x_resolution[0] / x_resolution[1] if x_resolution[1] != 0 else x_resolution[0]
                else:
                    x_res = x_resolution
                if isinstance(y_resolution, tuple):
                    y_res = y_resolution[0] / y_resolution[1] if y_resolution[1] != 0 else y_resolution[0]
                else:
                    y_res = y_resolution

                if x_res and y_res: # Ensure they are not zero
                    logging.info(f"Extracted X/Y resolution: ({x_res}, {y_res}) from {image_path}")
                    return float(x_res), float(y_res)

            logging.info(f"No DPI information found in {image_path}")
            return None
    except FileNotFoundError:
        logging.error(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        logging.error(f"Error reading DPI from {image_path}: {e}")
        return None

def set_image_dpi(image: Image.Image, dpi_x: float | None = None, dpi_y: float | None = None) -> Image.Image:
    """
    Sets DPI for a PIL Image object. Modifies the image info directly.

    Args:
        image: The PIL Image object.
        dpi_x: The DPI value for the x-axis.
        dpi_y: The DPI value for the y-axis. If None, dpi_x is used.

    Returns:
        The PIL Image object with DPI information updated in its info dictionary.
    """
    if dpi_x is None and dpi_y is None:
        logging.info("No DPI values provided to set.")
        return image

    if dpi_x is not None and dpi_y is None:
        dpi_y = dpi_x
    elif dpi_y is not None and dpi_x is None:
        dpi_x = dpi_y

    if dpi_x is not None and dpi_y is not None:
        image.info['dpi'] = (float(dpi_x), float(dpi_y))
        # Some software might read these specific exif tags for resolution
        image.info['x_resolution'] = float(dpi_x)
        image.info['y_resolution'] = float(dpi_y)
        # ResolutionUnit: 2 for inches, 3 for cm. Defaulting to inches.
        image.info['resolution_unit'] = 2
        logging.info(f"Set DPI to ({dpi_x}, {dpi_y}) for the image.")
    return image

# Placeholder for PIL Image to Tensor and Tensor to PIL Image conversion functions
# These might be more specific to the upscaler's needs and could be defined
# within upscaler.py or here if they are generic enough.

# Example (if using numpy as an intermediary, which is common):
# import numpy as np
# import torch

# def pil_to_tensor(pil_image: Image.Image) -> torch.Tensor:
#     """Converts a PIL Image to a PyTorch tensor."""
#     numpy_image = np.array(pil_image).astype(np.float32) / 255.0
#     # Assuming CHW format for PyTorch
#     if numpy_image.ndim == 2: # Grayscale image
#         numpy_image = np.expand_dims(numpy_image, axis=2)
#     tensor_image = torch.from_numpy(numpy_image).permute(2, 0, 1).unsqueeze(0)
#     return tensor_image

# def tensor_to_pil(tensor_image: torch.Tensor) -> Image.Image:
#     """Converts a PyTorch tensor to a PIL Image."""
#     # Assuming tensor is in CHW format, remove batch dim, permute to HWC
#     processed_tensor = tensor_image.squeeze(0).permute(1, 2, 0)
#     # Clamp values to [0, 1] and scale to [0, 255]
#     numpy_image = (processed_tensor.clamp(0, 1).numpy() * 255).astype(np.uint8)
#     if numpy_image.shape[2] == 1: # Grayscale
#         pil_image = Image.fromarray(numpy_image.squeeze(2), mode='L')
#     else: # RGB
#         pil_image = Image.fromarray(numpy_image, mode='RGB')
#     return pil_image

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    print("--- Testing utils.py ---")

    # Test calculate_new_dimensions
    w, h = 100, 200
    scale = 2.0
    new_w, new_h = calculate_new_dimensions(w, h, scale)
    print(f"Original: {w}x{h}, Scale: {scale} -> New: {new_w}x{new_h}")
    assert new_w == 200 and new_h == 400

    scale = 0.5
    new_w, new_h = calculate_new_dimensions(w, h, scale)
    print(f"Original: {w}x{h}, Scale: {scale} -> New: {new_w}x{new_h}")
    assert new_w == 50 and new_h == 100

    # Test DPI functions (requires a sample image with DPI info)
    # Create a dummy image for testing if one doesn't exist
    try:
        # Create a dummy PNG file for testing DPI functions
        dummy_image_path = "dummy_test_image.png"
        img = Image.new('RGB', (60, 30), color = 'red')
        img.save(dummy_image_path, dpi=(150, 150))
        print(f"Created dummy image: {dummy_image_path} with DPI 150x150")

        dpi_info = get_image_dpi(dummy_image_path)
        if dpi_info:
            print(f"DPI for {dummy_image_path}: {dpi_info}")
            assert dpi_info == (150.0, 150.0)
        else:
            print(f"No DPI info found for {dummy_image_path} or error occurred.")

        # Test setting DPI
        img_to_modify = Image.open(dummy_image_path)
        modified_img = set_image_dpi(img_to_modify, dpi_x=300, dpi_y=300)

        # Save and reopen to check if DPI was set (Pillow might not show it directly in .info before save)
        modified_image_path = "dummy_modified_dpi.png"
        modified_img.save(modified_image_path) # No need to pass DPI here, it's in info

        # Re-open and check DPI
        reopened_img = Image.open(modified_image_path)
        reopened_dpi = reopened_img.info.get('dpi')
        print(f"DPI for modified image {modified_image_path} (after save & reopen): {reopened_dpi}")
        if reopened_dpi:
            assert reopened_dpi == (300.0, 300.0)
        else:
            # Fallback check if 'dpi' key is not directly there but x/y_resolution are
            x_res = reopened_img.info.get('x_resolution')
            y_res = reopened_img.info.get('y_resolution')
            print(f"X/Y resolution: {x_res}, {y_res}")
            assert x_res == 300.0 and y_res == 300.0


        # Clean up dummy files
        import os
        os.remove(dummy_image_path)
        os.remove(modified_image_path)
        print("Cleaned up dummy images.")

    except ImportError:
        print("Pillow is not installed. Skipping DPI tests.")
    except Exception as e:
        print(f"An error occurred during DPI test setup or execution: {e}")

    print("--- Finished testing utils.py ---")

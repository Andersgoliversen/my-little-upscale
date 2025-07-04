import logging
import os
import torch
from PIL import Image
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import cv2 # For downsampling if needed

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Model Configuration ---
# User needs to manually download RealESRGAN_x4plus.pth
# from https://github.com/xinntao/Real-ESRGAN/releases
# and place it in the models/ directory.
# Optionally, RealESRGAN_x2plus.pth can also be downloaded for native 2x.
DEFAULT_MODEL_DIR = "models"
MODEL_NAME_X4 = "RealESRGAN_x4plus.pth"
MODEL_NAME_X2 = "RealESRGAN_x2plus.pth" # Optional

# Global variable to hold the loaded upscaler instance
# This helps in loading the model only once.
_upsampler_cache = {}

def get_realesrgan_upsampler(scale_factor: int, model_dir: str = DEFAULT_MODEL_DIR) -> RealESRGANer | None:
    """
    Loads the RealESRGAN model based on the desired scale factor.
    Caches the loaded model to avoid reloading.

    Args:
        scale_factor: The desired upscale factor (e.g., 2 or 4).
        model_dir: The directory where model files are stored.

    Returns:
        A RealESRGANer instance if successful, otherwise None.
    """
    global _upsampler_cache

    if scale_factor in _upsampler_cache:
        logging.info(f"Using cached RealESRGANer for scale {scale_factor}x.")
        return _upsampler_cache[scale_factor]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    model_path_x4 = os.path.join(model_dir, MODEL_NAME_X4)
    model_path_x2 = os.path.join(model_dir, MODEL_NAME_X2)

    selected_model_path = None
    model_netscale = 4 # Default to x4 model

    if scale_factor == 4:
        if os.path.exists(model_path_x4):
            selected_model_path = model_path_x4
            model_netscale = 4
        else:
            logging.error(f"Model not found: {model_path_x4}. Please download it.")
            return None
    elif scale_factor == 2:
        if os.path.exists(model_path_x2): # Prefer native x2 model if available
            selected_model_path = model_path_x2
            model_netscale = 2
        elif os.path.exists(model_path_x4): # Fallback to x4 model and then downscale
            selected_model_path = model_path_x4
            model_netscale = 4
            logging.info("Native 2x model not found. Using 4x model and will downscale.")
        else:
            logging.error(f"Neither {MODEL_NAME_X2} nor {MODEL_NAME_X4} found in {model_dir}.")
            return None
    else:
        logging.error(f"Unsupported scale factor: {scale_factor}. Only 2 or 4 are supported.")
        return None

    try:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=model_netscale)

        upsampler = RealESRGANer(
            scale=model_netscale,
            model_path=selected_model_path,
            dni_weight=None, # Use model's default behavior
            model=model,
            tile=0, # Tile size for processing, 0 for auto
            tile_pad=10,
            pre_pad=0,
            half=True if device.type == 'cuda' else False, # Use half precision on GPU
            gpu_id=None if device.type == 'cpu' else 0 # Specify GPU id if multiple GPUs
        )
        _upsampler_cache[scale_factor] = upsampler # Cache for the original requested scale
        logging.info(f"RealESRGAN model loaded successfully for {model_netscale}x from {selected_model_path}")
        return upsampler
    except Exception as e:
        logging.error(f"Failed to load RealESRGAN model: {e}")
        if str(e).startswith("Attempting to deserialize object on a CUDA device"):
            logging.error("This error might be due to loading a CUDA-trained model on a CPU-only environment without proper map_location. RealESRGANer should handle this, but check basicsr/torch versions if issues persist.")
        return None


def upscale_image(image_path: str, scale_factor: int, model_dir: str = DEFAULT_MODEL_DIR) -> Image.Image | None:
    """
    Upscales an image using Real-ESRGAN.

    Args:
        image_path: Path to the input image.
        scale_factor: The desired upscale factor (2 or 4).
        model_dir: Directory containing the model files.

    Returns:
        A PIL Image object of the upscaled image, or None if an error occurs.
    """
    try:
        img_pil = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        logging.error(f"Image file not found: {image_path}")
        return None
    except Exception as e:
        logging.error(f"Could not open image {image_path}: {e}")
        return None

    # Convert PIL image to OpenCV format (numpy array) which RealESRGANer expects
    img_cv = np.array(img_pil)
    # RealESRGANer expects BGR, Pillow opens as RGB
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    upsampler = get_realesrgan_upsampler(scale_factor, model_dir)
    if not upsampler:
        return None

    try:
        # Perform inference
        # The 'outscale' parameter in upsampler.enhance is the final desired scale.
        # The upsampler itself is loaded based on its native scale (model_netscale).
        output_cv, _ = upsampler.enhance(img_cv, outscale=scale_factor)

        if output_cv is None:
            logging.error("Upscaling process returned None.")
            return None

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logging.error("CUDA out of memory during upscaling. Try reducing tile size or processing smaller images.")
            # Potentially clear cache and try CPU if that's a desired fallback,
            # but RealESRGANer already has a CPU fallback if CUDA is not available.
            # For now, just report error.
        else:
            logging.error(f"Runtime error during upscaling: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during upscaling: {e}")
        return None

    # Convert OpenCV BGR output back to RGB Pillow Image
    output_rgb = cv2.cvtColor(output_cv, cv2.COLOR_BGR2RGB)
    upscaled_image_pil = Image.fromarray(output_rgb)

    logging.info(f"Image {image_path} successfully upscaled by {scale_factor}x.")
    return upscaled_image_pil


if __name__ == '__main__':
    print("--- Testing upscaler.py ---")
    # Before running this test, ensure you have:
    # 1. Installed all dependencies from requirements.txt
    # 2. Downloaded RealESRGAN_x4plus.pth (and optionally RealESRGAN_x2plus.pth)
    #    and placed it in a 'models' directory relative to this script,
    #    or update `test_model_dir`.

    test_model_dir = "models" # Relative to the script's location if run directly
    # Create models directory if it doesn't exist for the test
    if not os.path.exists(test_model_dir):
        os.makedirs(test_model_dir)
        print(f"Created directory: {test_model_dir}")
        print(f"Please download Real-ESRGAN models and place them in '{test_model_dir}'.")
        print(f"Attempting to download {MODEL_NAME_X4} for testing purposes...")
        try:
            import requests
            model_url_x4 = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
            response = requests.get(model_url_x4, stream=True)
            response.raise_for_status()
            with open(os.path.join(test_model_dir, MODEL_NAME_X4), 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Successfully downloaded {MODEL_NAME_X4} to {test_model_dir}")
        except Exception as e:
            print(f"Could not automatically download {MODEL_NAME_X4}. Please download it manually. Error: {e}")
            # exit() # Exit if model cannot be downloaded for test

    # Create a dummy image for testing
    dummy_input_image_path = "dummy_input_for_upscale.png"
    try:
        img = Image.new('RGB', (50, 40), color='blue') # Small dimensions for quick test
        img.save(dummy_input_image_path)
        print(f"Created dummy input image: {dummy_input_image_path} (50x40)")
    except Exception as e:
        print(f"Failed to create dummy image: {e}")
        exit()

    # Test Case 1: Upscale by 4x
    print("\n--- Test Case 1: Upscale by 4x ---")
    if os.path.exists(os.path.join(test_model_dir, MODEL_NAME_X4)):
        upscaled_img_4x = upscale_image(dummy_input_image_path, scale_factor=4, model_dir=test_model_dir)
        if upscaled_img_4x:
            print(f"4x Upscaled image dimensions: {upscaled_img_4x.size}")
            assert upscaled_img_4x.size == (200, 160)
            upscaled_img_4x.save("dummy_upscaled_4x.png")
            print("Saved 4x upscaled image to dummy_upscaled_4x.png")
        else:
            print("4x Upscaling failed.")
    else:
        print(f"{MODEL_NAME_X4} not found in {test_model_dir}. Skipping 4x test.")

    # Test Case 2: Upscale by 2x
    # This will use RealESRGAN_x4plus.pth and downscale if RealESRGAN_x2plus.pth is not present.
    print("\n--- Test Case 2: Upscale by 2x ---")
    # Check if x2 or x4 model exists for 2x test
    can_run_2x_test = os.path.exists(os.path.join(test_model_dir, MODEL_NAME_X2)) or \
                      os.path.exists(os.path.join(test_model_dir, MODEL_NAME_X4))

    if can_run_2x_test:
        upscaled_img_2x = upscale_image(dummy_input_image_path, scale_factor=2, model_dir=test_model_dir)
        if upscaled_img_2x:
            print(f"2x Upscaled image dimensions: {upscaled_img_2x.size}")
            assert upscaled_img_2x.size == (100, 80)
            upscaled_img_2x.save("dummy_upscaled_2x.png")
            print("Saved 2x upscaled image to dummy_upscaled_2x.png")
        else:
            print("2x Upscaling failed.")
    else:
        print(f"Neither {MODEL_NAME_X2} nor {MODEL_NAME_X4} found in {test_model_dir}. Skipping 2x test.")

    # Test Case 3: File not found
    print("\n--- Test Case 3: File not found ---")
    non_existent_image = "non_existent_image.png"
    result_non_existent = upscale_image(non_existent_image, scale_factor=4, model_dir=test_model_dir)
    assert result_non_existent is None
    print(f"Test with non-existent file '{non_existent_image}' completed (expected failure).")

    # Test Case 4: Unsupported scale factor
    print("\n--- Test Case 4: Unsupported scale factor ---")
    result_unsupported_scale = upscale_image(dummy_input_image_path, scale_factor=3, model_dir=test_model_dir)
    assert result_unsupported_scale is None
    print(f"Test with unsupported scale factor 3 completed (expected failure).")

    # Clean up dummy files
    print("\nCleaning up dummy files...")
    if os.path.exists(dummy_input_image_path):
        os.remove(dummy_input_image_path)
    if os.path.exists("dummy_upscaled_4x.png"):
        os.remove("dummy_upscaled_4x.png")
    if os.path.exists("dummy_upscaled_2x.png"):
        os.remove("dummy_upscaled_2x.png")
    print("Cleanup complete.")

    print("\n--- Finished testing upscaler.py ---")

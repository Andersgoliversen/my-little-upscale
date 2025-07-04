# My Little Upscale

A Python application for image upscaling with a graphical user interface using Real-ESRGAN.

## Features

- Load images (JPEG, PNG, BMP, TIFF, WebP).
- Upscale images by 2x or 4x.
- Preview original and upscaled images.
- Save upscaled images in various formats (PNG, TIFF, JPEG).
- Displays image dimensions and DPI.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Andersgoliversen/my-little-upscale.git
    cd my-little-upscale
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Pre-trained Models:**
    You need to manually download the Real-ESRGAN pre-trained models and place them in the `my-little-upscale/models/` directory.
    - Download `RealESRGAN_x4plus.pth` from the [Real-ESRGAN releases](https://github.com/xinntao/Real-ESRGAN/releases).
    - (Optional, if you want native 2x scaling) Download `RealESRGAN_x2plus.pth` and place it in the same `models/` directory. If not found, 2x scaling will be achieved by 4x upscaling followed by downsampling.

## How to Run

1.  Ensure you have completed the Setup steps.
2.  Run the application:
    ```bash
    python main.py
    ```

## Usage

1.  Click "Load Image" to select an image file.
2.  The original image will be displayed along with its dimensions and DPI.
3.  Select an upscale factor (2x or 4x).
4.  Click "Upscale Image".
5.  The upscaled image will be displayed along with its new dimensions.
6.  Click "Save As" to save the upscaled image.

---
*This README will be updated as the project progresses.*

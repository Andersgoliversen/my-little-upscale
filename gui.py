import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, UnidentifiedImageError
import os
import logging
import threading # For running upscale in a separate thread

# Assuming upscaler.py and utils.py are in the same directory or accessible in PYTHONPATH
try:
    import upscaler
    import utils # Ensure utils is also checked here
except ModuleNotFoundError as e:
    messagebox.showerror("Error", f"Could not import 'upscaler' or 'utils' modules. Ensure they are in the same directory and all dependencies are installed. Error: {e}")
    exit()


# Configure basic logging (can be centralized if needed)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImageUpscalerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("My Little Upscale")
        self.root.geometry("900x700") # Adjusted for better layout

        self.image_path = None
        self.original_pil_image = None
        self.upscaled_pil_image = None
        self.original_dpi = (None, None)

        # --- Main Layout Frames ---
        control_frame = ttk.Frame(root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X, expand=False)

        image_preview_frame = ttk.Frame(root, padding="5")
        image_preview_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        status_frame = ttk.Frame(root, padding="5")
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False)

        # --- Control Frame Widgets ---
        # Load Image Button
        self.load_button = ttk.Button(control_frame, text="Load Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Upscale Options
        ttk.Label(control_frame, text="Upscale Factor:").pack(side=tk.LEFT, padx=(20, 5), pady=5)
        self.scale_var = tk.StringVar(value="4") # Default to 4x
        self.scale_radio_2x = ttk.Radiobutton(control_frame, text="2x", variable=self.scale_var, value="2")
        self.scale_radio_2x.pack(side=tk.LEFT, padx=5, pady=5)
        self.scale_radio_4x = ttk.Radiobutton(control_frame, text="4x", variable=self.scale_var, value="4")
        self.scale_radio_4x.pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Label(control_frame, text="Target DPI:").pack(side=tk.LEFT, padx=(20, 5), pady=5)
        self.dpi_var = tk.StringVar(value="300")
        self.dpi_entry = ttk.Entry(control_frame, width=6, textvariable=self.dpi_var)
        self.dpi_entry.pack(side=tk.LEFT, padx=5, pady=5)
        vcmd = (self.root.register(self._validate_dpi), '%P')
        self.dpi_entry.config(validate='key', validatecommand=vcmd)

        # Upscale Button
        self.upscale_button = ttk.Button(control_frame, text="Upscale Image", command=self.start_upscale_thread, state=tk.DISABLED)
        self.upscale_button.pack(side=tk.LEFT, padx=20, pady=5)

        # Save Button
        self.save_button = ttk.Button(control_frame, text="Save As...", command=self.save_image, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=5, pady=5)

        # --- Image Preview Frame ---
        # Original Image Area
        original_frame = ttk.LabelFrame(image_preview_frame, text="Original Image", padding="10")
        original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.original_image_label = ttk.Label(original_frame, text="Load an image to see preview", anchor=tk.CENTER)
        self.original_image_label.pack(fill=tk.BOTH, expand=True)
        self.original_info_label = ttk.Label(original_frame, text="Dimensions: N/A | DPI: N/A")
        self.original_info_label.pack(side=tk.BOTTOM, fill=tk.X)

        # Upscaled Image Area
        upscaled_frame = ttk.LabelFrame(image_preview_frame, text="Upscaled Image", padding="10")
        upscaled_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.upscaled_image_label = ttk.Label(upscaled_frame, text="Upscaled image will appear here", anchor=tk.CENTER)
        self.upscaled_image_label.pack(fill=tk.BOTH, expand=True)
        self.upscaled_info_label = ttk.Label(upscaled_frame, text="Dimensions: N/A | DPI: N/A")
        self.upscaled_info_label.pack(side=tk.BOTTOM, fill=tk.X)

        # --- Status Frame Widgets ---
        self.status_label = ttk.Label(status_frame, text="Ready. Load an image to start.")
        self.status_label.pack(side=tk.LEFT, padx=5, pady=5)
        # Determinate progress bar for showing percentage completed
        self.progress_bar = ttk.Progressbar(status_frame, mode='determinate', maximum=100)
        # self.progress_bar.pack(side=tk.RIGHT, padx=5, pady=5)  # Packed when active

    def _update_status(self, message, show_progress=False):
        """Update the status label and optionally show the progress bar."""
        self.status_label.config(text=message)
        if show_progress:
            if not self.progress_bar.winfo_ismapped():
                self.progress_bar.pack(side=tk.RIGHT, padx=5, pady=5, fill=tk.X, expand=True)
            # Reset bar when shown in determinate mode
            self.progress_bar['value'] = 0
        else:
            if self.progress_bar.winfo_ismapped():
                self.progress_bar.pack_forget()
            self.progress_bar['value'] = 0
        self.root.update_idletasks()

    def _set_progress(self, percent):
        """Set progress bar value safely."""
        self.progress_bar['value'] = max(0, min(100, percent))
        self.root.update_idletasks()

    def _validate_dpi(self, text):
        return text.isdigit() or text == ""

    def _get_target_dpi(self) -> int | None:
        try:
            dpi = int(self.dpi_var.get())
            return dpi if dpi > 0 else None
        except ValueError:
            return None


    def _display_image(self, pil_image, label_widget, max_size=(380, 380)):
        if not pil_image:
            label_widget.configure(image='', text="Preview not available.")
            label_widget.image = None # Keep a reference
            return

        try:
            img_copy = pil_image.copy()
            img_copy.thumbnail(max_size, Image.Resampling.LANCZOS)
            photo_image = ImageTk.PhotoImage(img_copy)

            label_widget.configure(image=photo_image, text="")
            label_widget.image = photo_image # Keep a reference!
        except Exception as e:
            label_widget.configure(image='', text=f"Error displaying preview: {e}")
            label_widget.image = None
            logging.error(f"Error creating PhotoImage: {e}")


    def load_image(self):
        file_types = [
            ('Image Files', '*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp'),
            ('All Files', '*.*')
        ]
        path = filedialog.askopenfilename(title="Select an Image", filetypes=file_types)
        if not path:
            return

        self.image_path = path
        self._update_status(f"Loading image: {os.path.basename(path)}", True)

        try:
            self.original_pil_image = Image.open(self.image_path)
            # Ensure image is in RGB for consistent processing later if needed,
            # though upscaler handles its own conversions.
            # self.original_pil_image = self.original_pil_image.convert("RGB")

            self._display_image(self.original_pil_image, self.original_image_label)

            w, h = self.original_pil_image.size
            self.original_dpi = utils.get_image_dpi(self.image_path) or (None, None)
            dpi_str = f"{self.original_dpi[0] or 'N/A'}x{self.original_dpi[1] or 'N/A'}" if self.original_dpi[0] or self.original_dpi[1] else "N/A"

            self.original_info_label.config(text=f"Dimensions: {w}x{h} | DPI: {dpi_str}")

            self.upscale_button.config(state=tk.NORMAL)
            self.save_button.config(state=tk.DISABLED) # Disable save until upscale
            self.upscaled_image_label.configure(image='', text="Upscaled image will appear here")
            self.upscaled_image_label.image = None
            self.upscaled_info_label.config(text="Dimensions: N/A | DPI: N/A")
            self.upscaled_pil_image = None
            self._update_status(f"Loaded: {os.path.basename(path)}")

        except FileNotFoundError:
            messagebox.showerror("Error", f"File not found: {self.image_path}")
            self._update_status("Error: File not found.", False)
        except UnidentifiedImageError:
            messagebox.showerror("Error", f"Cannot identify image file: {self.image_path}. It may be corrupted or an unsupported format.")
            self._update_status("Error: Cannot identify image file.", False)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")
            logging.error(f"Failed to load image {self.image_path}: {e}")
            self._update_status(f"Error loading image: {e}", False)
        finally:
            if not self.original_pil_image: # If loading failed ensure things are reset
                 self.upscale_button.config(state=tk.DISABLED)


    def start_upscale_thread(self):
        if not self.image_path or not self.original_pil_image:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        try:
            scale_factor = int(self.scale_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid scale factor selected.")
            return

        self._update_status(f"Upscaling image by {scale_factor}x...", True)
        self._set_progress(0)
        self.upscale_button.config(state=tk.DISABLED)
        self.load_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)

        # Run upscaling in a separate thread to keep UI responsive
        # Pass a progress callback to update the progress bar from the worker thread
        thread = threading.Thread(target=self.perform_upscale, args=(scale_factor,))
        thread.daemon = True # Allow main program to exit even if thread is running
        thread.start()

    def perform_upscale(self, scale_factor):
        """Run the upscale operation in a background thread."""
        def progress(percent):
            """Thread-safe progress update."""
            self.root.after(0, self._set_progress, percent)

        try:
            # Pass progress callback so the GUI can show rough progress
            self.upscaled_pil_image = upscaler.upscale_image(
                self.image_path,
                scale_factor,
                progress_callback=progress
            )

            # Schedule GUI updates to be run in the main thread
            if self.upscaled_pil_image:
                self.root.after(0, self.on_upscale_complete, True, scale_factor) # Pass scale_factor
            else:
                self.root.after(0, self.on_upscale_complete, False, scale_factor, "Upscaling process failed or returned no image.")

        except Exception as e:
            logging.error(f"Error during upscaling process: {e}")
            # Ensure GUI update happens in main thread
            self.root.after(0, self.on_upscale_complete, False, scale_factor, f"Error: {e}") # Pass scale_factor

    def on_upscale_complete(self, success, scale_factor, error_message=None): # Accept scale_factor
        self.upscale_button.config(state=tk.NORMAL)
        self.load_button.config(state=tk.NORMAL)

        if success and self.upscaled_pil_image:
            self._display_image(self.upscaled_pil_image, self.upscaled_image_label)

            new_w, new_h = self.upscaled_pil_image.size

            target_dpi = self._get_target_dpi()
            if target_dpi:
                self.upscaled_pil_image = utils.set_image_dpi(
                    self.upscaled_pil_image, target_dpi, target_dpi
                )

            dpi_text = f"DPI: {target_dpi}" if target_dpi else "DPI: N/A"
            self.upscaled_info_label.config(
                text=f"Dimensions: {new_w}×{new_h} | {dpi_text}"
            )
            self.save_button.config(state=tk.NORMAL)
            # Show full progress before hiding the bar
            self._set_progress(100)
            self._update_status("Upscaling complete.", False)
        else:
            final_error_message = error_message or "Upscaling failed."
            messagebox.showerror("Upscale Error", final_error_message)
            self.upscaled_image_label.configure(image='', text="Upscaling failed.")
            self.upscaled_image_label.image = None
            self.upscaled_info_label.config(text="Dimensions: N/A | DPI: N/A")
            # Set to full progress to indicate completion even on failure
            self._set_progress(100)
            self._update_status(f"Upscaling failed: {final_error_message}", False)

        # Ensure progress bar is hidden when done
        if self.progress_bar.winfo_ismapped():
            self.progress_bar.pack_forget()


    def save_image(self):
        if not self.upscaled_pil_image:
            messagebox.showwarning("Warning", "No upscaled image to save.")
            return

        file_types = [
            ('PNG Image', '*.png'),
            ('JPEG Image', '*.jpg *.jpeg'),
            ('TIFF Image', '*.tiff *.tif'),
            ('BMP Image', '*.bmp'),
            ('All Files', '*.*')
        ]
        # Suggest a filename
        original_filename = os.path.basename(self.image_path)
        name, ext = os.path.splitext(original_filename)
        # Use the scale_var directly, which holds the string "2" or "4"
        suggested_filename = f"{name}_upscaled_{self.scale_var.get()}x"

        save_path = filedialog.asksaveasfilename(
            title="Save Upscaled Image As...",
            initialfile=suggested_filename,
            defaultextension=".png", # Default to PNG
            filetypes=file_types
        )

        if not save_path:
            return

        self._update_status(f"Saving image to {os.path.basename(save_path)}...", True)
        self._set_progress(0)
        try:
            # Handle JPEG quality - for simplicity, not adding a slider now, using high quality.
            save_options = {}
            file_ext = os.path.splitext(save_path)[1].lower()

            if file_ext in ['.jpg', '.jpeg']:
                save_options['quality'] = 95  # High quality for JPEG
                save_options['subsampling'] = 0 # Keep chroma subsampling high quality
            elif file_ext == '.png':
                save_options['optimize'] = True
            # TIFF can have compression options, e.g., 'tiff_compression': 'tiff_lzw'

            target_dpi = self._get_target_dpi()
            if target_dpi:
                save_options['dpi'] = (target_dpi, target_dpi)

            # Ensure DPI is in the image info if calculated (done in on_upscale_complete)
            # self.upscaled_pil_image already has DPI set if original was known

            # Writing to disk can take some time on large files; progress is rough
            self._set_progress(50)
            self.upscaled_pil_image.save(save_path, **save_options)
            self._set_progress(100)
            messagebox.showinfo("Success", f"Image saved successfully to:\n{save_path}")
            self._update_status(f"Saved: {os.path.basename(save_path)}", False)
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save image: {e}")
            logging.error(f"Failed to save image to {save_path}: {e}")
            self._update_status(f"Error saving image: {e}", False)
        finally:
            # Hide progress bar when finished
            if self.progress_bar.winfo_ismapped():
                self.progress_bar.pack_forget()


if __name__ == '__main__':
    # This is for testing the GUI directly.
    # In the final application, main.py will run this.
    logging.info("Starting ImageUpscalerApp GUI directly for testing...")

    # Check if models directory exists and if models are present
    models_dir = "models"
    model_x4_path = os.path.join(models_dir, upscaler.MODEL_NAME_X4)

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        messagebox.showinfo("Model Directory Created", f"'{models_dir}' directory was created. Please place Real-ESRGAN models there.")
    elif not os.path.exists(model_x4_path):
         # Try to download if not present for testing convenience
        if messagebox.askyesno("Model Download", f"{upscaler.MODEL_NAME_X4} not found in '{models_dir}'. Download it now for testing? (approx 65MB)"):
            try:
                import requests
                if not os.path.exists(models_dir): os.makedirs(models_dir)
                model_url_x4 = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
                print(f"Downloading {upscaler.MODEL_NAME_X4} to {models_dir}...")
                response = requests.get(model_url_x4, stream=True)
                response.raise_for_status()
                with open(os.path.join(models_dir, upscaler.MODEL_NAME_X4), 'wb') as f: # Use os.path.join here
                    total_length = response.headers.get('content-length')
                    if total_length is None: # no content length header
                        f.write(response.content)
                    else:
                        dl = 0
                        total_length = int(total_length)
                        for data in response.iter_content(chunk_size=4096):
                            dl += len(data)
                            f.write(data)
                            done = int(50 * dl / total_length)
                            print(f"\r[{'=' * done}{' ' * (50-done)}] {dl/1024/1024:.2f}MB / {total_length/1024/1024:.2f}MB", end='')
                print("\nDownload complete.")
                messagebox.showinfo("Download Complete", f"{upscaler.MODEL_NAME_X4} downloaded to '{models_dir}'.")
            except Exception as e:
                messagebox.showerror("Download Failed", f"Could not automatically download {upscaler.MODEL_NAME_X4}. Please download it manually from Real-ESRGAN GitHub releases. Error: {e}")
                # exit() # Or let the app run without models
        else:
            messagebox.showwarning("Model Missing", f"{upscaler.MODEL_NAME_X4} not found in '{models_dir}'. Upscaling will likely fail. Please download models.")


    root_tk = tk.Tk()
    app = ImageUpscalerApp(root_tk)
    root_tk.mainloop()
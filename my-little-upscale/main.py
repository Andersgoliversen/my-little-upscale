import tkinter as tk
import logging
import os
import sys

# Attempt to import gui and upscaler to check for critical components early
try:
    from gui import ImageUpscalerApp
    import upscaler # To trigger its initial checks/setup if any, like model availability warnings
except ModuleNotFoundError as e:
    # This is a critical failure, the application cannot run.
    # Best effort to show a GUI message if tkinter is available.
    print(f"Critical Error: A required module was not found: {e}. Application cannot start.", file=sys.stderr)
    try:
        root_check = tk.Tk()
        root_check.withdraw() # Hide the empty root window
        tk.messagebox.showerror("Startup Error", f"A required module was not found: {e}.\nPlease ensure all files are present and dependencies installed.\n\nCheck console for more details.")
        root_check.destroy()
    except tk.TclError: # In case tkinter itself is not available or display is not found
        pass # Error already printed to stderr
    sys.exit(1)
except ImportError as e:
    print(f"Critical Error: An import failed: {e}. Application cannot start.", file=sys.stderr)
    try:
        root_check = tk.Tk()
        root_check.withdraw()
        tk.messagebox.showerror("Startup Error", f"An import failed: {e}.\nPlease ensure all dependencies are installed correctly.\n\nCheck console for more details.")
        root_check.destroy()
    except tk.TclError:
        pass
    sys.exit(1)


# Configure basic logging for the main application
# This could be more sophisticated, e.g., writing to a file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

def main():
    """
    Main function to initialize and run the Image Upscaler application.
    """
    logging.info("Application started.")

    # Perform pre-flight checks (e.g., model directory and essential models)
    # This provides an early warning if models are missing, complementing gui.py's checks.
    models_dir = "models"
    model_x4_path = os.path.join(models_dir, upscaler.MODEL_NAME_X4) # Using constant from upscaler

    if not os.path.exists(models_dir):
        logging.warning(f"Models directory '{models_dir}' not found. Attempting to create it.")
        try:
            os.makedirs(models_dir)
            logging.info(f"Created models directory: '{models_dir}'.")
            # No models will be present yet, so the next check will trigger.
        except OSError as e:
            logging.error(f"Could not create models directory '{models_dir}': {e}")
            # Proceed, GUI might show further errors or offer download.

    if not os.path.exists(model_x4_path):
        logging.warning(f"Primary model '{upscaler.MODEL_NAME_X4}' not found in '{models_dir}'.")
        # The GUI's __main__ block has a download helper; here we just log.
        # The application can still start, and the GUI can inform the user.
        # Optionally, show a startup message here too, but gui.py handles it well.

    try:
        root = tk.Tk()
        app = ImageUpscalerApp(root)
        root.mainloop()
    except tk.TclError as e:
        logging.critical(f"Tkinter TclError: {e}. This might be due to a missing display (e.g., on a headless server) or other Tkinter initialization issues.")
        print(f"Critical Tkinter Error: {e}. Ensure a display environment is available.", file=sys.stderr)
        # Optionally, show a simple Tkinter error if possible, though if Tk fails, it's hard.
        sys.exit(1)
    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}", exc_info=True)
        # Attempt to show a GUI error message for unexpected errors
        try:
            root_err = tk.Tk()
            root_err.withdraw()
            tk.messagebox.showerror("Unhandled Exception", f"An unexpected error occurred: {e}\n\nPlease check the console output for more details.")
            root_err.destroy()
        except tk.TclError: # If Tkinter itself is failing
            pass # Error logged to console
        sys.exit(1)

    logging.info("Application closed normally.")

if __name__ == "__main__":
    main()

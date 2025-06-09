import os
import re
import logging
from datetime import datetime
from config import settings

# Global variable within this module to store the current run directory
# This is set once by create_run_output_dir when called from main.
_current_run_output_dir = ""

def create_run_output_dir(topic_title="general_run") -> str:
    """
    Creates a unique directory for the current run's output files
    and sets the module-level global variable.
    Returns the path to the created directory.
    """
    global _current_run_output_dir # [30]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # [30]
    # Sanitize topic title for folder name [30]
    sanitized_topic = "".join(c if c.isalnum() or c in (' ', '_') else '' for c in topic_title).rstrip() # [30]
    sanitized_topic = re.sub(r'\s+', '_', sanitized_topic) # [30]

    run_folder_name = f"{sanitized_topic[:50]}_{timestamp}" # [30]
    run_dir = os.path.join(settings.BASE_OUTPUT_DIR, run_folder_name) # [30]

    try:
        os.makedirs(run_dir, exist_ok=True) # [30]
        print(f"Output for this run will be saved in: {run_dir}") # [30]
        _current_run_output_dir = run_dir # Set the module global
        return run_dir # [30]
    except OSError as e: # [31]
        print(f"CRITICAL ERROR: Failed to create unique run output directory {run_dir}: {e}. Exiting.") # [31]
        logging.critical(f"Failed to create unique run output directory {run_dir}: {e}", exc_info=True) # [31]
        exit() # [31]

def get_run_specific_path(filename: str) -> str:
    """Returns the full path for a file within the current run's output directory."""
    if not _current_run_output_dir: # [32]
        # Fallback or error if directory wasn't created - though create_run_output_dir exits on failure
        logging.error("current_run_output_dir not set when trying to get path for filename.") # [32]
        return os.path.join(settings.BASE_OUTPUT_DIR, filename) # [32]

    return os.path.join(_current_run_output_dir, filename) # [32]

def save_text_file(filename: str, content: str):
    """Saves text content to a file within the current run's output directory."""
    file_path = get_run_specific_path(filename)
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        logging.info(f"Saved file: {file_path}")
    except IOError as e:
        logging.error(f"Failed to save file {file_path}: {e}", exc_info=True)
        print(f"ERROR: Failed to save file {filename}: {e}")

def save_json_file(filename: str, data: dict | list):
    """Saves JSON data to a file within the current run's output directory."""
    file_path = get_run_specific_path(filename)
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logging.info(f"Saved JSON file: {file_path}")
    except IOError as e:
        logging.error(f"Failed to save JSON file {file_path}: {e}", exc_info=True)
        print(f"ERROR: Failed to save JSON file {filename}: {e}")
    except TypeError as e:
         logging.error(f"Failed to serialize data to JSON for {file_path}: {e}", exc_info=True)
         print(f"ERROR: Failed to save JSON file {filename} due to data format issue: {e}")


# Add json import needed for save_json_file
import json
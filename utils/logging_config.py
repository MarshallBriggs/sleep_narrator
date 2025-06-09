import logging
import os

def setup_logging(run_output_dir: str, log_file_name: str):
    """Sets up logging to a file within the run-specific output directory."""
    log_file_path = os.path.join(run_output_dir, log_file_name)

    # Remove any existing handlers to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure basic logging
    logging.basicConfig(filename=log_file_path, level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    logging.info(f"Logging initialized. Log file: {log_file_path}")
import os
import json
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import base64

class TTSManager:
    """Manages text-to-speech conversion using Google Cloud TTS API."""
    
    def __init__(self, output_dir: str, config: Dict[str, Any]):
        """
        Initialize the TTS manager.
        
        Args:
            output_dir: Directory where audio files will be saved
            config: Dictionary containing TTS configuration options
        """
        logging.info(f"Initializing TTS manager with output_dir: {output_dir}")
        self.output_dir = Path(output_dir)
        self.audio_output_dir = self.output_dir / "audio_output"
        logging.info(f"Creating audio output directory at: {self.audio_output_dir}")
        self.audio_output_dir.mkdir(exist_ok=True)
        
        # TTS configuration
        self.voice_name = config.get("voice_name", "en-US-Chirp3-HD-Enceladus")
        self.language_code = config.get("language_code", "en-US")
        self.speaking_rate = config.get("speaking_rate", 0.9)
        self.audio_encoding = config.get("audio_encoding", "LINEAR16")
        
        # Ensure Google Cloud credentials are set
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not creds_path:
            logging.error("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable not set. Please set it to the path of your service account key file.")
        logging.info(f"Found GOOGLE_APPLICATION_CREDENTIALS at: {creds_path}")
        if not os.path.exists(creds_path):
            logging.error(f"Google Cloud credentials file not found at: {creds_path}")
            raise ValueError(f"Google Cloud credentials file not found at: {creds_path}")
    
    def _create_tts_request(self, text: str) -> Dict[str, Any]:
        """Create the TTS API request payload."""
        return {
            "input": {
                "markup": text
            },
            "voice": {
                "languageCode": self.language_code,
                "name": self.voice_name,
                "voiceClone": {}
            },
            "audioConfig": {
                "audioEncoding": self.audio_encoding,
                "speakingRate": self.speaking_rate
            }
        }
    
    def _get_auth_token(self) -> str:
        """Get Google Cloud authentication token."""
        try:
            logging.info("Attempting to get Google Cloud auth token using gcloud CLI")
            # Use full path to gcloud.cmd on Windows
            gcloud_path = r"C:\Users\Marshall\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
            gcloud_cmd = [gcloud_path, "auth", "print-access-token"]
            logging.debug(f"Running command: {' '.join(gcloud_cmd)}")
            result = subprocess.run(
                gcloud_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            logging.info("Successfully obtained auth token")
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to get auth token. Command output: {e.stdout}\nError: {e.stderr}")
            raise
        except FileNotFoundError:
            logging.error("gcloud command not found. Is Google Cloud SDK installed and in PATH?")
            raise
    
    def _get_project_id(self) -> str:
        """Get Google Cloud project ID."""
        try:
            logging.info("Attempting to get Google Cloud project ID using gcloud CLI")
            # Use full path to gcloud.cmd on Windows
            gcloud_path = r"C:\Users\Marshall\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
            gcloud_cmd = [gcloud_path, "config", "list", "--format=value(core.project)"]
            logging.debug(f"Running command: {' '.join(gcloud_cmd)}")
            result = subprocess.run(
                gcloud_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            project_id = result.stdout.strip()
            logging.info(f"Successfully obtained project ID: {project_id}")
            return project_id
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to get project ID. Command output: {e.stdout}\nError: {e.stderr}")
            raise
        except FileNotFoundError:
            logging.error("gcloud command not found. Is Google Cloud SDK installed and in PATH?")
            raise
    
    def convert_text_to_speech(self, text: str, output_filename: str) -> Optional[str]:
        """
        Convert text to speech and save as audio file.
        
        Args:
            text: Text to convert to speech
            output_filename: Name of the output audio file (without extension)
            
        Returns:
            Path to the generated audio file if successful, None otherwise
        """
        try:
            # Prepare the request
            request_data = self._create_tts_request(text)
            auth_token = self._get_auth_token()
            project_id = self._get_project_id()
            
            # Prepare curl command
            curl_cmd = [
                "curl", "-X", "POST",
                "-H", "Content-Type: application/json",
                "-H", f"X-Goog-User-Project: {project_id}",
                "-H", f"Authorization: Bearer {auth_token}",
                "--data", json.dumps(request_data),
                "https://texttospeech.googleapis.com/v1/text:synthesize"
            ]
            
            # Execute curl command
            result = subprocess.run(
                curl_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse response and save audio
            response = json.loads(result.stdout)
            if "audioContent" not in response:
                logging.error("No audio content in response")
                return None
            
            # Save the audio file
            output_path = self.audio_output_dir / f"{output_filename}.wav"
            with open(output_path, "wb") as f:
                f.write(base64.b64decode(response["audioContent"]))
            
            logging.info(f"Successfully generated audio file: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logging.error(f"Failed to convert text to speech: {e}")
            return None
    
    def process_script_sections(self, script_files: list[Path]) -> Dict[str, str]:
        """
        Process multiple script sections and convert them to audio.
        
        Args:
            script_files: List of paths to script section files
            
        Returns:
            Dictionary mapping script filenames to their corresponding audio file paths
        """
        results = {}
        logging.info(f"Starting to process {len(script_files)} script files")
        for script_file in script_files:
            try:
                logging.info(f"Processing script file: {script_file}")
                if not script_file.exists():
                    logging.error(f"Script file does not exist: {script_file}")
                    continue
                    
                # Read script content
                logging.info(f"Reading content from: {script_file}")
                with open(script_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                logging.info(f"Successfully read {len(content)} characters from {script_file}")
                
                # Generate audio filename from script filename
                audio_filename = script_file.stem
                logging.info(f"Will save audio as: {audio_filename}.wav")
                
                # Convert to speech
                audio_path = self.convert_text_to_speech(content, audio_filename)
                if audio_path:
                    logging.info(f"Successfully generated audio at: {audio_path}")
                    results[str(script_file)] = audio_path
                else:
                    logging.error(f"Failed to generate audio for: {script_file}")
                
            except Exception as e:
                logging.error(f"Failed to process script file {script_file}: {str(e)}", exc_info=True)
                continue
        
        logging.info(f"Completed processing {len(script_files)} files. Generated {len(results)} audio files.")
        return results 
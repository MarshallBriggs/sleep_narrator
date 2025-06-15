import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import base64
import time
import re
import tempfile
import wave
from google.cloud import texttospeech
from google.api_core import retry
from google.api_core.exceptions import ServiceUnavailable, InternalServerError
import shutil
from config.settings import (
    DEFAULT_TTS_CONFIG,
    TTS_MIN_SPEAKING_RATE,
    TTS_MAX_SPEAKING_RATE,
    TTS_CHUNK_SIZE_BYTES,
    TTS_RETRY_INITIAL_DELAY,
    TTS_RETRY_MAX_DELAY,
    TTS_RETRY_MULTIPLIER,
    TTS_RETRY_DEADLINE,
)

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
        
        # TTS configuration - use provided config or fall back to defaults
        self.voice_name = config.get("voice_name", DEFAULT_TTS_CONFIG["voice_name"])
        self.language_code = config.get("language_code", DEFAULT_TTS_CONFIG["language_code"])
        self.speaking_rate = config.get("speaking_rate", DEFAULT_TTS_CONFIG["speaking_rate"])
        self.audio_encoding = config.get("audio_encoding", DEFAULT_TTS_CONFIG["audio_encoding"])
        
        # Initialize Google Cloud TTS client
        try:
            self.client = texttospeech.TextToSpeechClient()
            logging.info("Successfully initialized Google Cloud TTS client")
        except Exception as e:
            logging.error(f"Failed to initialize Google Cloud TTS client: {e}")
            raise

        # Track temp files for cleanup
        self._temp_files: List[str] = []

    def __del__(self):
        """Cleanup temp files on object destruction."""
        self._cleanup_temp_files()

    def _cleanup_temp_files(self):
        """Clean up any remaining temp files."""
        for temp_path in self._temp_files:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                logging.debug(f"Failed to remove temp file {temp_path}: {e}")

    def _create_tts_request(self, text: str) -> texttospeech.SynthesisInput:
        """Create the TTS API request."""
        return texttospeech.SynthesisInput(text=text)

    def _get_voice(self) -> texttospeech.VoiceSelectionParams:
        """Get voice selection parameters."""
        return texttospeech.VoiceSelectionParams(
            language_code=self.language_code,
            name=self.voice_name
        )

    def _get_audio_config(self) -> texttospeech.AudioConfig:
        """Get audio configuration parameters."""
        return texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding[self.audio_encoding],
            speaking_rate=self.speaking_rate
        )

    def _should_retry(self, exc: Exception) -> bool:
        """Determine if an exception should trigger a retry."""
        if isinstance(exc, (ServiceUnavailable, InternalServerError)):
            return True
        if isinstance(exc, Exception) and any(err in str(exc) for err in ['502', '503', '504']):
            return True
        return False

    @retry.Retry(
        predicate=_should_retry,
        initial=TTS_RETRY_INITIAL_DELAY,
        maximum=TTS_RETRY_MAX_DELAY,
        multiplier=TTS_RETRY_MULTIPLIER,
        deadline=TTS_RETRY_DEADLINE,
        on_retry=lambda retry_state: logging.warning(
            f"Retry {retry_state.attempt}/∞ after {retry_state.retry_delay:.0f}s"
        )
    )
    def _synthesize_speech(self, text: str) -> bytes:
        """Synthesize speech with retry logic for temporary errors."""
        try:
            response = self.client.synthesize_speech(
                input=self._create_tts_request(text),
                voice=self._get_voice(),
                audio_config=self._get_audio_config()
            )
            return response.audio_content
        except Exception as e:
            logging.error(f"TTS synthesis failed: {e}")
            raise

    def _split_text_to_chunks(self, text: str, max_bytes: int = TTS_CHUNK_SIZE_BYTES) -> List[str]:
        """
        Split text into smaller chunks for TTS processing.
        Uses a smaller max_bytes value to reduce likelihood of timeouts.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = ''
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # If sentence itself is too long, forcibly split
            if len(sentence.encode('utf-8')) > max_bytes:
                logging.warning(f"A single sentence exceeds {max_bytes} bytes. Forcibly splitting.")
                # Split by words
                words = sentence.split()
                forced = ''
                for word in words:
                    if len((forced + ' ' + word).encode('utf-8')) > max_bytes:
                        if forced:
                            chunks.append(forced.strip())
                        forced = word
                    else:
                        forced = (forced + ' ' + word).strip()
                if forced:
                    chunks.append(forced.strip())
                continue
                
            if len((current + ' ' + sentence).encode('utf-8')) > max_bytes:
                if current:
                    chunks.append(current.strip())
                current = sentence
            else:
                current = (current + ' ' + sentence).strip()
                
        if current:
            chunks.append(current.strip())
            
        return chunks

    def _process_chunk(self, chunk: str, chunk_idx: int, total_chunks: int) -> Optional[str]:
        """Process a single chunk of text and return the path to the temp audio file."""
        chunk_bytes = len(chunk.encode('utf-8'))
        logging.info(f"Processing chunk {chunk_idx+1}/{total_chunks} (size: {chunk_bytes} bytes)")
        logging.debug(f"Chunk {chunk_idx+1} preview: {chunk[:100]!r}")
        
        try:
            audio_content = self._synthesize_speech(chunk)
            if not audio_content:
                logging.error(f"No audio content received for chunk {chunk_idx+1}")
                return None

            temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
            os.close(temp_fd)  # Close the file descriptor
            with open(temp_path, "wb") as f:
                f.write(audio_content)
            self._temp_files.append(temp_path)
            logging.info(f"Chunk {chunk_idx+1} audio saved to temp file: {temp_path}")
            return temp_path
            
        except Exception as e:
            logging.error(f"Exception during TTS processing for chunk {chunk_idx+1}: {e}")
            return None

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
            text_bytes = len(text.encode('utf-8'))
            if text_bytes > TTS_CHUNK_SIZE_BYTES:
                logging.info(f"Text is {text_bytes} bytes, chunking required.")
                text_chunks = self._split_text_to_chunks(text)
            else:
                text_chunks = [text]

            temp_files = []
            failed_chunks = []
            
            # Process all chunks
            for idx, chunk in enumerate(text_chunks):
                temp_path = self._process_chunk(chunk, idx, len(text_chunks))
                if temp_path:
                    temp_files.append(temp_path)
                else:
                    failed_chunks.append((idx, chunk))

            # Try failed chunks one more time after a delay
            if failed_chunks:
                logging.info(f"Retrying {len(failed_chunks)} failed chunks after delay...")
                time.sleep(30)  # Wait 30 seconds before retrying
                for idx, chunk in failed_chunks:
                    temp_path = self._process_chunk(chunk, idx, len(text_chunks))
                    if temp_path:
                        temp_files.append(temp_path)

            if not temp_files:
                logging.error(f"No audio files were generated for {output_filename}.wav. See previous errors.")
                return None

            output_path = self.audio_output_dir / f"{output_filename}.wav"
            try:
                with wave.open(str(output_path), 'wb') as out_wav:
                    for i, temp_path in enumerate(temp_files):
                        with wave.open(temp_path, 'rb') as in_wav:
                            if i == 0:
                                out_wav.setparams(in_wav.getparams())
                            out_wav.writeframes(in_wav.readframes(in_wav.getnframes()))
                logging.info(f"Successfully generated audio file: {output_path}")
            except Exception as e:
                logging.error(f"Failed to concatenate audio chunks for {output_filename}.wav: {e}")
                return None

            if len(temp_files) < len(text_chunks):
                logging.warning(f"Some chunks failed for {output_filename}.wav. Audio may be incomplete.")
            return str(output_path)

        except Exception as e:
            logging.error(f"Failed to convert text to speech for {output_filename}: {e}")
            return None
        finally:
            self._cleanup_temp_files()

    def process_script_sections(self, script_files: list[Path]) -> Dict[str, str]:
        """
        Process multiple script sections and convert them to audio.
        
        Args:
            script_files: List of paths to script section files
            
        Returns:
            Dictionary mapping script filenames to their corresponding audio file paths
        """
        # Sort script files by name to maintain order
        script_files = sorted(script_files)
        
        results = {}
        total_files = len(script_files)
        logging.info(f"Starting to process {total_files} script files")
        print(f"Starting TTS processing for {total_files} files...")
        for idx, script_file in enumerate(script_files, 1):
            try:
                progress_msg = f"Processing {idx}/{total_files}: {script_file.name}"
                print(progress_msg)
                logging.info(progress_msg)
                if not script_file.exists():
                    logging.error(f"Script file does not exist: {script_file}")
                    continue
                    
                # Read script content
                logging.info(f"Reading content from: {script_file}")
                with open(script_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                logging.info(f"Successfully read {len(content)} characters from {script_file}")
                
                # Generate audio filename from script filename, preserving the numerical prefix
                audio_filename = script_file.stem  # This will keep the numerical prefix since it's part of the filename
                logging.info(f"Will save audio as: {audio_filename}.wav")
                
                # Convert to speech with timing
                start_time = time.time()
                audio_path = self.convert_text_to_speech(content, audio_filename)
                end_time = time.time()
                elapsed = end_time - start_time
                if audio_path:
                    success_msg = f"Successfully generated audio at: {audio_path} (Time: {elapsed:.2f}s)"
                    print(success_msg)
                    logging.info(success_msg)
                    results[str(script_file)] = audio_path
                else:
                    fail_msg = f"Failed to generate audio for: {script_file} (Time: {elapsed:.2f}s)"
                    print(fail_msg)
                    logging.error(fail_msg)
                
            except Exception as e:
                logging.error(f"Failed to process script file {script_file}: {str(e)}", exc_info=True)
                print(f"Exception while processing {script_file}: {e}")
                continue
        summary_msg = f"Completed processing {total_files} files. Generated {len(results)} audio files."
        print(summary_msg)
        logging.info(summary_msg)
        return results 
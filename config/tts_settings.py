"""Configuration settings for Text-to-Speech functionality."""

# Default TTS configuration
DEFAULT_TTS_CONFIG = {
    "voice_name": "en-US-Chirp3-HD-Enceladus",  # Google Cloud TTS voice name
    "language_code": "en-US",                   # Language code
    "speaking_rate": 0.9,                       # Speaking rate (0.25 to 4.0)
    "audio_encoding": "LINEAR16",               # Audio encoding format
}

# Available voice options (for reference)
AVAILABLE_VOICES = {
    "en-US": [
        "en-US-Chirp3-HD-Enceladus",  # Default
        "en-US-Neural2-A",
        "en-US-Neural2-C",
        "en-US-Neural2-D",
        "en-US-Neural2-E",
        "en-US-Neural2-F",
        "en-US-Neural2-G",
        "en-US-Neural2-H",
        "en-US-Neural2-I",
        "en-US-Neural2-J",
    ]
}

# Audio encoding options
AUDIO_ENCODING_OPTIONS = [
    "LINEAR16",    # Uncompressed 16-bit signed little-endian samples
    "MP3",         # MP3 audio
    "OGG_OPUS",    # Ogg Opus audio
    "MULAW",       # 8-bit samples that compand 14-bit audio samples using μ-law
    "ALAW",        # 8-bit samples that compand 14-bit audio samples using A-law
]

# Speaking rate range
MIN_SPEAKING_RATE = 0.25
MAX_SPEAKING_RATE = 4.0 
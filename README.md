# Sleep Narrator

An AI-powered script generation and narration system that creates engaging, well-researched audio content. Perfect for creating sleep stories, educational content, or narrative podcasts.

## Features

- 🤖 **AI-Powered Research**: Automatically gathers and synthesizes comprehensive research on any topic
- 📝 **Intelligent Script Generation**: Creates well-structured, engaging narratives with natural flow
- 🎯 **Section-Based Architecture**: Breaks content into logical sections with proper pacing
- 🎙️ **High-Quality TTS**: Converts scripts to natural-sounding speech using Google Cloud TTS
- 📊 **Length Control**: Maintains precise control over total content duration
- 🔄 **Interactive Workflow**: User-friendly process with confirmation steps and feedback loops
- 📁 **Organized Output**: Automatically organizes and numbers all generated content

## Prerequisites

- Python 3.8 or higher
- Google Cloud account with Text-to-Speech API enabled
- Google Cloud credentials (service account key)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MarshallBriggs/sleep_narrator.git
cd sleep_narrator
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up Google Cloud credentials:
   - Create a service account in Google Cloud Console
   - Download the JSON key file
   - Set the environment variable:
     ```bash
     # On Windows:
     set GOOGLE_APPLICATION_CREDENTIALS=path/to/your/credentials.json
     # On Unix or MacOS:
     export GOOGLE_APPLICATION_CREDENTIALS=path/to/your/credentials.json
     ```

## Usage

1. Run the main script:
```bash
python main.py
```

2. Follow the interactive prompts:
   - Enter your topic
   - Specify target duration
   - Review and confirm section structure
   - Choose whether to generate audio

3. Output files will be organized in the `output` directory:
   - Each run creates a timestamped folder
   - Script sections are numbered (01_, 02_, etc.)
   - Audio files maintain the same numbering
   - Final combined script is included

## Output Structure

```
output/
└── your_topic_YYYYMMDD_HHMMSS/
    ├── 01_script_section_introduction.txt
    ├── 01_script_section_introduction.wav
    ├── 02_script_section_main_content.txt
    ├── 02_script_section_main_content.wav
    ├── ...
    ├── final_video_script.txt
    └── application.log
```

## Configuration

Key settings can be modified in `config/settings.py`:
- TTS voice selection
- Speaking rate
- Audio encoding format
- Script generation parameters
- Research and context limits

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Acknowledgments

- Google Cloud Text-to-Speech API for high-quality voice synthesis
- Gemini AI for intelligent content generation
- All contributors and users of this project

## Support

If you encounter any issues or have questions, please:
1. Check the [Issues](https://github.com/MarshallBriggs/sleep_narrator/issues) page
2. Review the application logs in the output directory
3. Open a new issue if needed

---

Made with ❤️ for content creators and storytellers 
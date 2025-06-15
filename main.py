import time
import os
import json
import logging
import re  # For filename sanitization
import argparse  # For command line argument parsing
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import GoogleSearchRetrieval
from google.generativeai.types import Tool
from pathlib import Path

# Import modules from your project structure
from config import settings
from utils import file_utils, logging_config, estimation_utils
from ui import cli
from api import gemini_client
from logic import research, structuring, generation, stitching
from tts.tts_manager import TTSManager

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Video Script Generator v2.8')
    parser.add_argument('--input-file', '-i', type=str, help='Path to a text file containing input parameters')
    parser.add_argument('--no-tts', action='store_true', help='Skip TTS processing')
    return parser.parse_args()

def process_tts(run_output_dir: str, script_files: list[Path]) -> None:
    """
    Process text-to-speech conversion for script sections.
    
    Args:
        run_output_dir: Directory where the run output is stored
        script_files: List of script section files to process
    """
    try:
        # Initialize TTS manager with default config
        tts_manager = TTSManager(run_output_dir, settings.DEFAULT_TTS_CONFIG)
        
        # Process all script sections
        results = tts_manager.process_script_sections(script_files)
        
        # Log results
        if results:
            logging.info(f"Successfully generated {len(results)} audio files")
            for script_file, audio_file in results.items():
                logging.info(f"Generated audio for {script_file}: {audio_file}")
        else:
            logging.warning("No audio files were generated")
            
    except Exception as e:
        logging.error(f"TTS processing failed: {e}")
        print(f"Error during TTS processing: {e}")

def main():
    overall_start_time = time.time()

    # Parse command line arguments
    args = parse_arguments()

    # Get initial user inputs via CLI or file
    user_topic_direction, total_target_minutes, research_influence, raw_topic_title, run_tts = cli.get_initial_user_inputs(args.input_file)

    if not user_topic_direction or total_target_minutes is None or research_influence is None: 
        logging.warning("No valid user inputs.")
        print("No valid user inputs. Exiting.")
        return

    # Create run output directory
    run_output_dir = file_utils.create_run_output_dir(raw_topic_title if raw_topic_title else "Unnamed_Topic")

    # Setup logging to the run directory
    logging_config.setup_logging(run_output_dir, settings.LOG_FILE_NAME)

    logging.info(f"Application v2.8 started. Output directory: {run_output_dir}")
    print(f"--- Automated Long-Form Narrative Script Generator v2.8 (Text Only) ---")

    # Initialize Gemini models
    gemini_model_research = None
    gemini_model_script_narrator = None
    gemini_model_structurer = None

    try:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logging.critical("GOOGLE_API_KEY not set. Please set it in your .env file or environment variables.")
            print("CRITICAL ERROR: GOOGLE_API_KEY not set. Exiting.")
            print("Please ensure you have a .env file in the project_root directory")
            print("with the line: GOOGLE_API_KEY=YOUR_API_KEY_HERE")
            print("replacing YOUR_API_KEY_HERE with your actual key.")
            return

        genai.configure(api_key=api_key)

        # Initialize models with their respective roles and tools
        gsr_tool_config = GoogleSearchRetrieval()
        google_search_tool = Tool(google_search_retrieval=gsr_tool_config)
        gemini_model_research = genai.GenerativeModel(settings.GEMINI_MODEL_NAME, tools=[google_search_tool])
        logging.info(f"Gemini research model initialized: {settings.GEMINI_MODEL_NAME} with Search.")
        print(f"Gemini research model initialized: {settings.GEMINI_MODEL_NAME} with Search.")

        gemini_model_script_narrator = genai.GenerativeModel(
            model_name=settings.GEMINI_MODEL_NAME,
            system_instruction=settings.GEMINI_SYSTEM_INSTRUCTION_NARRATOR
        )
        logging.info(f"Gemini script narrator model initialized: {settings.GEMINI_MODEL_NAME}.")
        print(f"Gemini script narrator model initialized: {settings.GEMINI_MODEL_NAME}.")

        gemini_model_structurer = genai.GenerativeModel(
            model_name=settings.GEMINI_MODEL_NAME,
            system_instruction=settings.GEMINI_SYSTEM_INSTRUCTION_STRUCTURER
        )
        logging.info(f"Gemini structuring model initialized: {settings.GEMINI_MODEL_NAME}.")
        print(f"Gemini structuring model initialized: {settings.GEMINI_MODEL_NAME}.")

    except Exception as e:
        logging.critical(f"Failed to initialize Gemini client(s): {e}", exc_info=True)
        print(f"CRITICAL ERROR: Gemini client init failed: {e}. Exiting.")
        return

    if not all([gemini_model_research, gemini_model_script_narrator, gemini_model_structurer]):
        logging.critical("One or more Gemini models failed to init.")
        print("CRITICAL ERROR: Gemini model init failed. Exiting.")
        return

    # --- Phase 1: Global Research ---
    global_research_content = research.perform_global_research(
        gemini_model_research,
        user_topic_direction,
        total_target_minutes
    )

    if not global_research_content:
        logging.error("Global research failed.")
        print("Pipeline halted: Global research failed.")
        return

    # --- Phase 2: Section Structuring & User Feedback ---
    confirmed_sections = None
    proposed_sections_list = structuring.propose_section_structure(
        gemini_model_structurer,
        global_research_content,
        user_topic_direction,
        total_target_minutes
    )

    if not proposed_sections_list:
        logging.error("Initial section proposal failed.")
        print("Pipeline halted: Initial section proposal failed.")
        return

    # Loop for user feedback and retooling
    for i in range(settings.MAX_ITERATIVE_EXPANSION_ATTEMPTS + 3):
        user_feedback_str, current_proposal_for_retooling = cli.get_user_feedback_on_sections(
            proposed_sections_list,
            total_target_minutes
        )

        if not user_feedback_str: # User confirmed
            confirmed_sections = current_proposal_for_retooling
            logging.info(f"User confirmed section structure: {json.dumps(confirmed_sections, indent=2)}")
            print("Section structure confirmed by user.")
            # Save confirmed sections
            file_utils.save_json_file("confirmed_sections.json", confirmed_sections)
            break # Exit feedback loop

        # User provided feedback, attempt retooling (allow MAX_ITERATIVE_EXPANSION_ATTEMPTS + 2 retool attempts)
        if i < settings.MAX_ITERATIVE_EXPANSION_ATTEMPTS + 2 : # Check if retool attempts remain
             original_proposal_str_for_retool = json.dumps(current_proposal_for_retooling, indent=2)
             retooled_list = structuring.retool_section_structure(
                 gemini_model_structurer,
                 original_proposal_str_for_retool,
                 user_feedback_str,
                 global_research_content,
                 total_target_minutes
             )
             if retooled_list:
                 proposed_sections_list = retooled_list # Update proposal for next iteration
             else: # Retooling failed
                 print("AI failed to retool sections based on feedback. Displaying previous proposal for confirmation or further feedback.")
                 # Keep proposed_sections_list as the previous valid one

        else: # Max retool attempts reached
            logging.warning("Max attempts for section retooling reached. Using last proposed structure.")
            print("Max attempts for retooling sections reached. Please confirm the last proposed structure if acceptable, or restart.")
            confirmed_sections = current_proposal_for_retooling # Use the last proposal
            # Save the last proposal
            file_utils.save_json_file("confirmed_sections_lapsed.json", confirmed_sections)
            break # Exit feedback loop


    if not confirmed_sections: # If feedback loop finished without confirmation
        logging.error("Section structuring could not be confirmed.")
        print("Pipeline halted: Section structure not confirmed.")
        return

    # --- Phase 3: Script Generation for Each Section ---
    generated_section_scripts_map = {} # Dict to store scripts by title
    final_section_order = [sec['title'] for sec in confirmed_sections] # List of titles for ordering

    for section_info in confirmed_sections: # Iterate through confirmed sections
        title = section_info.get('title', 'Untitled Section')
        description = section_info.get('description', '')
        section_target_mins = section_info.get('estimated_minutes', 5) # Default to 5 min

        section_script = generation.generate_single_section_script(
            gemini_model_script_narrator,
            title,
            description,
            section_target_mins,
            global_research_content,
            user_topic_direction,
            research_influence
        )

        if not section_script: # If section generation fails
            logging.error(f"Failed to generate script for section: {title}. Halting.")
            print(f"ERROR: Failed to generate script for section: {title}. Halting.")
            return # Halt pipeline

        generated_section_scripts_map[title] = section_script # Store generated script

        # Save individual section script
        # Sanitize title for filename
        section_filename = f"script_section_{re.sub(r'[^a-zA-Z0-9_]+', '', title.replace(' ','_'))[:50]}.txt"
        file_utils.save_text_file(section_filename, section_script)

    # --- Phase 4: Stitching and Smoothing ---
    final_script_output_path = file_utils.get_run_specific_path("final_video_script.txt")

    final_script_content = stitching.stitch_and_smooth_script(
        gemini_model_script_narrator,
        generated_section_scripts_map,
        final_section_order,
        user_topic_direction,
        total_target_minutes
    )

    if not final_script_content: # If stitching/smoothing fails
        logging.error("Final script stitching/smoothing failed.")
        print("Pipeline halted: Final script generation failed.")
        return

    # Save the final script
    file_utils.save_text_file("final_video_script.txt", final_script_content)
    logging.info(f"Final polished script saved to {final_script_output_path}")
    print(f"Final polished script saved to {final_script_output_path}")

    # After script generation and stitching is complete, handle TTS processing
    if not args.no_tts:
        if run_tts is None:  # Interactive mode - ask user
            print("\nWould you like to convert the generated scripts to speech? (yes/no)")
            tts_choice = input().lower().strip()
            run_tts = tts_choice in ['yes', 'y']
        
        if run_tts:
            tts_start_time = time.time()
            # Find all script section files
            script_files = list(Path(run_output_dir).glob("script_section_*.txt"))
            logging.info(f"TTS processing list - {script_files}")
            print(f"\nTTS processing list - {script_files}")
            if script_files:
                print("\nStarting text-to-speech conversion...")
                process_tts(run_output_dir, script_files)
                print("TTS processing complete. Audio files are in the 'audio_output' directory.")
            else:
                print("No script section files found for TTS processing.")

            # Log completion
            tts_end_time = time.time()
            tts_total_time = tts_end_time - tts_start_time
            logging.info(f"TTS processing completed in {tts_total_time:.2f} seconds")
            print(f"\nTTS processing completed in {tts_total_time:.2f} seconds")
        else:
            print("Skipping TTS processing.")

    # --- Pipeline Complete - Final Reporting ---
    overall_end_time = time.time()
    total_execution_time = overall_end_time - overall_start_time

    logging.info(f"Application v2.8 finished. Total execution time: {total_execution_time:.2f} seconds.")
    print(f"\n--- Pipeline Complete (v2.8) ---")

    if final_script_content: # If a script was generated
        final_estimated_length = estimation_utils.estimate_script_length_minutes(final_script_content)
        print(f"Final script generated at: {final_script_output_path} (Estimated length: {final_estimated_length:.2f} minutes)")

        # Attempt to count tokens for the final script
        try:
            final_script_token_count_response = gemini_model_script_narrator.count_tokens(final_script_content)
            final_script_token_count = final_script_token_count_response.total_tokens
            print(f"Token count for the final script content: {final_script_token_count}")
            logging.info(f"Token count for the final script content: {final_script_token_count}")
        except Exception as e:
            logging.error(f"Could not count tokens for final script: {e}", exc_info=True)
            print(f"Could not count tokens for final script: {e}")
    else: # If no script was generated
        print(f"Script generation failed. Check logs at: {settings.LOG_FILE_NAME}")


    # Report total token usage and paths
    token_usage = gemini_client.get_token_usage()
    print(f"Global research results: {file_utils.get_run_specific_path('research_results.txt')}")
    print(f"Logs available at: {file_utils.get_run_specific_path(settings.LOG_FILE_NAME)}") # Use constant from settings
    print(f"All outputs for this run are in: {file_utils._current_run_output_dir}") # Access module global from file_utils
    print(f"Total accumulated Prompt Tokens: {token_usage['prompt_tokens']}")
    print(f"Total accumulated Candidates Tokens: {token_usage['candidates_tokens']}")
    print(f"Total accumulated Tokens (sum of all calls): {token_usage['total_tokens']}")
    print(f"Total execution time: {total_execution_time:.2f} seconds.")


if __name__ == "__main__":
    print("Starting Video Script Generator v2.8 (Text Only)...")
    main()
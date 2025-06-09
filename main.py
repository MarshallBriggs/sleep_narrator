import time
import os
import json
import logging
import re  # For filename sanitization
import argparse  # For command line argument parsing
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import GoogleSearchRetrieval
from google.generativeai.types import Tool

# Import modules from your project structure
from config import settings
from utils import file_utils, logging_config, estimation_utils
from ui import cli
from api import gemini_client
from logic import research, structuring, generation, stitching

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Video Script Generator v2.8')
    parser.add_argument('--input-file', '-i', type=str, help='Path to a text file containing input parameters')
    return parser.parse_args()

def main():
    overall_start_time = time.time() # [98]

    # Parse command line arguments
    args = parse_arguments()

    # Get initial user inputs via CLI or file [99]
    user_topic_direction, total_target_minutes, research_influence, raw_topic_title = cli.get_initial_user_inputs(args.input_file) # [99]

    if not user_topic_direction or total_target_minutes is None or research_influence is None: # [99]
        logging.warning("No valid user inputs.") # [99]
        print("No valid user inputs. Exiting.") # [99]
        return # [99]

    # Create run output directory [99]
    run_output_dir = file_utils.create_run_output_dir(raw_topic_title if raw_topic_title else "Unnamed_Topic") # [99]

    # Setup logging to the run directory [99]
    logging_config.setup_logging(run_output_dir, settings.LOG_FILE_NAME) # [99]

    logging.info(f"Application v2.8 started. Output directory: {run_output_dir}") # [99]
    print(f"--- Automated Long-Form Narrative Script Generator v2.8 (Text Only) ---") # [99]

    # Initialize Gemini models [100-102]
    gemini_model_research = None
    gemini_model_script_narrator = None
    gemini_model_structurer = None

    try: # [100]
        api_key = os.environ.get("GOOGLE_API_KEY") # [100]
        if not api_key: # [100]
            logging.critical("GOOGLE_API_KEY not set. Please set it in your .env file or environment variables.") # [100]
            print("CRITICAL ERROR: GOOGLE_API_KEY not set. Exiting.") # [100]
            # Add specific instructions
            print("Please ensure you have a .env file in the project_root directory")
            print("with the line: GOOGLE_API_KEY=YOUR_API_KEY_HERE")
            print("replacing YOUR_API_KEY_HERE with your actual key.")
            return # [100]

        genai.configure(api_key=api_key) # [100]

        # Initialize models with their respective roles and tools [100, 101]
        gsr_tool_config = GoogleSearchRetrieval() # [100]
        google_search_tool = Tool(google_search_retrieval=gsr_tool_config) # [100]
        gemini_model_research = genai.GenerativeModel(settings.GEMINI_MODEL_NAME, tools=[google_search_tool]) # [100]
        logging.info(f"Gemini research model initialized: {settings.GEMINI_MODEL_NAME} with Search.") # [100]
        print(f"Gemini research model initialized: {settings.GEMINI_MODEL_NAME} with Search.") # [100]

        gemini_model_script_narrator = genai.GenerativeModel( # [101]
            model_name=settings.GEMINI_MODEL_NAME,
            system_instruction=settings.GEMINI_SYSTEM_INSTRUCTION_NARRATOR
        )
        logging.info(f"Gemini script narrator model initialized: {settings.GEMINI_MODEL_NAME}.") # [101]
        print(f"Gemini script narrator model initialized: {settings.GEMINI_MODEL_NAME}.") # [101]

        gemini_model_structurer = genai.GenerativeModel( # [101]
            model_name=settings.GEMINI_MODEL_NAME,
            system_instruction=settings.GEMINI_SYSTEM_INSTRUCTION_STRUCTURER
        )
        logging.info(f"Gemini structuring model initialized: {settings.GEMINI_MODEL_NAME}.") # [102]
        print(f"Gemini structuring model initialized: {settings.GEMINI_MODEL_NAME}.") # [102]

    except Exception as e: # [102]
        logging.critical(f"Failed to initialize Gemini client(s): {e}", exc_info=True) # [102]
        print(f"CRITICAL ERROR: Gemini client init failed: {e}. Exiting.") # [102]
        return # [102]

    if not all([gemini_model_research, gemini_model_script_narrator, gemini_model_structurer]): # Final check [102]
        logging.critical("One or more Gemini models failed to init.") # [102]
        print("CRITICAL ERROR: Gemini model init failed. Exiting.") # [102]
        return # [102]

    # --- Phase 1: Global Research --- [102, 103]
    global_research_content = research.perform_global_research(
        gemini_model_research,
        user_topic_direction,
        total_target_minutes
    ) # [103]

    if not global_research_content: # [103]
        logging.error("Global research failed.") # [103]
        print("Pipeline halted: Global research failed.") # [103]
        return # [103]

    # --- Phase 2: Section Structuring & User Feedback --- [103-106]
    confirmed_sections = None
    proposed_sections_list = structuring.propose_section_structure(
        gemini_model_structurer,
        global_research_content,
        user_topic_direction,
        total_target_minutes
    ) # [103]

    if not proposed_sections_list: # [103]
        logging.error("Initial section proposal failed.") # [103]
        print("Pipeline halted: Initial section proposal failed.") # [103]
        return # [103]

    # Loop for user feedback and retooling [104]
    for i in range(settings.MAX_ITERATIVE_EXPANSION_ATTEMPTS + 3): # Use settings constant [104]
        user_feedback_str, current_proposal_for_retooling = cli.get_user_feedback_on_sections(
            proposed_sections_list,
            total_target_minutes
        ) # [104]

        if not user_feedback_str: # User confirmed [104]
            confirmed_sections = current_proposal_for_retooling # [104]
            logging.info(f"User confirmed section structure: {json.dumps(confirmed_sections, indent=2)}") # [104]
            print("Section structure confirmed by user.") # [104]
            # Save confirmed sections [104]
            file_utils.save_json_file("confirmed_sections.json", confirmed_sections) # [104]
            break # Exit feedback loop [105]

        # User provided feedback, attempt retooling (allow MAX_ITERATIVE_EXPANSION_ATTEMPTS + 2 retool attempts) [105]
        if i < settings.MAX_ITERATIVE_EXPANSION_ATTEMPTS + 2 : # Check if retool attempts remain [105]
             original_proposal_str_for_retool = json.dumps(current_proposal_for_retooling, indent=2) # [105]
             retooled_list = structuring.retool_section_structure(
                 gemini_model_structurer,
                 original_proposal_str_for_retool,
                 user_feedback_str,
                 global_research_content,
                 total_target_minutes
             ) # [105]
             if retooled_list: # [105]
                 proposed_sections_list = retooled_list # Update proposal for next iteration [105]
             else: # Retooling failed [105]
                 print("AI failed to retool sections based on feedback. Displaying previous proposal for confirmation or further feedback.") # [105]
                 # Keep proposed_sections_list as the previous valid one

        else: # Max retool attempts reached [106]
            logging.warning("Max attempts for section retooling reached. Using last proposed structure.") # [106]
            print("Max attempts for retooling sections reached. Please confirm the last proposed structure if acceptable, or restart.") # [106]
            confirmed_sections = current_proposal_for_retooling # Use the last proposal [106]
            # Save the last proposal [106]
            file_utils.save_json_file("confirmed_sections_lapsed.json", confirmed_sections) # [106]
            break # Exit feedback loop [106]


    if not confirmed_sections: # If feedback loop finished without confirmation [107]
        logging.error("Section structuring could not be confirmed.") # [107]
        print("Pipeline halted: Section structure not confirmed.") # [107]
        return # [107]

    # --- Phase 3: Script Generation for Each Section --- [107, 108]
    generated_section_scripts_map = {} # Dict to store scripts by title [107]
    final_section_order = [sec['title'] for sec in confirmed_sections] # List of titles for ordering [107]

    for section_info in confirmed_sections: # Iterate through confirmed sections [107]
        title = section_info.get('title', 'Untitled Section') # [107]
        description = section_info.get('description', '') # [107]
        section_target_mins = section_info.get('estimated_minutes', 5) # Default to 5 min [107]

        section_script = generation.generate_single_section_script(
            gemini_model_script_narrator,
            title,
            description,
            section_target_mins,
            global_research_content,
            user_topic_direction,
            research_influence
        ) # [107, 108]

        if not section_script: # If section generation fails [108]
            logging.error(f"Failed to generate script for section: {title}. Halting.") # [108]
            print(f"ERROR: Failed to generate script for section: {title}. Halting.") # [108]
            return # Halt pipeline [108]

        generated_section_scripts_map[title] = section_script # Store generated script [108]

        # Save individual section script [108]
        # Sanitize title for filename [108]
        section_filename = f"script_section_{re.sub(r'[^a-zA-Z0-9_]+', '', title.replace(' ','_'))[:50]}.txt" # [108]
        file_utils.save_text_file(section_filename, section_script) # [108]

    # --- Phase 4: Stitching and Smoothing --- [109, 110]
    final_script_output_path = file_utils.get_run_specific_path("final_video_script.txt") # [109]

    final_script_content = stitching.stitch_and_smooth_script(
        gemini_model_script_narrator,
        generated_section_scripts_map,
        final_section_order,
        user_topic_direction,
        total_target_minutes
    ) # [109]

    if not final_script_content: # If stitching/smoothing fails [109]
        logging.error("Final script stitching/smoothing failed.") # [109]
        print("Pipeline halted: Final script generation failed.") # [109]
        return # [109]

    # Save the final script [109, 110]
    file_utils.save_text_file("final_video_script.txt", final_script_content) # [109, 110]
    logging.info(f"Final polished script saved to {final_script_output_path}") # [109]
    print(f"Final polished script saved to {final_script_output_path}") # [110]


    # --- Pipeline Complete - Final Reporting --- [110-113]
    overall_end_time = time.time() # [110]
    total_execution_time = overall_end_time - overall_start_time # [110]

    logging.info(f"Application v2.8 finished. Total execution time: {total_execution_time:.2f} seconds.") # [110]
    print(f"\n--- Pipeline Complete (v2.8) ---") # [110]

    if final_script_content: # If a script was generated [110]
        final_estimated_length = estimation_utils.estimate_script_length_minutes(final_script_content) # [111]
        print(f"Final script generated at: {final_script_output_path} (Estimated length: {final_estimated_length:.2f} minutes)") # [111]

        # Attempt to count tokens for the final script [111]
        try: # [111]
            final_script_token_count_response = gemini_model_script_narrator.count_tokens(final_script_content) # [111]
            final_script_token_count = final_script_token_count_response.total_tokens # [111]
            print(f"Token count for the final script content: {final_script_token_count}") # [111]
            logging.info(f"Token count for the final script content: {final_script_token_count}") # [111]
        except Exception as e: # [112]
            logging.error(f"Could not count tokens for final script: {e}", exc_info=True) # [112]
            print(f"Could not count tokens for final script: {e}") # [112]
    else: # If no script was generated [112]
        print(f"Script generation failed. Check logs at: {settings.LOG_FILE_NAME}") # [112] # Use constant from settings


    # Report total token usage and paths [112, 113]
    token_usage = gemini_client.get_token_usage()
    print(f"Global research results: {file_utils.get_run_specific_path('research_results.txt')}") # [112]
    print(f"Logs available at: {file_utils.get_run_specific_path(settings.LOG_FILE_NAME)}") # Use constant from settings [112]
    print(f"All outputs for this run are in: {file_utils._current_run_output_dir}") # Access module global from file_utils [112]
    print(f"Total accumulated Prompt Tokens: {token_usage['prompt_tokens']}") # [112]
    print(f"Total accumulated Candidates Tokens: {token_usage['candidates_tokens']}") # [112]
    print(f"Total accumulated Tokens (sum of all calls): {token_usage['total_tokens']}") # [113]
    print(f"Total execution time: {total_execution_time:.2f} seconds.") # [113]


if __name__ == "__main__": # [113]
    print("Starting Video Script Generator v2.8 (Text Only)...") # [113]
    main() # [113]
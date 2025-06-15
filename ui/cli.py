import logging
import json # Imported for json.dumps for logging/display in feedback section
import os

def read_inputs_from_file(file_path: str) -> tuple[str | None, int | None, float | None, str | None, bool | None]:
    """Reads initial inputs from a text file. Each input should be on a new line."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            
        if len(lines) < 4:
            logging.error(f"Input file {file_path} has insufficient lines. Need at least 4 lines.")
            print(f"ERROR: Input file {file_path} has insufficient lines. Need at least 4 lines.")
            return None, None, None, None, None
            
        topic = lines[0]
        direction = lines[1]
        
        try:
            total_target_minutes = int(lines[2])
            if total_target_minutes <= 0:
                logging.error(f"Invalid minutes value in input file: {total_target_minutes}")
                print(f"ERROR: Invalid minutes value in input file: {total_target_minutes}")
                return None, None, None, None, None
        except ValueError:
            logging.error(f"Invalid minutes value in input file: {lines[2]}")
            print(f"ERROR: Invalid minutes value in input file: {lines[2]}")
            return None, None, None, None, None
            
        try:
            research_influence = float(lines[3])
            if not 0.0 <= research_influence <= 1.0:
                logging.error(f"Invalid research influence value in input file: {research_influence}")
                print(f"ERROR: Invalid research influence value in input file: {research_influence}")
                return None, None, None, None, None
        except ValueError:
            logging.error(f"Invalid research influence value in input file: {lines[3]}")
            print(f"ERROR: Invalid research influence value in input file: {lines[3]}")
            return None, None, None, None, None

        # Handle optional TTS setting (5th line)
        run_tts = None
        if len(lines) >= 5:
            tts_choice = lines[4].lower().strip()
            run_tts = tts_choice in ['yes', 'y', 'true', '1']
            logging.info(f"TTS processing {'enabled' if run_tts else 'disabled'} from input file")
            
        user_topic_direction = f"Topic: {topic}\nDirection: {direction if direction.strip() else 'General overview'}"
        logging.info(f"Read inputs from file: Topic/Direction='{user_topic_direction}', Total Length={total_target_minutes} min, Influence={research_influence}, TTS={run_tts}")
        
        return user_topic_direction, total_target_minutes, research_influence, topic, run_tts
        
    except Exception as e:
        logging.error(f"Failed to read input file {file_path}: {e}", exc_info=True)
        print(f"ERROR: Failed to read input file {file_path}: {e}")
        return None, None, None, None, None

def get_initial_user_inputs(input_file: str | None = None) -> tuple[str | None, int | None, float | None, str | None, bool | None]:
    """
    Gets initial inputs either from a file or interactively.
    Args:
        input_file: Optional path to a file containing inputs. If provided, reads from file instead of prompting.
    """
    if input_file and os.path.exists(input_file):
        return read_inputs_from_file(input_file)
        
    logging.info("Requesting initial user inputs interactively.")
    print("\n--- Video Content Configuration (v2.8) ---")

    topic = input("Enter the main topic/subject for the video: ")
    direction = input("Enter any specific directions or focus areas: ")

    total_target_minutes = None
    while True:
        try:
            total_target_minutes_str = input("Enter TOTAL target script length in minutes (e.g., 30, 60, 120): ")
            total_target_minutes = int(total_target_minutes_str)
            if total_target_minutes > 0:
                break
            else:
                print("Invalid. Please enter a positive number of minutes.")
        except ValueError:
            print("Invalid. Please enter a whole number.")

    research_influence = None
    while True:
        try:
            research_influence_str = input("Enter research influence factor for scriptwriting (0.0 to 1.0, 1.0 = strictly uses research): ")
            research_influence = float(research_influence_str)
            if 0.0 <= research_influence <= 1.0:
                break
            else:
                print("Invalid. Enter a number between 0.0 and 1.0.")
        except ValueError:
            print("Invalid. Please enter a number.")

    if not topic.strip():
        logging.warning("User provided an empty topic.")
        print("WARNING: Topic cannot be empty.")
        return None, None, None, None, None

    user_topic_direction = f"Topic: {topic}\nDirection: {direction if direction.strip() else 'General overview'}"
    logging.info(f"User inputs: Topic/Direction='{user_topic_direction}', Total Length={total_target_minutes} min, Influence={research_influence}")

    # For interactive mode, we don't set TTS preference here - it will be asked after script generation
    return user_topic_direction, total_target_minutes, research_influence, topic, None


def get_user_feedback_on_sections(proposed_sections: list[dict], total_target_minutes: int) -> tuple[str | None, list[dict]]:
    """
    Displays proposed sections to the user and collects feedback.
    Returns feedback string or None if confirmed, and the current proposal.
    """
    print("\n--- AI Proposed Section Structure ---")
    current_total_estimate = 0
    for i, section in enumerate(proposed_sections):
        print(f"{i+1}. Title: {section.get('title', 'N/A')} ({section.get('estimated_minutes', 'N/A')} min)")
        print(f" Description: {section.get('description', 'N/A')}")
        current_total_estimate += section.get('estimated_minutes', 0)

    print(f"------------------------------------")
    print(f"Sum of AI estimated minutes: {current_total_estimate} (Your overall target was: {total_target_minutes})")
    print("\nReview the proposed sections. You can type 'confirm' or provide feedback to adjust, e.g.:")
    print(" 'keep 1,3, remove 2, title of 1 is New Title A, time of 1 is 12, time of 3 is 13'")
    print(" 'reorder to 2,1,3' or 'section 1 needs to cover XYZ and be 15 minutes'")
    print(" 'break up section 2 into Early Years (5 min) and Later Years (7 min)'")

    feedback = input("Your feedback (or 'confirm'): ").strip()

    logging.info(f"User feedback on sections: '{feedback}'. Current proposal: {json.dumps(proposed_sections, indent=2)}") # Log proposal with feedback

    if feedback.lower() == 'confirm':
        return None, proposed_sections # None indicates confirmation

    return feedback, proposed_sections
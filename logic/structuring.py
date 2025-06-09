import logging
import json
from api import gemini_client
from config import settings
from utils import file_utils

def propose_section_structure(gemini_model_structurer, research_content: str, user_topic_direction: str, total_target_minutes: int) -> list[dict] | None:
    """
    Uses the structurer model to propose an initial section structure based on research and user inputs.
    Saves the proposal to a file.
    """
    logging.info("AI proposing section structure...")
    print("\nPhase 2a: AI Proposing Section Structure...")

    # Calculate a reasonable range for the number of sections
    num_sections_lower = max(2, int(total_target_minutes / 15) if total_target_minutes > 15 else 2)
    num_sections_upper = max(num_sections_lower + 2 , int(total_target_minutes / 7) if total_target_minutes > 7 else 4)

    # Construct the proposal prompt
    proposal_prompt = (
        f"You are an expert content strategist. Based on the provided 'Comprehensive Research Material' and the 'User's Original Request', "
        f"propose a structured outline for a long-form narrative script that will be approximately {total_target_minutes} minutes long in total. "
        f"The overall aim is to create a section plan that will form the backbone of a serene, sleep-inducing narrative. Each section should contribute to a gentle unfolding of the topic. "
        f"Suggest between {num_sections_lower} and {num_sections_upper} distinct narrative sections. "
        f"For each proposed section, provide: \n"
        f"1. A concise, engaging 'title'.\n"
        f"2. A single sentence 'description' of its core theme or content.\n"
        f"3. An 'estimated_minutes' for this section (integer).\n"
        f"The sum of your 'estimated_minutes' for all sections should be approximately {total_target_minutes} minutes. "
        f"Ensure the sections flow logically, building upon each other where appropriate, and collectively cover the topic comprehensively and engagingly based on the research. "
        f"Output your proposal STRICTLY as a JSON list of objects, where each object has 'title', 'description', and 'estimated_minutes' keys.\n\n"
        f"User's Original Request:\n{user_topic_direction}\n\n"
        f"Comprehensive Research Material (use this to inform your section proposals):\n{research_content[:settings.RESEARCH_CONTEXT_TRUNCATION_CHAR_LIMIT]}"
    )

    # Call the API for the proposal
    raw_response_json, _ = gemini_client.call_gemini_api(gemini_model_structurer, proposal_prompt, settings.SECTION_PROPOSAL_CONFIG)

    sections_list_to_process = None
    # Attempt to find the list within the JSON response
    if isinstance(raw_response_json, list):
        sections_list_to_process = raw_response_json
    elif isinstance(raw_response_json, dict):
        if 'sections' in raw_response_json and isinstance(raw_response_json['sections'], list):
            sections_list_to_process = raw_response_json['sections']
        elif 'data' in raw_response_json and isinstance(raw_response_json['data'], list):
            sections_list_to_process = raw_response_json['data']
        elif 'items' in raw_response_json and isinstance(raw_response_json['items'], list):
            sections_list_to_process = raw_response_json['items']
        else:
            # Check if any value in the dict is a list
            for key in raw_response_json:
                if isinstance(raw_response_json[key], list):
                    sections_list_to_process = raw_response_json[key]
                    logging.info(f"Found sections list under a non-standard key '{key}' in the JSON response.")
                    break

    if not sections_list_to_process:
        logging.error(f"AI returned a dictionary, but no list of sections found within it. Response: {raw_response_json}")
        print("ERROR: AI returned a dictionary, but no list of sections found within it.")
        return None

    # Validate and clean proposed sections
    valid_sections = []
    if isinstance(sections_list_to_process, list): # Ensure it's actually a list before iterating
        for section in sections_list_to_process:
            if isinstance(section, dict) and 'title' in section and 'description' in section and 'estimated_minutes' in section:
                try:
                    # Ensure minutes is an integer and at least 1
                    section['estimated_minutes'] = int(section['estimated_minutes'])
                    if section['estimated_minutes'] < 1: section['estimated_minutes'] = 1
                    valid_sections.append(section)
                except ValueError:
                    logging.warning(f"Invalid 'estimated_minutes' in proposed section: {section}")
            else:
                logging.warning(f"Invalid section structure in AI proposal: {section}")

    if valid_sections:
        logging.info(f"AI proposed {len(valid_sections)} valid sections.")
        # Save the initial proposal using the file utility
        file_utils.save_json_file("section_proposal_initial.json", valid_sections)
        return valid_sections
    else:
        logging.error(f"AI proposed sections (extracted list: {sections_list_to_process}) but none were valid after parsing.");
        print("ERROR: AI proposed sections in an invalid format or with missing fields.");
        return None

# Note: The retool_section_structure function is highly similar in structure
# It will also take the structurer model, user feedback, original proposal, research, and target minutes.
# It needs to be implemented following the prompt and parsing logic.

def retool_section_structure(gemini_model_structurer, original_proposal_str: str, user_feedback: str, research_content: str, total_target_minutes: int) -> list[dict] | None:
    """
    Uses the structurer model to revise the section structure based on user feedback.
    """
    logging.info(f"AI retooling section structure based on feedback: {user_feedback}")
    print("\nPhase 2c: AI Retooling Section Structure based on your feedback...")

    # Construct the retool prompt
    retool_prompt = (
        f"You are an expert content strategist. You previously proposed a section structure. The user has provided feedback. "
        f"Your task is to revise the section structure based *strictly* on the user's feedback. "
        f"Interpret commands like 'keep 1,3', 'remove 2', 'reorder to 3,1,2', 'title of 1 is New Title', 'time of 2 is 10 min', 'break up section X into Y (A min) and Z (B min)'. "
        f"If the user asks to remove themes (e.g., 'no coverage of his time as Vader'), ensure all sections primarily focused on that theme are removed. "
        f"After applying the user's explicit changes, if the sum of 'estimated_minutes' for the new set of sections significantly deviates "
        f"from the overall target of {total_target_minutes} minutes, intelligently adjust the times of the sections "
        f"(prioritizing those the user didn't explicitly set time for, or proportionally adjusting others) to get closer to the total target. "
        f"While adhering strictly to explicit user instructions, if the feedback is minimal (e.g., only a time change for one section), ensure the overall narrative logic and comprehensiveness derived from the 'Comprehensive Research Material' are maintained in the revised structure. "
        f"The number of sections should primarily be guided by the user's keep/remove/add/break up instructions.\n\n"
        f"Original Proposal (for context - section numbers are 1-indexed as shown to user):\n{original_proposal_str}\n\n"
        f"User's Feedback for Revision:\n{user_feedback}\n\n"
        f"Comprehensive Research Material (use this to ensure sections are still relevant if titles/descriptions change, or if new sections are implied by 'break up' commands):\n{research_content[:settings.RESEARCH_CONTEXT_TRUNCATION_CHAR_LIMIT]}\n\n" # Truncate [53]
        f"Instructions for Revision:\n"
        f"- Directly apply user's instructions for keeping, removing, reordering, renaming sections, or setting specific section times. If asked to 'break up' a section, create new logical sub-sections with appropriate titles, descriptions, and time allocations that sum to the original section's time or as specified by user, ensuring these new sub-sections are still well-supported by the 'Comprehensive Research Material'.\n"
        f"- After applying direct changes, ensure the NEW sum of 'estimated_minutes' for all sections in your revised proposal is approximately {total_target_minutes} minutes. Adjust other section times as needed.\n"
        f"- For each section in your revised proposal, provide: a 'title', a one-sentence 'description', and an 'estimated_minutes' (integer, must be at least 1).\n"
        f"- Output your revised proposal STRICTLY as a JSON list of objects.\n"
    )

    # Call the API for retooling
    raw_retooled_response_json, _ = gemini_client.call_gemini_api(gemini_model_structurer, retool_prompt, settings.SECTION_PROPOSAL_CONFIG)

    retooled_sections_list_to_process = None
    # Attempt to find the list within the JSON response (similar logic to proposal)
    if isinstance(raw_retooled_response_json, list):
        retooled_sections_list_to_process = raw_retooled_response_json
    elif isinstance(raw_retooled_response_json, dict):
        if 'sections' in raw_retooled_response_json and isinstance(raw_retooled_response_json['sections'], list):
            retooled_sections_list_to_process = raw_retooled_response_json['sections']
        elif 'data' in raw_retooled_response_json and isinstance(raw_retooled_response_json['data'], list):
            retooled_sections_list_to_process = raw_retooled_response_json['data']
        elif 'items' in raw_retooled_response_json and isinstance(raw_retooled_response_json['items'], list):
            retooled_sections_list_to_process = raw_retooled_response_json['items']
        else:
            # Check if any value is a list
            for key in raw_retooled_response_json:
                if isinstance(raw_retooled_response_json[key], list):
                    retooled_sections_list_to_process = raw_retooled_response_json[key]
                    logging.info(f"Found retooled sections list under a non-standard key '{key}' in the JSON response.")
                    break


    if not retooled_sections_list_to_process:
        logging.error(f"AI returned a dictionary for retooling, but no list of sections found. Response: {raw_retooled_response_json}")
        print("ERROR: AI returned a dictionary for retooling, but no list of sections found.")
        return None

    # Validate and clean retooled sections
    valid_sections = []
    if isinstance(retooled_sections_list_to_process, list): # Ensure it's a list
        for section in retooled_sections_list_to_process:
            if isinstance(section, dict) and 'title' in section and 'description' in section and 'estimated_minutes' in section:
                try:
                    section['estimated_minutes'] = max(1, int(section['estimated_minutes'])) # Ensure integer and at least 1
                    valid_sections.append(section)
                except ValueError:
                    logging.warning(f"Invalid 'estimated_minutes' in retooled section: {section}")
            else:
                logging.warning(f"Invalid section structure in AI retooling: {section}")

    if valid_sections:
        logging.info(f"AI retooled to {len(valid_sections)} valid sections.")
        # Note: The original script didn't save the retooled JSON after each retool step,
        # but it might be useful for debugging. For now, sticking to the original behavior.
        return valid_sections
    else:
        logging.error(f"AI retooled sections (extracted list: {retooled_sections_list_to_process}) but none were valid after parsing.");
        print("ERROR: AI retooled sections in an invalid format or with missing fields.");
        return None

    # Fallback if response was not a list or dict containing a list
    # This part is handled by the `if not retooled_sections_list_to_process:` block above
    # logging.error(f"AI failed to retool sections or returned unexpected format. Raw Response: {raw_retooled_response_json}");
    # print("ERROR: AI failed to retool or returned an unexpected format.");
    # return None
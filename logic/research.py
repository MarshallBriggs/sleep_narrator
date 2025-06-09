import logging
from api import gemini_client
from config import settings
from utils import file_utils

def perform_global_research(gemini_model_research, user_topic_direction: str, total_target_minutes: int) -> str | None:
    """
    Performs global research using the research model based on user inputs and target length.
    Saves the research results to a file.
    """
    logging.info(f"Starting global research for: {user_topic_direction} to support ~{total_target_minutes} min total script.") # [28]
    print(f"\nPhase 1: Performing Global Research (Web Search Enabled, for ~{total_target_minutes} min total script)...") # [28]

    # Construct the research prompt [39-42]
    research_prompt_detail = (
        f"Please conduct exceptionally detailed and comprehensive research using available tools based on the following user request. "
        f"The overall goal is to gather enough material for a long-form narrative approximately {total_target_minutes} minutes long. "
        f"Focus on gathering extensive key facts, figures, historical context, in-depth narratives covering multiple facets and sub-topics, "
        f"supporting details, and diverse perspectives related to the core request. **Identify the most pivotal or defining factual details, speculative consequences, or unique aspects of the 'what-if' scenario that would be most impactful for a narrative. Also, note any logical branching points or areas of uncertainty within 'what-if' scenarios that could be explored with plausible invention.** " # [39]
        f"For each major aspect or event, aim to find several paragraphs of detailed information. " # [40]
        f"The output should be a **synthesized summary of the gathered information, written in your own words.** " # [40]
        f"While detailed facts, figures, and narrative elements are essential, the goal is a coherent body of research material, " # [40]
        f"not a direct concatenation or extensive quotation from source documents. If a very brief, direct quote is absolutely essential " # [40]
        f"to convey a specific point that cannot be paraphrased, it must be clearly identified as such, but extensive quoting should be avoided. " # [40]
        f"When gathering information, try to draw from diverse and credible sources to ensure a well-rounded understanding of the topic. " # [41]
        f"Focus on information that would be particularly suitable for developing into a gentle, sleep-inducing narrative, " # [41]
        f"such as descriptive passages, interesting but not jarring anecdotes, context that can be presented calmly, **and specific, illustrative examples of causes and effects or key factual points that can be woven into a calm, story-like account.** " # [41]
        f"The output should be a rich, detailed body of text, not a structured outline. " # [42]
        f"Ensure depth, breadth, and clarity.\n\nUser Request:\n{user_topic_direction}" # [42]
    )

    # Call the API for research [42]
    research_text, _ = gemini_client.call_gemini_api(gemini_model_research, research_prompt_detail, settings.RESEARCH_GENERATION_CONFIG) # [42]

    if research_text: # [42]
        # Save the research results using the file utility [42]
        file_utils.save_text_file("research_results.txt", research_text) # [42, 43]
        logging.info(f"Global research complete. Saved to {file_utils.get_run_specific_path('research_results.txt')}") # [42]
        print(f"Global research complete. Saved to {file_utils.get_run_specific_path('research_results.txt')}") # [42]
        return research_text # [42]
    else: # [43]
        logging.error("Global research phase failed.") # [43]
        print("ERROR: Global research failed.") # [43]
        return None # [43]
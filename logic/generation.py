import logging
from api import gemini_client
from config import settings
from utils import estimation_utils

def generate_single_section_script(
    gemini_model_script,
    section_title: str,
    section_description: str,
    section_target_minutes: int,
    global_research_text: str,
    original_user_topic_direction: str,
    research_influence: float
) -> str | None:
    """
    Generates the narrative script for a single section, including iterative expansion.
    """
    logging.info(f"Generating script for section: '{section_title}' (Target: ~{section_target_minutes} min).")
    print(f"\nGenerating script for section: '{section_title}' (Target: ~{section_target_minutes} min)...")

    # Determine if it's a "what-if" scenario
    is_what_if_scenario = (
        "what if" in original_user_topic_direction.lower() or
        "if he had" in original_user_topic_direction.lower() or
        "if the us didnt join" in original_user_topic_direction.lower() or
        "had he beaten" in original_user_topic_direction.lower()
    )

    # Determine the generation config (dynamic or fixed max tokens)
    section_gen_config: settings.GeminiGenerationConfig
    if settings.USE_DYNAMIC_MAX_TOKENS_FOR_SCRIPT_SECTIONS:
        # Estimate target tokens based on target minutes and buffers
        estimated_section_tokens = int(section_target_minutes * settings.WORDS_PER_MINUTE_NARRATION * settings.TOKENS_PER_WORD_ESTIMATE * (1 + settings.TOKEN_BUFFER_PERCENTAGE))
        # Ensure dynamic max tokens don't exceed the model's absolute limit
        dynamic_max_tokens_section = min(estimated_section_tokens, settings.MODEL_ABSOLUTE_MAX_OUTPUT_TOKENS)

        section_gen_config = settings.GeminiGenerationConfig(
            temperature=settings.SCRIPT_SECTION_GENERATION_CONFIG_BASE.temperature,
            max_output_tokens=dynamic_max_tokens_section,
            top_p=settings.SCRIPT_SECTION_GENERATION_CONFIG_BASE.top_p,
            top_k=settings.SCRIPT_SECTION_GENERATION_CONFIG_BASE.top_k
        )
        logging.info(f"Section '{section_title}': Dynamic max_output_tokens: {dynamic_max_tokens_section}")
    else:
        section_gen_config = settings.GeminiGenerationConfig(
            temperature=settings.SCRIPT_SECTION_GENERATION_CONFIG_BASE.temperature,
            max_output_tokens=settings.TESTING_SCRIPT_SECTION_MAX_TOKENS,
            top_p=settings.SCRIPT_SECTION_GENERATION_CONFIG_BASE.top_p,
            top_k=settings.SCRIPT_SECTION_GENERATION_CONFIG_BASE.top_k
        )
        logging.info(f"Section '{section_title}': Fixed testing max_output_tokens: {settings.TESTING_SCRIPT_SECTION_MAX_TOKENS}")

    # Determine influence instruction based on research_influence factor
    influence_instruction = ""
    if research_influence >= 0.8:
        influence_instruction = "You MUST primarily and strictly base your script on the provided 'Comprehensive Research Material' relevant to this section's theme and description. This means weaving specific facts, anecdotes, descriptions, and narrative threads from the research directly into your narration for this section. Avoid introducing significant information or narrative paths not supported by this research for this section."
    elif research_influence <= 0.2:
        influence_instruction = "Use the 'Comprehensive Research Material' relevant to this section's theme and description as a foundational guide and inspiration. You have significant creative freedom to expand, introduce complementary details and illustrative examples from your general knowledge that align with the serene tone, and weave a compelling narrative for this section."
    else:
        influence_instruction = "Use the 'Comprehensive Research Material' relevant to this section's theme and description as the primary basis. You may supplement moderately with illustrative details or gentle elaborations from your general knowledge to enhance the narrative flow and descriptive richness of this section, ensuring all additions maintain the established calm persona."

    # Determine what-if specific creative instruction
    what_if_creative_instruction = ""
    if is_what_if_scenario:
        what_if_creative_instruction = (
            "**Given that this is a 'what-if' scenario, you are encouraged to invent plausible narrative beats, character interactions, or logical consequences that align with the established premise and the serene tone. Use the research as a springboard for these creative yet logical developments, filling in gaps or exploring unstated possibilities to create an engaging speculative narrative. Ensure these inventions flow naturally from the 'what-if' conditions and that their key consequences are gently described, showing their impact within this section or setting up logical developments for future parts of the narrative.**"
        )
    else:
        what_if_creative_instruction = (
            "**As this topic appears to be factual or historical, adhere strictly to the provided research and established information when detailing events and consequences. Avoid inventing narrative points not supported by the research.**"
        )

    # Construct the initial section script generation prompt
    section_script_prompt = (
        f"\n"
        f"Original User Topic/Direction (for overall context):\n{original_user_topic_direction}\n\n"
        f"Comprehensive Research Material (draw relevant details from this for the current section):\n{global_research_text[:settings.RESEARCH_CONTEXT_TRUNCATION_CHAR_LIMIT]}\n"
        f"\n\n"
        f"\n"
        f"Title: {section_title}\n"
        f"Description: {section_description}\n"
        f"Target Length for this section: Approximately {section_target_minutes} minutes.\n"
        f"\n\n"
        f"\n"
        f"Write the script content ONLY for this specific section: '{section_title}'. "
        f"It is imperative that this section's content is sufficiently long and detailed to be spoken over approximately {section_target_minutes} minutes. "
        f"**Aim for your initial generation to be as close as possible to this target length. Do not significantly undershoot this target length for THIS SECTION in your first attempt. Expand on relevant details from the research material that fit the section's title and description to achieve this initial length.** "
        f"Focus on describing the unfolding events, the atmosphere of the times or settings, the observable actions or developments, and the subtle currents of change or existence in a gentle, nebulous, and calming way. "
        f"**While adhering to the serene persona, ensure this section conveys substantive information relevant to its title and description. Actively draw upon the 'Comprehensive Research Material' to include the *most impactful and defining* specific details, key developments, named entities or locations if central to the point, and their tangible consequences as outlined in the research. Gently illustrate these points, perhaps by tracing a consequence one step further or by using varied serene descriptive language that reflects the nature of the specific information. The listener should feel they are gently learning specific insights or understanding concrete implications related to the topic.** "
        f"{what_if_creative_instruction}\n"
        f"The paramount goal is a script section that, when narrated, will be an effective sleep aid and fit into the larger narrative. "
        f"{influence_instruction}\n"
        f"Adhere strictly to the System Instruction (calm, observational chronicler, pure narration, no scene directions, exceptionally gentle language, etc.). "
        f"Conclude this section in a way that feels complete for its specific theme, yet leaves a natural opening for a subsequent, related topic to follow, without explicitly foreshadowing or referencing other section titles. "
        f"Focus SOLELY on delivering the words the narrator will speak for THIS SECTION.\n"
        f""
    )

    # Call the API for initial generation
    script_text, finish_reason = gemini_client.call_gemini_api(gemini_model_script, section_script_prompt, section_gen_config)

    # Prepend the section title to the generated content
    if script_text:
        script_text = f"{section_title}\n\n{script_text}"

    # --- Iterative expansion if needed and not cut off by max_tokens initially ---
    if script_text and finish_reason != "MAX_TOKENS": # Only attempt expansion if initial gen was not cut off
        current_length_minutes = estimation_utils.estimate_script_length_minutes(script_text)
        expansion_attempts = 0
        target_word_count_for_section = int(section_target_minutes * settings.WORDS_PER_MINUTE_NARRATION)
        expansion_finish_reason = None  # Initialize before the loop

        # Loop for expansion until length is acceptable or max attempts reached
        while (
            (section_target_minutes - current_length_minutes) > settings.SCRIPT_LENGTH_ACCEPTABLE_VARIANCE_MINUTES / 2 and # Needs significant expansion
            current_length_minutes < (section_target_minutes * (1 + (settings.TOKEN_BUFFER_PERCENTAGE / 3))) and # Don't expand if already significantly over
            expansion_attempts < settings.MAX_ITERATIVE_EXPANSION_ATTEMPTS # Limit attempts
        ):
            expansion_attempts += 1
            current_word_count = int(current_length_minutes * settings.WORDS_PER_MINUTE_NARRATION)
            additional_words_needed = max(0, target_word_count_for_section - current_word_count)
            # Estimate paragraphs to add
            num_paragraphs_to_add = max(1, round(additional_words_needed / settings.AVERAGE_WORDS_PER_PARAGRAPH_FOR_EXPANSION))

            # Check if remaining needed length is too small to be worth expanding
            if additional_words_needed < (settings.WORDS_PER_MINUTE_NARRATION * settings.MIN_SECTION_TIME_FOR_EXPANSION_PROMPT / 2) and expansion_attempts > 1:
                logging.info(f"Section '{section_title}': Remaining words needed ({additional_words_needed}) too small for effective expansion. Stopping.")
                break

            logging.info(f"Section '{section_title}': length {current_length_minutes:.2f} min ({current_word_count} words), target {section_target_minutes} min ({target_word_count_for_section} words). Needs ~{additional_words_needed} more words (approx. {num_paragraphs_to_add} paragraphs). Expansion attempt {expansion_attempts}/{settings.MAX_ITERATIVE_EXPANSION_ATTEMPTS}.")
            print(f"INFO: Section '{section_title}' too short ({current_length_minutes:.2f} min). Expanding (attempt {expansion_attempts}) to add ~{num_paragraphs_to_add} more paragraphs...")

            # Calculate expansion max tokens
            expansion_max_tokens = int(target_word_count_for_section * settings.TOKENS_PER_WORD_ESTIMATE * (1 + settings.TOKEN_BUFFER_PERCENTAGE * 1.5)) # Slightly larger buffer for expansion
            expansion_max_tokens = min(expansion_max_tokens, settings.MODEL_ABSOLUTE_MAX_OUTPUT_TOKENS)

            expansion_gen_config_section = settings.GeminiGenerationConfig(
                temperature=settings.SCRIPT_SECTION_GENERATION_CONFIG_BASE.temperature, # Potentially slightly higher temp for more creative expansion if desired
                max_output_tokens=expansion_max_tokens,
                top_p=settings.SCRIPT_SECTION_GENERATION_CONFIG_BASE.top_p,
                top_k=settings.SCRIPT_SECTION_GENERATION_CONFIG_BASE.top_k
            )
            logging.info(f"Section '{section_title}' Expansion attempt {expansion_attempts}: using max_output_tokens: {expansion_gen_config_section.max_output_tokens}")

            # Determine expansion what-if specific creative instruction
            expansion_what_if_creative_instruction = ""
            if is_what_if_scenario:
                expansion_what_if_creative_instruction = (
                    "**As this is a 'what-if' scenario, when expanding, feel free to introduce new plausible narrative developments, character interactions, or logical consequences that extend the story, using the research as a creative springboard. When doing so, also consider and gently elaborate on the immediate ripple effects or logical next steps that stem from these invented elements, ensuring they enrich the ongoing story. Ensure these inventions are consistent with the established premise and serene tone.**"
                )
            else:
                 expansion_what_if_creative_instruction = (
                    "**For this factual/historical topic, ensure expansion focuses on elaborating on existing information from the research or adding further supporting details. Do not invent new narrative points.**"
                 )


            # Construct the expansion prompt
            expansion_prompt_section = (
                f"The following script was generated for the section titled '{section_title}'. "
                f"The target length for this section is approximately {section_target_minutes} minutes (around {target_word_count_for_section} words). "
                f"The current version is only {current_length_minutes:.2f} minutes long (around {current_word_count} words). "
                f"It needs approximately {additional_words_needed} more words (which is about {num_paragraphs_to_add} substantial paragraphs) to reach its target.\n\n"
                f"Original User Topic/Direction (for overall context):\n{original_user_topic_direction}\n\n"
                f"Comprehensive Research Material (use this to find more details relevant to '{section_title}'):\n{global_research_text[:settings.RESEARCH_CONTEXT_TRUNCATION_CHAR_LIMIT]}\n\n"
                f"Current Script for Section '{section_title}' (to be expanded and integrated):\n{script_text}\n\n"
                f"Task: Please significantly expand the 'Current Script for Section \"{section_title}\"' by adding approximately {num_paragraphs_to_add} new, substantial paragraphs of narrative content. "
                f"To achieve this, identify 1-2 specific themes, events, or descriptive passages *within the current section's text* that are underdeveloped or too brief. "
                f"For each of these identified areas, add the requested number of new, detailed paragraphs, drawing rich details, descriptions, or elaborations from the 'Comprehensive Research Material' that are relevant to '{section_title}'. "
                f"**When adding new paragraphs, focus not only on atmospheric expansion but also on introducing or elaborating on the *most impactful and defining* specific factual details, events, their causes, or their concrete consequences as detailed in the 'Comprehensive Research Material' that are pertinent to this section. For instance, if the research mentions a specific resource gained or lost, a pivotal character decision, or a key strategic shift, find a gentle way to weave that specific detail and its immediate implications into the expanded narrative. These details should be woven into the narrative with the established gentle and calm tone, using varied serene language appropriate to the information.** "
                f"{expansion_what_if_creative_instruction}\n"
                f"Alternatively, if more appropriate for this section's theme and if the current section structure allows, you may introduce one new, substantial narrative subsection that logically extends its story and is well-supported by the research, ensuring it contributes significantly to the word count. "
                f"Ensure that the newly added paragraphs are not merely repetitive but introduce new depth, detail, or gentle elaboration to the chosen themes, always maintaining the established serene narrative style and drawing from the 'Comprehensive Research Material'. "
                f"All new content must seamlessly integrate with the existing text for this section. "
                f"All expansions MUST strictly adhere to the sleep-inducing persona and style defined in the System Instruction (calm, observational chronicler, exceptionally gentle language, describing unfolding events and atmosphere). "
                f"Provide the complete, expanded script for THIS SECTION ONLY, aiming for a total word count of around {target_word_count_for_section} words for this section."
            )

            # Call API for expansion
            expanded_section_text, expansion_finish_reason = gemini_client.call_gemini_api(gemini_model_script, expansion_prompt_section, expansion_gen_config_section)

            if expanded_section_text:
                # Ensure the section title is preserved in the expanded text
                if not expanded_section_text.startswith(section_title):
                    expanded_section_text = f"{section_title}\n\n{expanded_section_text}"
                
                new_len_min = estimation_utils.estimate_script_length_minutes(expanded_section_text)
                # Only update if expansion was meaningful (added at least 0.2 minutes)
                if new_len_min > current_length_minutes + 0.2:
                    script_text = expanded_section_text
                    current_length_minutes = new_len_min
                    logging.info(f"Section '{section_title}' expansion {expansion_attempts} new length: {current_length_minutes:.2f} min. Finish reason: {expansion_finish_reason}")
                    print(f"INFO: Section '{section_title}' expansion {expansion_attempts} complete. New estimated length: {current_length_minutes:.2f} min.")

                    if expansion_finish_reason == "MAX_TOKENS": # If expansion was cut off, log warning
                        logging.warning(f"Section '{section_title}' expansion {expansion_attempts} hit MAX_TOKENS. Content may be incomplete. Current length: {current_length_minutes:.2f} min.")
                        # Decide if further expansion is useful or if it's good enough
                        if current_length_minutes >= section_target_minutes * 0.9: # If close enough
                             break # Stop expansion
                        # Otherwise, continue expanding if attempts remain and not too short

                else: # If expansion didn't add meaningful length
                    logging.warning(f"Section '{section_title}' expansion {expansion_attempts} didn't significantly lengthen. Current length: {new_len_min:.2f} min vs previous {current_length_minutes:.2f} min. Finish Reason: {expansion_finish_reason}. Stopping expansion for this section.")
                    break # Stop if expansion isn't adding much or if it hits a limit and is still too short

            else: # If expansion API call failed or returned empty
                logging.warning(f"Section '{section_title}' expansion {expansion_attempts} failed or returned empty. Finish Reason: {expansion_finish_reason}. Stopping expansion for this section.")
                break # Stop expansion

            # After the loop, check if we stopped due to MAX_TOKENS and are still significantly short
            if expansion_finish_reason == "MAX_TOKENS" and current_length_minutes < section_target_minutes * 0.85:
                logging.warning(f"Section '{section_title}' hit MAX_TOKENS during expansion {expansion_attempts} and is still significantly short. Stopping expansion to avoid excessive calls.")
                # No break needed here, loop condition handles it.

    elif script_text and finish_reason == "MAX_TOKENS": # If initial generation hit MAX_TOKENS
         logging.warning(f"Initial generation for section '{section_title}' hit MAX_TOKENS. Script may be incomplete. Length: {estimation_utils.estimate_script_length_minutes(script_text):.2f} min.")
         # For now, proceed with the truncated script as in original logic.
         # A more advanced version could attempt continuation here.

    return script_text # Return the final (potentially expanded or truncated) script text
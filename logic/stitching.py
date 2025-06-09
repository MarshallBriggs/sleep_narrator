import logging
from api import gemini_client
from config import settings
from utils import estimation_utils

def stitch_and_smooth_script(
    gemini_model_script,
    section_scripts_map: dict[str, str],
    section_order: list[str],
    original_user_topic_direction: str,
    total_target_minutes: int
) -> str | None:
    """
    Concatenates generated section scripts and performs an iterative smoothing pass.
    """
    logging.info("Starting final stitching and smoothing pass...")
    print("\nPhase 4: Stitching sections and performing final smoothing pass...")

    # Concatenate scripts in the specified order
    full_concatenated_script = ""
    for section_title in section_order:
        script_content = section_scripts_map.get(section_title, "")
        if script_content:
            full_concatenated_script += script_content.strip() + "\n\n"

    full_concatenated_script = full_concatenated_script.strip()

    if not full_concatenated_script:
        logging.error("No script content to stitch."); return None

    # Estimate length before smoothing
    estimated_length_before_smoothing = estimation_utils.estimate_script_length_minutes(full_concatenated_script)
    logging.info(f"Estimated length before smoothing: {estimated_length_before_smoothing:.2f} minutes.")
    print(f"INFO: Estimated length before smoothing: {estimated_length_before_smoothing:.2f} minutes.")

    # --- Smoothing Pass (potentially iterative) ---
    # Adjust max_output_tokens for smoothing: aim for slightly more than current length to allow for transitions
    # but respect the absolute model limit.
    # Estimate current tokens: (Using a character-based approximation)
    # The prompt input char limit is also a factor, chunking might be needed for very long scripts.
    # The original script processes up to SMOOTHING_PROMPT_INPUT_CHAR_LIMIT at a time in the loop.
    # The max_output_tokens calculation seems to be for the *potential* total smoothed script length,
    # but the prompt structure implies smoothing is done on chunks up to SMOOTHING_PROMPT_INPUT_CHAR_LIMIT.
    # Let's adapt the max_output_tokens calculation to the chunk being processed.

    final_script_parts = []
    current_script_to_process = full_concatenated_script
    max_smoothing_iterations = 5
    smoothing_iterations = 0

    # Loop through chunks of the script for smoothing
    while smoothing_iterations < max_smoothing_iterations and current_script_to_process:
        smoothing_iterations += 1
        # Take a chunk up to the defined input limit
        prompt_text_for_smoothing = current_script_to_process[:settings.SMOOTHING_PROMPT_INPUT_CHAR_LIMIT]
        logging.info(f"Smoothing pass iteration {smoothing_iterations}. Processing chunk length: {len(prompt_text_for_smoothing)} chars. Remaining: {len(current_script_to_process) - len(prompt_text_for_smoothing)} chars.")


        # Calculate max_output_tokens for this specific chunk dynamically
        # Aim to output roughly the same number of tokens as the input chunk, plus a small buffer
        # Using a refined approximation based on source
        chunk_input_tokens_approx = int(len(prompt_text_for_smoothing) / 3.5)
        chunk_output_max_tokens = min(chunk_input_tokens_approx + 300, settings.MODEL_ABSOLUTE_MAX_OUTPUT_TOKENS) # Add buffer, cap at model max

        final_script_gen_config = settings.GeminiGenerationConfig(
            temperature=settings.STITCHING_CONFIG.temperature,
            max_output_tokens=chunk_output_max_tokens,
            top_p=settings.STITCHING_CONFIG.top_p,
            top_k=settings.STITCHING_CONFIG.top_k
        )
        logging.info(f"Smoothing pass iteration {smoothing_iterations}: Input chunk approx tokens: {chunk_input_tokens_approx}, Max output tokens for this chunk: {chunk_output_max_tokens}")


        # Construct smoothing prompt for the chunk
        smoothing_prompt = (
            f"The following text consists of concatenated script sections (or a continuation of previously smoothed text), forming a single long-form narrative on the topic: '{original_user_topic_direction}'. "
            f"The target total length for this narrative was approximately {total_target_minutes} minutes. The current concatenated length is approximately {estimated_length_before_smoothing:.2f} minutes. "
            f"Your task is to review this entire provided text. Focus on three main objectives:\n"
            f"1. Ensure smooth, natural-sounding transitions between where the original section boundaries would have been (if this is the first part of the script) or ensure seamless continuation from previous smoothing work. Make minor edits (e.g., adding a transitional phrase, slightly rephrasing) to improve overall coherence and narrative flow.\n"
            f"2. Identify and rewrite any phrases, sentences, or short passages that DO NOT align with the established 'Sleep Narrator' persona (defined in the System Instructions as: an exceptionally calm, gentle, soothing, observational chronicler, using soft and evocative language, avoiding jarring or stimulating words, and maintaining a detached, peaceful perspective even on dramatic topics). Replace such phrases with language that IS consistent with this sleep-inducing style. Pay particular attention to ensuring descriptions of potentially dramatic or intense events are filtered through the calm, observational lens, avoiding any language that could evoke strong emotional responses in the listener other than peace and gentle curiosity.\n"
            f"3. Ensure the overall narrative maintains a consistent 'calm, detached observer' or 'gentle chronicler' perspective throughout. Gently rephrase parts that slip into direct first-person emotional accounts or become too starkly analytical. Ensure even 'what-if' or imaginative content is presented with this same serene, observational, and gently descriptive tone. Ensure a relatively consistent level of descriptive detail and narrative pacing.\n"
            f"**VERY IMPORTANT: Your primary goal is to improve flow and ensure tonal consistency WHILE STRICTLY MAINTAINING the approximate total word count and length of the provided text. The final output length for THIS CHUNK should be very close to the input chunk's length.** "
            f"**DO NOT significantly condense or remove content. If you find passages that seem to contradict the persona, your first priority is to REPHRASE them to fit the persona. Only in extreme cases where rephrasing is impossible should minimal content be removed.** "
            f"Add short transitional sentences if needed for coherence, but the focus is on polishing and preserving length for this chunk. "
            f"Provide the final, polished, continuous script text for THIS CHUNK, strictly adhering to the System Instruction for the narrator's voice.\n\n"
            f"Script Text Chunk to Smooth:\n{prompt_text_for_smoothing}"
        )

        # Call API for smoothing the chunk
        smoothed_chunk, finish_reason = gemini_client.call_gemini_api(gemini_model_script, smoothing_prompt, final_script_gen_config)

        if smoothed_chunk:
            final_script_parts.append(smoothed_chunk)
            logging.info(f"Smoothing pass iteration {smoothing_iterations} successful. Finish reason: {finish_reason}.")

            # Determine if there's more script to process
            # This simplified logic assumes the chunk processed is exactly the input chunk size
            # or was cut off at MAX_TOKENS by the model output itself.
            processed_len = len(prompt_text_for_smoothing) # Assume input chunk was processed
            if finish_reason == "MAX_TOKENS": # If the *output* was cut off
                 logging.warning(f"Smoothing pass iteration {smoothing_iterations} hit MAX_TOKENS. The smoothed chunk may be incomplete.")
                 # This doesn't necessarily mean the *input* chunk wasn't consumed,
                 # but if the output is shorter than expected, it's safer to move to the next chunk from where the *input* ended.
                 pass # Continue to update current_script_to_process below

            current_script_to_process = current_script_to_process[processed_len:] # Move past the processed chunk

            if not current_script_to_process.strip(): # If remaining is just whitespace
                current_script_to_process = "" # No more to process
            # Otherwise, loop continues with the next chunk

        else: # Smoothing failed for this chunk
            logging.warning(f"Smoothing pass iteration {smoothing_iterations} failed or returned empty. Using raw concatenated script for the remainder.");
            final_script_parts.append(current_script_to_process) # Add remaining raw script if smoothing fails
            current_script_to_process = "" # Stop processing
            break # Exit loop if a chunk fails

    # After the smoothing loop
    if current_script_to_process and smoothing_iterations >= max_smoothing_iterations:
         logging.warning("Max smoothing iterations reached, but script remaining to process. Appending raw remaining script.")
         final_script_parts.append(current_script_to_process)


    # Combine all smoothed chunks
    final_script = "\n\n".join(final_script_parts).strip()

    if not final_script: # Should not happen if there was input
        logging.error("Final script is empty after smoothing. Reverting to raw concatenated script.");
        return full_concatenated_script # Return the raw script as a fallback

    return final_script # Return the final smoothed script
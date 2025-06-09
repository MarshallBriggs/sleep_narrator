import os
import time
import json
import logging
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold, Tool
from google.ai.generativelanguage_v1beta.types import GoogleSearchRetrieval
from google.generativeai.types import GenerationConfig as GeminiGenerationConfig

# Global variables for token tracking
total_prompt_tokens_used = 0
total_candidates_tokens_used = 0
total_tokens_accumulated = 0

def get_token_usage():
    """Returns the current accumulated token usage."""
    return {
        "prompt_tokens": total_prompt_tokens_used,
        "candidates_tokens": total_candidates_tokens_used,
        "total_tokens": total_tokens_accumulated
    }

def call_gemini_api(model: genai.GenerativeModel, prompt_content: str, generation_config: GeminiGenerationConfig, max_retries=3, initial_delay=5, is_continuation=False, previous_text="") -> tuple[str | dict | None, str]:
    """
    Calls the Gemini API with retry logic and token counting.
    Returns the generated content (str or dict for JSON) and the finish reason string.
    """
    global total_prompt_tokens_used, total_candidates_tokens_used, total_tokens_accumulated # Use global counters

    full_prompt_for_continuation = prompt_content # Default for non-continuation or first attempt [16]

    if is_continuation and previous_text: # [16]
        full_prompt_for_continuation = (
            f"{previous_text}\n\n"
            f"\n"
            f"Please continue the narration seamlessly from where the previous text ended. "
            f"Ensure you maintain the exact same tone, style, and all previous instructions. "
            f"Do not repeat any content from the provided 'previous text'. "
            f"Focus solely on generating the next part of the narrative as if no interruption occurred.\n"
            f""
        )
        # Log only the instruction part for continuation for brevity [17]
        logging.info(f"Calling Gemini API (Continuation) using model {model.model_name}. Max tokens: {generation_config.max_output_tokens if generation_config else 'default'}. Prompt (instruction part): ...")
    else: # [17]
        logging.info(f"Calling Gemini API using model {model.model_name}. Max tokens: {generation_config.max_output_tokens if generation_config else 'default'}. Prompt (first 100 chars): {str(prompt_content)[:100]}")

    current_retries = 0
    while current_retries < max_retries: # [18]
        try:
            safety_settings = { # [18]
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }

            # Use full_prompt_for_continuation which includes previous_text if it's a continuation call [18]
            final_prompt_to_send = full_prompt_for_continuation if is_continuation else prompt_content

            response = model.generate_content( # [19]
                final_prompt_to_send,
                generation_config=generation_config,
                safety_settings=safety_settings
            )

            # --- Token Counting --- [19]
            prompt_tokens_this_call = 0
            candidates_tokens_this_call = 0
            total_tokens_this_call_reported = 0

            if hasattr(response, 'usage_metadata') and response.usage_metadata: # [19]
                usage = response.usage_metadata
                prompt_tokens_this_call = usage.prompt_token_count
                candidates_tokens_this_call = usage.candidates_token_count if hasattr(usage, 'candidates_token_count') else 0
                total_tokens_this_call_reported = usage.total_token_count if hasattr(usage, 'total_token_count') else (prompt_tokens_this_call + candidates_tokens_this_call)
            else: # Manual counting if metadata is missing [20]
                try:
                    prompt_tokens_this_call = model.count_tokens(final_prompt_to_send).total_tokens
                    if hasattr(response, 'text') and response.text: # Check if response has text before counting
                         candidates_tokens_this_call = model.count_tokens(response.text).total_tokens
                    # Note: manual counting might not exactly match API's internal total for complex cases
                    total_tokens_this_call_reported = prompt_tokens_this_call + candidates_tokens_this_call
                    logging.warning(f"Token usage metadata not found. Manually counted: Prompt={prompt_tokens_this_call}, Candidates={candidates_tokens_this_call}")
                except Exception as e_count: # [21]
                    logging.error(f"Could not count tokens manually after missing metadata: {e_count}")
                    # Set to 0 to avoid inflating totals if counting fails
                    prompt_tokens_this_call = 0
                    candidates_tokens_this_call = 0
                    total_tokens_this_call_reported = 0


            total_prompt_tokens_used += prompt_tokens_this_call # [21]
            total_candidates_tokens_used += candidates_tokens_this_call # [21]
            total_tokens_accumulated += total_tokens_this_call_reported # [21]

            logging.info(f"Token usage for this call: Prompt={prompt_tokens_this_call}, Candidates={candidates_tokens_this_call}, Total Reported={total_tokens_this_call_reported}") # [21]
            logging.info(f"Accumulated tokens: Prompt={total_prompt_tokens_used}, Candidates={total_candidates_tokens_used}, Total={total_tokens_accumulated}") # [21]
            # --- End Token Counting --- [22]

            if not response.candidates: # [22]
                logging.warning(f"Gemini response has no candidates. Prompt feedback: {response.prompt_feedback}") # [22]
                block_reason_msg = "" # [22]
                if response.prompt_feedback and response.prompt_feedback.block_reason: # [22]
                    block_reason_msg = f" Reason: {response.prompt_feedback.block_reason.name}." # [22]
                    if response.prompt_feedback.block_reason_message: # [22]
                         block_reason_msg += f" Message: {response.prompt_feedback.block_reason_message}" # [22]
                print(f"WARNING: Gemini response was blocked or empty.{block_reason_msg}") # [22]
                return None, "BLOCKED_OR_EMPTY" # Return finish reason [23]

            # Updated finish_reason handling for new API structure
            finish_reason = "UNKNOWN"
            if hasattr(response, 'candidates') and response.candidates:
                # Try to get finish_reason from the first candidate
                if len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'finish_reason'):
                        finish_reason = candidate.finish_reason.name if candidate.finish_reason else "UNKNOWN"
                    elif hasattr(candidate, 'finishReason'):  # Alternative attribute name
                        finish_reason = candidate.finishReason.name if candidate.finishReason else "UNKNOWN"

            # Handle JSON response type specifically [23]
            if generation_config and generation_config.response_mime_type == "application/json": # [23]
                try: # [24]
                    if isinstance(response.text, str): # [24]
                        cleaned_text = response.text.strip() # [24]
                        if cleaned_text.startswith("```json"): cleaned_text = cleaned_text[7:] # [24]
                        if cleaned_text.endswith("```"): cleaned_text = cleaned_text[:-3] # [24]
                        cleaned_text = cleaned_text.strip() # [24]
                        json_response = json.loads(cleaned_text) # [24]
                        # Log the JSON response
                        logging.debug(f"Gemini API JSON response: {json.dumps(json_response, indent=2)}")
                    else: # [24]
                        # If not string, maybe it's already the expected type (less likely for API text response)
                        json_response = response.text
                        # Log the non-string JSON response
                        logging.debug(f"Gemini API non-string JSON response: {str(json_response)}")

                    logging.info("Gemini API call successful (JSON response).") # [24]
                    return json_response, finish_reason # [24]

                except json.JSONDecodeError as e: # [24]
                    logging.error(f"Failed to parse JSON response: {e}. Response text: {response.text[:500]}...", exc_info=True) # [24]
                    print(f"ERROR: Failed to parse JSON response from AI for section proposal.") # [24]
                    return None, "JSON_DECODE_ERROR" # [24]
                except AttributeError: # [25]
                    logging.error(f"Unexpected response structure for JSON. response.candidates.content: {getattr(response.candidates, 'content', 'N/A')}", exc_info=True) # [25]
                    return None, "ATTRIBUTE_ERROR_JSON" # [25]

            # Handle Text response [25]
            generated_text = response.text

            # Log the text response (truncated if too long)
            if generated_text:
                # Truncate long responses in logs to avoid overwhelming the log file
                log_text = generated_text[:1000] + "..." if len(generated_text) > 1000 else generated_text
                logging.debug(f"Gemini API text response: {log_text}")

            # Check for empty response text unless finish reason explains it [25]
            if not generated_text.strip() and finish_reason not in ["STOP", "MAX_TOKENS", "SAFETY"]: # Allow empty if max_tokens, stop, or safety [25, 26]
                logging.warning(f"Gemini API returned an empty text response despite having candidates. Finish Reason: {finish_reason}") # [25]
                print(f"WARNING: Gemini API returned an empty text response. Finish Reason: {finish_reason}") # [25]
                # If it's truly empty for unexpected reasons, return None [26]
                return None, finish_reason # [26]

            logging.info(f"Gemini API call successful (Text response). Finish Reason: {finish_reason}") # [26]
            return generated_text, finish_reason # Return text and finish reason [26]

        except (genai.types.generation_types.BlockedPromptException, genai.types.generation_types.StopCandidateException) as specific_gen_error: # [26]
             logging.error(f"Gemini generation error: {specific_gen_error}", exc_info=True) # [26]
             print(f"ERROR: Gemini generation process failed: {specific_gen_error}") # [26]
             return None, "GENERATION_ERROR" # [26]
        except Exception as e: # [27]
            logging.error(f"Gemini API error (Attempt {current_retries + 1}/{max_retries}): {e}", exc_info=True) # [27]
            print(f"ERROR: Gemini API call failed (Attempt {current_retries + 1}/{max_retries}): {e}") # [27]

            # Handle specific API errors [27]
            if "404" in str(e) and "is not found for API version v1beta" in str(e): # [27]
                 return None, "API_NOT_FOUND" # [27]
            elif "429" in str(e) or "ResourceExhausted" in str(e) or "doesn't have a free quota tier" in str(e): # [27]
                 return None, "RATE_LIMIT_OR_QUOTA" # [27]
            elif "500" in str(e) or "503" in str(e): # [27]
                delay = initial_delay * (2 ** current_retries) # [27]
                time.sleep(delay) # [27]
                current_retries += 1 # [27]
            else: # [28]
                return None, "UNKNOWN_API_ERROR" # [28]

    # If max retries are reached [28]
    logging.error("Max retries reached for Gemini API call.") # [28]
    print("ERROR: Max retries reached.") # [28]
    return None, "MAX_RETRIES_REACHED" # [28]
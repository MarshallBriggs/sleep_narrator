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

    full_prompt_for_continuation = prompt_content # Default for non-continuation or first attempt

    if is_continuation and previous_text:
        full_prompt_for_continuation = (
            f"{previous_text}\n\n"
            f"\n"
            f"Please continue the narration seamlessly from where the previous text ended. "
            f"Ensure you maintain the exact same tone, style, and all previous instructions. "
            f"Do not repeat any content from the provided 'previous text'. "
            f"Focus solely on generating the next part of the narrative as if no interruption occurred.\n"
            f""
        )
        # Log only the instruction part for continuation for brevity
        logging.info(f"Calling Gemini API (Continuation) using model {model.model_name}. Max tokens: {generation_config.max_output_tokens if generation_config else 'default'}. Prompt (instruction part): ...")
    else:
        logging.info(f"Calling Gemini API using model {model.model_name}. Max tokens: {generation_config.max_output_tokens if generation_config else 'default'}. Prompt (first 100 chars): {str(prompt_content)[:100]}")

    current_retries = 0
    while current_retries < max_retries:
        try:
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }

            # Use full_prompt_for_continuation which includes previous_text if it's a continuation call
            final_prompt_to_send = full_prompt_for_continuation if is_continuation else prompt_content

            response = model.generate_content(
                final_prompt_to_send,
                generation_config=generation_config,
                safety_settings=safety_settings
            )

            # --- Token Counting ---
            prompt_tokens_this_call = 0
            candidates_tokens_this_call = 0
            total_tokens_this_call_reported = 0

            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = response.usage_metadata
                prompt_tokens_this_call = usage.prompt_token_count
                candidates_tokens_this_call = usage.candidates_token_count if hasattr(usage, 'candidates_token_count') else 0
                total_tokens_this_call_reported = usage.total_token_count if hasattr(usage, 'total_token_count') else (prompt_tokens_this_call + candidates_tokens_this_call)
            else: # Manual counting if metadata is missing
                try:
                    prompt_tokens_this_call = model.count_tokens(final_prompt_to_send).total_tokens
                    if hasattr(response, 'text') and response.text: # Check if response has text before counting
                         candidates_tokens_this_call = model.count_tokens(response.text).total_tokens
                    # Note: manual counting might not exactly match API's internal total for complex cases
                    total_tokens_this_call_reported = prompt_tokens_this_call + candidates_tokens_this_call
                    logging.warning(f"Token usage metadata not found. Manually counted: Prompt={prompt_tokens_this_call}, Candidates={candidates_tokens_this_call}")
                except Exception as e_count:
                    logging.error(f"Could not count tokens manually after missing metadata: {e_count}")
                    # Set to 0 to avoid inflating totals if counting fails
                    prompt_tokens_this_call = 0
                    candidates_tokens_this_call = 0
                    total_tokens_this_call_reported = 0


            total_prompt_tokens_used += prompt_tokens_this_call
            total_candidates_tokens_used += candidates_tokens_this_call
            total_tokens_accumulated += total_tokens_this_call_reported

            logging.info(f"Token usage for this call: Prompt={prompt_tokens_this_call}, Candidates={candidates_tokens_this_call}, Total Reported={total_tokens_this_call_reported}")
            logging.info(f"Accumulated tokens: Prompt={total_prompt_tokens_used}, Candidates={total_candidates_tokens_used}, Total={total_tokens_accumulated}")
            # --- End Token Counting ---

            if not response.candidates:
                logging.warning(f"Gemini response has no candidates. Prompt feedback: {response.prompt_feedback}")
                block_reason_msg = ""
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    block_reason_msg = f" Reason: {response.prompt_feedback.block_reason.name}."
                    if response.prompt_feedback.block_reason_message:
                         block_reason_msg += f" Message: {response.prompt_feedback.block_reason_message}"
                print(f"WARNING: Gemini response was blocked or empty.{block_reason_msg}")
                return None, "BLOCKED_OR_EMPTY"

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

            # Handle JSON response type specifically
            if generation_config and generation_config.response_mime_type == "application/json":
                try:
                    if isinstance(response.text, str):
                        cleaned_text = response.text.strip()
                        if cleaned_text.startswith("```json"): cleaned_text = cleaned_text[7:]
                        if cleaned_text.endswith("```"): cleaned_text = cleaned_text[:-3]
                        cleaned_text = cleaned_text.strip()
                        json_response = json.loads(cleaned_text)
                        # Log the JSON response
                        logging.debug(f"Gemini API JSON response: {json.dumps(json_response, indent=2)}")
                    else:
                        # If not string, maybe it's already the expected type (less likely for API text response)
                        json_response = response.text
                        # Log the non-string JSON response
                        logging.debug(f"Gemini API non-string JSON response: {str(json_response)}")

                    logging.info("Gemini API call successful (JSON response).")
                    return json_response, finish_reason

                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse JSON response: {e}. Response text: {response.text[:500]}...", exc_info=True)
                    print(f"ERROR: Failed to parse JSON response from AI for section proposal.")
                    return None, "JSON_DECODE_ERROR"
                except AttributeError:
                    logging.error(f"Unexpected response structure for JSON. response.candidates.content: {getattr(response.candidates, 'content', 'N/A')}", exc_info=True)
                    return None, "ATTRIBUTE_ERROR_JSON"

            # Handle Text response
            generated_text = response.text

            # Log the text response (truncated if too long)
            if generated_text:
                # Truncate long responses in logs to avoid overwhelming the log file
                log_text = generated_text[:1000] + "..." if len(generated_text) > 1000 else generated_text
                logging.debug(f"Gemini API text response: {log_text}")

            # Check for empty response text unless finish reason explains it
            if not generated_text.strip() and finish_reason not in ["STOP", "MAX_TOKENS", "SAFETY"]: # Allow empty if max_tokens, stop, or safety
                logging.warning(f"Gemini API returned an empty text response despite having candidates. Finish Reason: {finish_reason}")
                print(f"WARNING: Gemini API returned an empty text response. Finish Reason: {finish_reason}")
                # If it's truly empty for unexpected reasons, return None
                return None, finish_reason

            logging.info(f"Gemini API call successful (Text response). Finish Reason: {finish_reason}")
            return generated_text, finish_reason # Return text and finish reason

        except (genai.types.generation_types.BlockedPromptException, genai.types.generation_types.StopCandidateException) as specific_gen_error:
             logging.error(f"Gemini generation error: {specific_gen_error}", exc_info=True)
             print(f"ERROR: Gemini generation process failed: {specific_gen_error}")
             return None, "GENERATION_ERROR"
        except Exception as e:
            logging.error(f"Gemini API error (Attempt {current_retries + 1}/{max_retries}): {e}", exc_info=True)
            print(f"ERROR: Gemini API call failed (Attempt {current_retries + 1}/{max_retries}): {e}")

            # Handle specific API errors
            if "404" in str(e) and "is not found for API version v1beta" in str(e):
                 return None, "API_NOT_FOUND"
            elif "429" in str(e) or "ResourceExhausted" in str(e) or "doesn't have a free quota tier" in str(e):
                 return None, "RATE_LIMIT_OR_QUOTA"
            elif "500" in str(e) or "503" in str(e):
                delay = initial_delay * (2 ** current_retries)
                time.sleep(delay)
                current_retries += 1
            else:
                return None, "UNKNOWN_API_ERROR"

    # If max retries are reached
    logging.error("Max retries reached for Gemini API call.")
    print("ERROR: Max retries reached.")
    return None, "MAX_RETRIES_REACHED"
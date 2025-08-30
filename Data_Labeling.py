# ============================================
# 1) Setup (PC Environment)
# ============================================
# pip install --upgrade openai json5

import os, json, json5, sys, time, textwrap, ast
from getpass import getpass
from typing import List, Dict, Any
from openai import OpenAI

# Set API key directly
os.environ["OPENAI_API_KEY"] = "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"

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

INPUT_PATH  = os.path.expanduser("~/Desktop/review-Alabama.json")     # macOS path
OUTPUT_PATH = os.path.expanduser("~/Desktop/review-Alabama_labeled.jsonl")

ALLOWED_LABELS = [
    "Irrelevant Content",  # Not related to the business experience/service/products
    "Advertisement",       # Promotes other businesses/services; links or contact info
    "Rant",                # Overly emotional/hostile without substantive critique
    "Spam",                # Fake, gibberish, automated, or repetitively posted
    "Good Review",         # Positive, constructive review about the business
]
# Optional: if you want a "clean" bucket when none of the above apply, uncomment:
# ALLOWED_LABELS.append("None")


SYSTEM_PROMPT = f"""You are a strict policy classifier for Google-style location reviews.
You must choose exactly ONE label from this set:
{ALLOWED_LABELS}

Definitions:
- Irrelevant Content: Content not related to the business experience, service, or products.
- Advertisement: Promotes other businesses/services or includes links/contact info.
- Rant: Non-constructive, overly emotional, or hostile content without substantive critique.
- Spam: Fake, gibberish, automated, or repetitively posted content.
- Good Review:Relevant, constructive review about the business experience, service, or products.
If none apply, choose "None" ONLY if that option is present in the allowed set.

Return ONLY JSON:
{{
  "label": "<one of {ALLOWED_LABELS}>"
}}
Do not include any extra keys or text.
"""

def _try_parse_any(s: str) -> Any:
    """
    Be lenient: try JSON, then JSON5, then Python literal_eval (as last resort).
    Also supports JSONL (one object per line).
    """
    s_stripped = s.strip()

    # Try JSONL first: if multiple lines each decodes cleanly.
    items = []
    lines = [ln for ln in s_stripped.splitlines() if ln.strip()]
    multi_line_candidates = 0
    for ln in lines:
        try:
            items.append(json.loads(ln))
            multi_line_candidates += 1
        except Exception:
            items = []
            break
    if multi_line_candidates > 1 and items:
        return items

    # Try standard JSON
    try:
        obj = json.loads(s_stripped)
        return obj
    except Exception:
        pass

    # Try JSON5 (handles single quotes, trailing commas, etc.)
    try:
        obj = json5.loads(s_stripped)
        return obj
    except Exception:
        pass

    # Last resort: Python literal (e.g., a dict with single quotes)
    try:
        obj = ast.literal_eval(s_stripped)
        return obj
    except Exception:
        pass

    raise ValueError("Unable to parse the input file as JSON/JSON5/JSONL/Python literal.")

def load_reviews(path: str) -> List[Dict[str, Any]]:
    print(f"DEBUG: Attempting to load reviews from: {path}")
    
    if not os.path.exists(path):
        print(f"ERROR: File '{path}' not found.")
        
        # Create a sample file for testing on PC
        print(f"Creating sample file for testing purposes...")
        sample_data = [
            {"text": "Great food and service! Highly recommend this place.", "rating": 5, "author": "user1"},
            {"text": "Terrible experience, rude staff and cold food.", "rating": 1, "author": "user2"},
            {"text": "Average place, nothing special but decent prices.", "rating": 3, "author": "user3"},
            {"text": "Visit my website for better deals! www.example.com", "rating": 5, "author": "spammer"},
            {"text": "This place is HORRIBLE!!! WORST EXPERIENCE EVER!!!", "rating": 1, "author": "angry_user"}
        ]
        
        # Create the directory if it doesn't exist
        try:
            output_dir = r"C:\Users\zhang\Desktop\review-Alabama.json"
            os.makedirs(output_dir, exist_ok=True)
            print(f"Ensured directory exists: {output_dir}")
            
            with open(path, "w", encoding="utf-8") as f:
                json.dump(sample_data, f, ensure_ascii=False, indent=2)
            print(f"Sample file created at: {path}")
        except (PermissionError, OSError) as e:
            print(f"Failed to create sample file: {e}")
            raise PermissionError(f"Unable to create sample file: {e}")

    # Try multiple reading strategies for permission issues
    try:
        print(f"DEBUG: Opening file: {path}")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        print(f"DEBUG: Successfully read file. Content length: {len(raw)} characters")
    except PermissionError:
        print(f"ERROR: Permission denied accessing file: {path}")
        # Try changing permissions
        try:
            import stat
            print("Attempting to change file permissions...")
            os.chmod(path, stat.S_IREAD | stat.S_IWRITE | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()
            print(f"DEBUG: Successfully read file after permission change. Content length: {len(raw)} characters")
        except Exception as perm_error:
            # Try alternative paths
            alternative_paths = [
                os.path.join(os.getcwd(), os.path.basename(path)),
                os.path.join(os.path.expanduser("~"), os.path.basename(path)),
                os.path.join(os.environ.get("TEMP", os.getcwd()), os.path.basename(path))
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    try:
                        with open(alt_path, "r", encoding="utf-8", errors="ignore") as f:
                            raw = f.read()
                        print(f"DEBUG: Successfully read from alternative path: {alt_path}")
                        break
                    except Exception:
                        continue
            else:
                raise PermissionError(f"Cannot access file '{path}' or any alternatives: {perm_error}")
    except Exception as e:
        print(f"ERROR: Error reading file '{path}': {e}")
        raise FileNotFoundError(f"Error reading file '{path}': {e}")

    try:
        obj = _try_parse_any(raw)
        print(f"DEBUG: Successfully parsed file content. Type: {type(obj)}")
    except Exception as parse_error:
        print(f"ERROR: Parse error: {parse_error}")
        print(f"DEBUG: First 200 characters of raw content: {raw[:200]}")
        raise ValueError(f"Unable to parse the input file: {parse_error}")
    
    if isinstance(obj, dict):
        print("DEBUG: Parsed as single dictionary object")
        return [obj]
    elif isinstance(obj, list):
        # Either a list of dicts or list of anything; keep dicts only
        dict_objects = [x for x in obj if isinstance(x, dict)]
        print(f"DEBUG: Parsed as list. Found {len(dict_objects)} dictionary objects out of {len(obj)} total items")
        return dict_objects
    else:
        raise ValueError(f"Parsed content is neither an object nor a list of objects. Got type: {type(obj)}")

def classify_review(review_obj: Dict[str, Any], review_index: int, total_reviews: int, retries: int = 3, backoff: float = 2.0) -> str:
    """
    Calls OpenAI to classify a single review object.
    Reports progress during classification.
    Returns the label string.
    
    ERROR HANDLING:
    - If all API attempts fail, returns "Error" label
    - If review content is corrupted/empty, returns "Error" label  
    - If model returns invalid label, returns "Error" label
    - If JSON parsing fails, returns "Error" label
    - All errors are logged with detailed information
    """
    # Check for corrupted/empty review content first
    if not review_obj or not isinstance(review_obj, dict):
        print(f"ERROR: [{review_index}/{total_reviews}] Invalid review object: {type(review_obj)}")
        return "Error"
    
    review_text = review_obj.get('text', str(review_obj))
    if not review_text or len(str(review_text).strip()) == 0:
        print(f"ERROR: [{review_index}/{total_reviews}] Empty or missing review text")
        return "Error"
    
    # Keep the review compact in the prompt
    try:
        review_json = json.dumps(review_obj, ensure_ascii=False)
        review_text_preview = str(review_text)[:100]
        print(f"\nPROGRESS: [{review_index}/{total_reviews}] Classifying review: {review_text_preview}...")
    except Exception as e:
        print(f"ERROR: [{review_index}/{total_reviews}] Failed to serialize review object: {e}")
        return "Error"

    for attempt in range(1, retries + 1):
        try:
            print(f"PROGRESS: [{review_index}/{total_reviews}] API call attempt {attempt}...")
            resp = client.chat.completions.create(
                model="gpt-4o-mini",         # using a valid model
                temperature=0,               # deterministic
                max_tokens=150,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Review object:\n{review_json}"}
                ],
            )
            content = resp.choices[0].message.content.strip()
            print(f"DEBUG: API response: {content}")
            
            # Strict JSON parse
            try:
                out = json.loads(content)
                print(f"DEBUG: Parsed JSON response: {out}")
            except json.JSONDecodeError as json_error:
                print(f"ERROR: [{review_index}/{total_reviews}] JSON parse error on attempt {attempt}: {json_error}")
                print(f"DEBUG: Raw API content: {content}")
                if attempt == retries:
                    print(f"ERROR: [{review_index}/{total_reviews}] All attempts failed due to JSON parsing errors")
                    return "Error"
                continue
            
            label = out.get("label", "").strip()
            print(f"PROGRESS: [{review_index}/{total_reviews}] ✓ Classified as: {label}")

            # Validate label
            if label not in ALLOWED_LABELS:
                print(f"ERROR: [{review_index}/{total_reviews}] Invalid label '{label}' on attempt {attempt}. Allowed: {ALLOWED_LABELS}")
                if attempt == retries:
                    print(f"ERROR: [{review_index}/{total_reviews}] All attempts returned invalid labels")
                    return "Error"
                continue

            return label

        except Exception as e:
            error_type = type(e).__name__
            print(f"ERROR: [{review_index}/{total_reviews}] Attempt {attempt} failed: {error_type}: {e}")
            
            # Log specific error details for debugging
            if "rate_limit" in str(e).lower():
                print(f"ERROR: [{review_index}/{total_reviews}] Rate limit exceeded - will retry with longer backoff")
                time.sleep(backoff * attempt * 2)  # Double backoff for rate limits
            elif "quota" in str(e).lower():
                print(f"FATAL: [{review_index}/{total_reviews}] API quota exceeded - cannot continue")
                return "Error"
            elif "authentication" in str(e).lower() or "unauthorized" in str(e).lower():
                print(f"FATAL: [{review_index}/{total_reviews}] Authentication error - check API key")
                return "Error"
            elif "timeout" in str(e).lower():
                print(f"ERROR: [{review_index}/{total_reviews}] Request timeout - will retry")
            else:
                print(f"ERROR: [{review_index}/{total_reviews}] Unexpected error: {e}")
            
            if attempt == retries:
                # On final failure, surface a fallback result
                print(f"FATAL: [{review_index}/{total_reviews}] All {retries} attempts failed. Returning 'Error' label.")
                print(f"FATAL: [{review_index}/{total_reviews}] Final error was: {error_type}: {e}")
                return "Error"
            
            print(f"PROGRESS: [{review_index}/{total_reviews}] Waiting {backoff * attempt} seconds before retry...")
            time.sleep(backoff * attempt)

def get_safe_output_path(original_path: str) -> str:
    """
    Ensures the output directory exists and returns a writable path.
    """
    try:
        # Extract directory from the original path
        output_dir = os.path.dirname(original_path)
        
        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        print(f"DEBUG: Ensured output directory exists: {output_dir}")
        
        # Test if we can write to this location
        test_file = os.path.join(output_dir, f"test_write_{int(time.time())}.tmp")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        
        print(f"DEBUG: Confirmed write access to: {original_path}")
        return original_path
        
    except (PermissionError, OSError) as e:
        print(f"ERROR: Cannot write to original path: {e}")
        
        # Fallback to alternative paths
        possible_paths = [
            os.path.join(os.getcwd(), os.path.basename(original_path)),
            os.path.join(os.path.expanduser("~"), os.path.basename(original_path)),
            os.path.join(os.environ.get("TEMP", os.getcwd()), os.path.basename(original_path)),
            f"review-Alabama_labeled_{int(time.time())}.jsonl"  # Final fallback with timestamp
        ]
        
        for path in possible_paths:
            try:
                # Test if we can write to this location
                test_dir = os.path.dirname(path) if os.path.dirname(path) else os.getcwd()
                os.makedirs(test_dir, exist_ok=True)
                
                # Test write access by creating a temporary file
                test_file = os.path.join(test_dir, f"test_write_{int(time.time())}.tmp")
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
                
                print(f"DEBUG: Selected alternative output path: {path}")
                return path
            except (PermissionError, OSError):
                print(f"DEBUG: Cannot write to: {path}")
                continue
        
        raise PermissionError("No writable location found for output file")

def main():
    print("DEBUG: Starting main function...")
    print(f"DEBUG: Working directory: {os.getcwd()}")
    print(f"DEBUG: Input path: {INPUT_PATH}")
    print(f"DEBUG: Output path: {OUTPUT_PATH}")
    
    try:
        reviews = load_reviews(INPUT_PATH)
        print(f"SUCCESS: Loaded {len(reviews)} review object(s).")
    except Exception as e:
        print(f"FATAL ERROR: Failed to load reviews: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Process first 10000 reviews
    original_count = len(reviews)
    reviews = reviews[:10000]
    print(f"DEBUG: Processing first {len(reviews)} reviews (out of {original_count} total).")

    # Classify with detailed progress reporting and error tracking
    labeled = []
    error_count = 0
    print(f"\n{'='*60}")
    print(f"STARTING CLASSIFICATION OF {len(reviews)} REVIEWS")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    for i, r in enumerate(reviews, 1):
        try:
            label = classify_review(r, i, len(reviews))
            labeled.append({"index": i, "label": label, "raw": r})
            
            # Track errors for summary
            if label == "Error":
                error_count += 1
                print(f"WARNING: [{i}/{len(reviews)}] Review {i} was labeled as 'Error' due to classification failure")
            
            # Calculate and show progress
            elapsed_time = time.time() - start_time
            avg_time_per_review = elapsed_time / i
            estimated_remaining = (len(reviews) - i) * avg_time_per_review
            
            print(f"PROGRESS: [{i}/{len(reviews)}] ✓ Complete | Label: {label} | Elapsed: {elapsed_time:.1f}s | ETA: {estimated_remaining:.1f}s")
            
        except Exception as e:
            print(f"FATAL ERROR: Failed to classify review {i}: {type(e).__name__}: {e}")
            labeled.append({"index": i, "label": "Error", "raw": r})
            error_count += 1

    # Save JSONL with improved error handling
    print(f"\n{'='*60}")
    print("SAVING RESULTS...")
    print(f"{'='*60}")
    
    try:
        safe_output_path = get_safe_output_path(OUTPUT_PATH)
        
        print(f"PROGRESS: Writing {len(labeled)} labeled reviews to: {safe_output_path}")
        with open(safe_output_path, "w", encoding="utf-8") as f:
            for row in labeled:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        
        print(f"SUCCESS: ✓ Saved labeled results to: {safe_output_path}")
        print(f"DEBUG: File size: {os.path.getsize(safe_output_path)} bytes")
        
    except Exception as e:
        print(f"FATAL ERROR: Cannot save results: {e}")
        # As absolute last resort, print results to console
        print("Printing results to console as fallback:")
        for row in labeled:
            print(json.dumps(row, ensure_ascii=False))

    # Show final summary with detailed label counts and error analysis
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    try:
        from collections import Counter
        counts = Counter(x["label"] for x in labeled)
        print("Label distribution:")
        for label, count in counts.items():
            percentage = (count / len(labeled)) * 100
            if label == "Error":
                print(f"  {label}: {count} ({percentage:.1f}%) ⚠️  - Classification failures")
            else:
                print(f"  {label}: {count} ({percentage:.1f}%)")
        
        print(f"\nError analysis:")
        print(f"  Total errors: {error_count}/{len(labeled)} ({(error_count/len(labeled)*100):.1f}%)")
        if error_count > 0:
            print(f"  ⚠️  {error_count} reviews could not be properly classified")
            print(f"  These may need manual review or reprocessing")
        else:
            print(f"  ✓ All reviews were successfully classified")
        
        total_time = time.time() - start_time
        print(f"\nProcessing statistics:")
        print(f"  Total reviews processed: {len(labeled)}")
        print(f"  Successfully classified: {len(labeled) - error_count}")
        print(f"  Failed classifications: {error_count}")
        print(f"  Success rate: {((len(labeled) - error_count)/len(labeled)*100):.1f}%")
        print(f"  Total time elapsed: {total_time:.1f} seconds")
        print(f"  Average time per review: {total_time / len(labeled):.1f} seconds")
        
    except Exception as e:
        print(f"ERROR: Failed to generate summary: {e}")

    print(f"\n{'='*60}")
    print("CLASSIFICATION COMPLETE!")
    if error_count > 0:
        print(f"⚠️  WARNING: {error_count} reviews were labeled as 'Error'")
        print("These reviews may need manual review or reprocessing.")
    print(f"{'='*60}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nINTERRUPTED: Script stopped by user (Ctrl+C)")
    except Exception as e:
        print(f"FATAL ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")

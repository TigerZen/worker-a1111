import time
import runpod
import requests
from requests.adapters import HTTPAdapter, Retry
import json # For logging payloads if needed

# Ensure this is the correct API URL for your AUTOMATIC1111 instance on RunPod
LOCAL_URL = "http://127.0.0.1:3000/sdapi/v1"

automatic_session = requests.Session()
# Configure retries for robustness, especially during startup or heavy load
# Increased total retries and adjusted backoff_factor
retries_config = Retry(total=15, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
automatic_session.mount('http://', HTTPAdapter(max_retries=retries_config))
automatic_session.mount('https://', HTTPAdapter(max_retries=retries_config)) # If API ever uses HTTPS locally

# ---------------------------------------------------------------------------- #
#                         AUTOMATIC1111 API Functions                          #
# ---------------------------------------------------------------------------- #

def wait_for_service(url, service_name="AUTOMATIC1111 WebUI API"):
    """
    Check if the service is ready to receive requests.
    """
    retries_count = 0
    log_interval = 15  # Log every N retries
    check_interval_seconds = 0.5

    print(f"Waiting for {service_name} at {url} to be ready...")
    while True:
        try:
            # Use a lightweight endpoint for health check. /progress or /config are good.
            # /sdapi/v1/memory or /sdapi/v1/progress?skip_current_image=true are also options.
            response = automatic_session.get(f"{LOCAL_URL}/progress?skip_current_image=true", timeout=10)
            response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
            print(f"{service_name} is ready.")
            return
        except requests.exceptions.RequestException as e:
            retries_count += 1
            if retries_count == 1 or retries_count % log_interval == 0:
                # Log only periodically to avoid spamming logs
                print(f"{service_name} not ready yet (attempt {retries_count}). Retrying... Error: {type(e).__name__}")
        except Exception as err: # Catch any other unexpected error during check
            retries_count += 1
            if retries_count == 1 or retries_count % log_interval == 0:
                print(f"{service_name} not ready yet (attempt {retries_count}). Unexpected error: {err}. Retrying...")
        
        time.sleep(check_interval_seconds)

def set_sd_model_checkpoint(model_title_or_filename):
    """
    Sets the Stable Diffusion checkpoint model in AUTOMATIC1111.
    :param model_title_or_filename: The title or filename of the model (e.g., 'model.safetensors').
                                   This file must exist in 'models/Stable-diffusion'.
    """
    print(f"Attempting to set SD model checkpoint to: {model_title_or_filename}")
    payload = {"sd_model_checkpoint": model_title_or_filename}
    
    try:
        response = automatic_session.post(url=f'{LOCAL_URL}/options', json=payload, timeout=60)
        response.raise_for_status()
        
        # Optional: Verify the change by fetching current options
        time.sleep(1) # Give A1111 a moment to process the change
        current_options = automatic_session.get(url=f'{LOCAL_URL}/options', timeout=30).json()
        new_checkpoint = current_options.get('sd_model_checkpoint')
        
        if new_checkpoint and (model_title_or_filename in new_checkpoint): # Check if the name is part of the returned title (which might include hash)
            print(f"Successfully set SD model checkpoint. Current API reports: {new_checkpoint}")
        else:
            print(f"Warning: SD model checkpoint may not have been set as expected, or verification failed.")
            print(f"  Requested: {model_title_or_filename}")
            print(f"  API reports: {new_checkpoint}")
            print(f"  Ensure '{model_title_or_filename}' is the correct name recognized by A1111 and is present in 'models/Stable-diffusion'.")

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP Error setting SD model checkpoint to '{model_title_or_filename}': {http_err.response.status_code}")
        print(f"Response body: {http_err.response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Request Error setting SD model checkpoint to '{model_title_or_filename}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred while setting model checkpoint '{model_title_or_filename}': {e}")

def run_sd_inference(inference_payload):
    """
    Run inference (txt2img or img2img) via AUTOMATIC1111 API.
    :param inference_payload: Dictionary matching A1111's API schema.
    """
    # Determine API endpoint based on presence of "init_images" for img2img
    api_endpoint = "txt2img"
    if "init_images" in inference_payload and inference_payload["init_images"]:
        api_endpoint = "img2img"
    
    full_api_url = f'{LOCAL_URL}/{api_endpoint}'
    print(f"Sending request to AUTOMATIC1111 API: {full_api_url}")
    # For debugging, you can print the payload, but be mindful of sensitive data or large base64 images.
    # print(f"Payload: {json.dumps(inference_payload, indent=2, default=str)}")

    response = automatic_session.post(url=full_api_url, json=inference_payload, timeout=600) # 10 min timeout
    response.raise_for_status()  # Raise an exception for HTTP error codes
    return response.json()

# ---------------------------------------------------------------------------- #
#                               RunPod Handler                               #
# ---------------------------------------------------------------------------- #
def handler(event):
    """
    This is the handler function called by RunPod serverless.
    """
    job_input = event["input"]
    
    # --- LoRA Injection ---
    # Expects job_input["loras"] = [{"name": "lora_filename_NO_EXTENSION", "weight": 0.7}, ...]
    # LoRa files must be in AUTOMATIC1111's 'models/Lora' directory.
    loras_to_apply = job_input.pop("loras", [])  # Get LoRAs and remove from main payload
    
    current_prompt = job_input.get("prompt", "") # Get existing prompt or default to empty string

    if loras_to_apply:
        lora_prompt_segment = ""
        for lora_entry in loras_to_apply:
            lora_name = lora_entry.get("name")
            lora_weight = lora_entry.get("weight")
            if lora_name and isinstance(lora_weight, (int, float)):
                lora_prompt_segment += f" <lora:{lora_name}:{lora_weight}>"
            else:
                print(f"Warning: Invalid or incomplete LoRA entry skipped: {lora_entry}")
        
        if lora_prompt_segment:
            job_input["prompt"] = current_prompt + lora_prompt_segment
            print(f"Prompt modified with LoRAs. New prompt: \"{job_input['prompt']}\"")
    # --- End LoRA Injection ---

    # --- Apply Default A1111 Payload Values (Optional but Recommended) ---
    # These help ensure the API call doesn't fail if common fields are missing.
    # Customize these defaults as per your typical needs.
    defaults = {
        "steps": 20,
        "sampler_name": "Euler a", # Or "sampler_index"
        "cfg_scale": 7.0,
        "width": 512,
        "height": 512,
        "seed": -1, # -1 for random seed
        "negative_prompt": "",
        # Add other common parameters if needed
    }
    for key, value in defaults.items():
        job_input.setdefault(key, value)
    # --- End Default Values ---

    try:
        api_response = run_sd_inference(job_input)
        # The response from A1111 API usually contains images as base64 strings
        # and an 'info' string (JSON parsable).
        # This script returns the raw JSON response from A1111.
        # You might want to process it further, e.g., upload images to a bucket.
        return api_response
    except requests.exceptions.HTTPError as http_err:
        error_message = f"AUTOMATIC1111 API HTTP Error: {http_err}"
        status_code = 500
        if http_err.response is not None:
            error_message += f" - Status: {http_err.response.status_code} - Response: {http_err.response.text}"
            status_code = http_err.response.status_code
        print(error_message)
        return {"error": error_message, "status_code": status_code}
    except requests.exceptions.RequestException as req_err: # Handles network issues, timeouts etc.
        error_message = f"AUTOMATIC1111 API Request Error: {req_err}"
        print(error_message)
        return {"error": error_message, "status_code": 503} # Service Unavailable
    except Exception as e: # Catch-all for other unexpected errors
        error_message = f"An unexpected error occurred in the handler: {e}"
        import traceback
        print(traceback.format_exc()) # Print full stack trace for debugging
        return {"error": error_message, "status_code": 500}

# ---------------------------------------------------------------------------- #
#                               Main Execution                               #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("Starting AUTOMATIC1111 RunPod Serverless Handler...")
    
    # 1. Wait for the AUTOMATIC1111 API service to be ready.
    wait_for_service(url=LOCAL_URL, service_name="AUTOMATIC1111 WebUI API")
    
    # 2. Set the desired Stable Diffusion Checkpoint Model.
    #    IMPORTANT: The model file 'Gemini_ILMixV5.safetensors' MUST be present in
    #    the AUTOMATIC1111 'models/Stable-diffusion' directory on the RunPod worker.
    #    The name here should match how A1111 lists the model (usually the filename).
    checkpoint_model_filename = "Gemini_ILMixV5.safetensors" 
    set_sd_model_checkpoint(checkpoint_model_filename)
    
    print("AUTOMATIC1111 setup complete. Starting RunPod Serverless worker...")
    runpod.serverless.start({"handler": handler})

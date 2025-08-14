from ..smp import *
import os
import sys
from .base import BaseAPI
import aiohttp
import asyncio

APIBASES = {
    # 'OFFICIAL': 'https://api.openai.com/v1/chat/completions',
    'XIAOHU': "https://xiaohumini.site/v1/chat/completions",
    'HUOSHAN': "https://ark.cn-beijing.volces.com/api/v3",
    'ALI':"https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    'LOCAL': 'http://0.0.0.0:2333/v1/chat/completions'
}


def GPT_context_window(model):
    length_map = {
        'gpt-4': 8192,
        'gpt-4-0613': 8192,
        'gpt-4-turbo-preview': 128000,
        'gpt-4-1106-preview': 128000,
        'gpt-4-0125-preview': 128000,
        'gpt-4-vision-preview': 128000,
        'gpt-4-turbo': 128000,
        'gpt-4-turbo-2024-04-09': 128000,
        'gpt-3.5-turbo': 16385,
        'gpt-3.5-turbo-0125': 16385,
        'gpt-3.5-turbo-1106': 16385,
        'gpt-3.5-turbo-instruct': 4096,
    }
    if model in length_map:
        return length_map[model]
    else:
        return 128000


import os
import json
import asyncio # Import asyncio
import aiohttp # Import aiohttp
import numpy as np
import time
# No longer need concurrent.futures or threading Lock for basic async IO
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from threading import Lock

# Assuming BaseAPI, encode_image_to_base64, APIBASES, etc. are defined elsewhere
# Mocking necessary components for demonstration if not present
class BaseAPI: # Mock BaseAPI if not defined
    def __init__(self, wait, retry, system_prompt, verbose, **kwargs):
        self.wait = wait
        self.retry = retry
        self.system_prompt = system_prompt
        self.verbose = verbose
        import logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

def encode_image_to_base64(img, target_size=-1): # Mock image encoding
    import base64
    from io import BytesIO
    buffered = BytesIO()
    img_byte_arr = b"mock_image_bytes" # Placeholder
    buffered.write(img_byte_arr)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

APIBASES = { # Mock API Bases
    'OFFICIAL': 'https://api.openai.com/v1/chat/completions',
    # Add other bases
}

from PIL import Image
from io import BytesIO
original_image_open = Image.open
def mock_image_open(path):
    try:
        return original_image_open(path)
    except FileNotFoundError:
        mock_img = Image.new('RGB', (600, 400))
        mock_img.size = (600, 400)
        return mock_img
Image.open = mock_image_open

class OpenAIWrapper(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = 'gpt-3.5-turbo-0613',
                 retry: int = 5, # Retry logic needs adaptation for async
                 wait: int = 5,  # Wait logic needs adaptation for async
                 key: str = None,
                 verbose: bool = False,
                 system_prompt: str = None,
                 temperature: float = 0,
                 timeout: int = 60, # aiohttp uses ClientTimeout
                 api_base: str = None,
                 max_tokens: int = 1024,
                 img_size: int = 512,
                 img_detail: str = 'low',
                 use_azure: bool = False,
                 **kwargs):

        # --- Most initializations remain the same ---
        self.model = model
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_azure = use_azure
        self.timeout_config = aiohttp.ClientTimeout(total=timeout) # Configure timeout for aiohttp

        # --- Key and API Base determination logic (remains the same logic) ---
        # (Simplified for brevity - assume this part correctly sets self.key)
        if 'step' in model:
            env_key = os.environ.get('STEPAI_API_KEY', '')
            if key is None: key = env_key
        # ... other key logic ...
        elif 'doubao' in model:
             env_key = os.environ.get('DOUBAO_API_KEY', '')
             if key is None: key = env_key
        elif 'qwen' in model:
             env_key = os.environ.get('DASHSCOPE_API_KEY', '')
             if key is None: key = env_key
        # ... rest of key logic ...
        else: # Default OpenAI / Azure
            if use_azure:
                env_key = os.environ.get('AZURE_OPENAI_API_KEY', None)
                # ... azure key checks ...
                if key is None: key = env_key
            else:
                env_key = os.environ.get('OPENAI_API_KEY', '')
                # ... openai key checks ...
                if key is None: key = env_key
        self.key = key
        assert self.key is not None, "API Key could not be determined."

        assert img_size > 0 or img_size == -1
        self.img_size = img_size
        assert img_detail in ['high', 'low']
        self.img_detail = img_detail

        # Initialize BaseAPI (might need adjustment if BaseAPI has async needs)
        super().__init__(wait=wait, retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

        # --- API Base setup logic (remains the same logic) ---
        # (Simplified for brevity - assume this part correctly sets self.api_base)
        if use_azure:
            # ... Azure setup ...
             api_base_template = ('{endpoint}openai/deployments/{deployment_name}/chat/completions?api-version={api_version}')
             # ... get endpoint, deployment_name, api_version ...
             self.api_base = api_base_template.format(...) # Fill details
        else:
            if api_base is None:
                # ... logic to determine api_base from model/env ...
                 if 'doubao' in model: api_base = os.environ.get('DOUBAO_API_BASE')
                 elif 'qwen' in model: api_base = os.environ.get('DASHSCOPE_API_BASE')
                 # ... etc ...
                 else: api_base = 'OFFICIAL'
            # ... logic to resolve api_base identifier to URL ...
            if api_base in APIBASES: self.api_base = APIBASES[api_base]
            elif api_base and api_base.startswith('http'): self.api_base = api_base
            else: raise ValueError(f"Invalid API Base: {api_base}")
        assert self.api_base is not None, "API Base could not be determined."


        # --- Session Management ---
        # Session will be created in __aenter__
        self._session: aiohttp.ClientSession | None = None

        # Logging at the end of __init__ (no lock needed for print)
        print(f'--------------------------------------------------------------\n[!] Initialized Wrapper. API Base: {self.api_base}; Key Loaded.\n-------------------------------------------------------------------------')
        self.logger.info(f'Initialized Wrapper. API Base: {self.api_base}')

    # --- Async Context Manager for session handling ---
    async def __aenter__(self):
        """Creates the aiohttp session."""
        if self._session is None or self._session.closed:
             # You might want to customize connector limits here if needed
             # connector = aiohttp.TCPConnector(limit_per_host=...)
             self._session = aiohttp.ClientSession(timeout=self.timeout_config)
             self.logger.info("aiohttp ClientSession created.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Closes the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self.logger.info("aiohttp ClientSession closed.")
            self._session = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensures the session exists, useful if not using 'async with'."""
        if self._session is None or self._session.closed:
             self._session = aiohttp.ClientSession(timeout=self.timeout_config)
             self.logger.info("aiohttp ClientSession created (explicitly).")
        return self._session

    async def close_session(self):
        """Explicitly closes the session if not using 'async with'."""
        await self.__aexit__(None, None, None)

    # --- Input preparation methods remain synchronous ---
    # Note: Image.open is sync I/O, could block event loop.
    # Consider loop.run_in_executor for true async image loading if needed.
    def prepare_itlist(self, inputs):
        # (Logic remains the same as before)
        assert np.all([isinstance(x, dict) for x in inputs])
        has_images = np.sum([x['type'] == 'image' for x in inputs])
        content_list = []
        if has_images:
            for msg in inputs:
                if msg['type'] == 'text':
                    content_list.append(dict(type='text', text=msg['value']))
                elif msg['type'] == 'image':
                    try:
                        # WARNING: Image.open is synchronous file I/O
                        img = Image.open(msg['value'])
                        b64 = encode_image_to_base64(img, target_size=self.img_size)
                        img_struct = dict(url=f'data:image/jpeg;base64,{b64}', detail=self.img_detail)
                        content_list.append(dict(type='image_url', image_url=img_struct))
                    except Exception as e:
                         self.logger.error(f"Error processing image {msg.get('value', 'N/A')}: {e}")
                         content_list.append(dict(type='text', text='[Image Error]'))
        else:
            assert all([x['type'] == 'text' for x in inputs])
            text = '\n'.join([x['value'] for x in inputs])
            content_list = [dict(type='text', text=text)]
        return content_list

    def prepare_inputs(self, inputs):
        # (Logic remains the same as before)
        input_msgs = []
        if self.system_prompt is not None:
            input_msgs.append(dict(role='system', content=self.system_prompt))
        assert isinstance(inputs, list) and len(inputs) > 0, "Inputs must be a non-empty list."

        if isinstance(inputs[0], dict) and 'role' in inputs[0]:
            assert inputs[-1]['role'] == 'user', f"Last message role must be 'user', got: {inputs[-1].get('role')}"
            for item in inputs:
                 # Ensure content is a list for prepare_itlist
                 content_input = item['content'] if isinstance(item['content'], list) else [{'type': 'text', 'value': item['content']}]
                 try:
                     prepared_content = self.prepare_itlist(content_input)
                     input_msgs.append(dict(role=item['role'], content=prepared_content))
                 except Exception as e:
                     self.logger.error(f"Error preparing content for role {item['role']}: {e}")
                     # Handle error appropriately, e.g., skip message or add error placeholder
                     input_msgs.append(dict(role=item['role'], content=[{'type':'text', 'text': '[Content Preparation Error]'}]))

        elif isinstance(inputs[0], dict) and 'type' in inputs[0]:
             input_msgs.append(dict(role='user', content=self.prepare_itlist(inputs)))
        else:
             self.logger.warning("Input format unrecognized. Assuming list of strings for single user message.")
             input_msgs.append(dict(role='user', content=[{'type':'text', 'text':'\n'.join(map(str, inputs))}]))

        return input_msgs

    # --- Core request sending method is now async ---
    async def _send_single_request(self, input_msgs, **kwargs) -> tuple:
        """Sends a single prepared request using aiohttp."""
        session = await self._ensure_session() # Ensure session is active

        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        specific_kwargs = kwargs.get('api_specific_kwargs', {})

        # --- Prepare Headers and Payload (logic similar to sync version) ---
        headers = {}
        payload = {}
        target_url = "" # Determine target_url based on model/config

        if self.use_azure:
            target_url = self.api_base # Azure URL from init
            headers = {'Content-Type': 'application/json', 'api-key': self.key}
            payload = dict(messages=input_msgs, max_tokens=max_tokens, temperature=temperature, n=1, **specific_kwargs)
        elif 'internvl2-pro' in self.model:
            target_url = self.api_base
            headers = {'Content-Type': 'application/json', 'Authorization': self.key}
            payload = dict(model=self.model, messages=input_msgs, max_tokens=max_tokens, temperature=temperature, **specific_kwargs)
        elif 'doubao' in self.model:
            target_url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions" # Doubao endpoint
            headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.key}'}
            payload = dict(model='doubao-1-5-vision-pro-32k-250115', messages=input_msgs, temperature=temperature, **specific_kwargs)
        else: # Default OpenAI / Compatible
            target_url = self.api_base
            headers = {'Accept': 'application/json', 'Content-Type': 'application/json', 'Authorization': f'Bearer {self.key}'}
            payload = dict(model=self.model, messages=input_msgs, max_tokens=max_tokens, n=1, temperature=temperature, **specific_kwargs)

        response_text = ""
        response_status = -1
        response_obj = None # Store the aiohttp response temporarily if needed

        try:
            # Use session.post with async with for context management
            async with session.post(target_url, headers=headers, json=payload) as response:
                response_status = response.status
                response_obj = response # Keep reference if needed during handling
                # Check if the request was successful (2xx status code)
                if 200 <= response_status < 300:
                    try:
                        # Try to parse JSON response
                        resp_struct = await response.json()
                        # Extract answer (handle variations)
                        if 'choices' in resp_struct and len(resp_struct['choices']) > 0:
                            message = resp_struct['choices'][0].get('message', {})
                            answer = message.get('content', '').strip()
                            if not answer and 'tool_calls' in message:
                                answer = json.dumps(message['tool_calls']) # Example tool call handling
                            elif not answer:
                                answer = "[API returned empty content]"
                        else: # Handle other potential success formats
                            answer = json.dumps(resp_struct) # Fallback: return full JSON

                        # Return successful result: status, answer, raw JSON structure
                        return response_status, answer, resp_struct

                    except (json.JSONDecodeError, aiohttp.ContentTypeError) as json_err:
                        # Handle cases where response is success but not valid JSON
                        response_text = await response.text()
                        self.logger.warning(f"API success status {response_status} but failed to decode JSON: {json_err}. Response text: {response_text[:500]}")
                        # Decide return format for non-JSON success
                        return response_status, f"[API Success, Non-JSON Response]: {response_text[:200]}", response_text
                else:
                    # Handle non-2xx status codes (errors)
                    response_text = await response.text()
                    error_message = f"{self.fail_msg} Status Code: {response_status}."
                    try: # Try to get specific error message from JSON body
                        error_details = json.loads(response_text) # Manual loads needed here
                        error_message += f" Details: {error_details.get('error', {}).get('message', response_text[:500])}"
                    except json.JSONDecodeError:
                        error_message += f" Response: {response_text[:500]}"
                    self.logger.error(error_message)
                    # Return error result: status, message, raw text
                    return response_status, error_message, response_text

        # Handle specific aiohttp/asyncio errors
        except asyncio.TimeoutError:
             self.logger.error(f"Request timed out after {self.timeout_config.total} seconds.")
             return 408, f"{self.fail_msg} Request Timeout.", None # Use standard 408 code
        except aiohttp.ClientConnectorError as conn_err:
             self.logger.error(f"Connection Error: {conn_err}")
             return 503, f"{self.fail_msg} Connection Error: {conn_err}", None # Service unavailable?
        except aiohttp.ClientError as client_err: # Catch other client-side errors
             self.logger.error(f"aiohttp Client Error: {type(client_err).__name__}: {client_err}")
             return 500, f"{self.fail_msg} Client Error: {client_err}", None
        except Exception as e: # Catch unexpected errors
             self.logger.error(f"Unexpected error during API call: {type(e).__name__}: {e}")
             # Try to get response text if available
             err_resp_text = response_text if response_text else "N/A"
             return 500, f"{self.fail_msg} Unexpected Error: {e}. Response: {err_resp_text[:200]}", err_resp_text


    # --- Single request method (async) ---
    async def generate(self, inputs, **kwargs) -> tuple:
         """Prepares inputs and sends a single asynchronous request."""
         # Note: Retry logic needs to be implemented asynchronously if desired
         # Simple retry example (could be more sophisticated):
         for attempt in range(self.retry + 1):
             try:
                 input_msgs = self.prepare_inputs(inputs)
                 # Token length calculation could be added here if needed
                 status, answer, response_data = await self._send_single_request(input_msgs, **kwargs)

                 # Check status code for success/failure
                 if 200 <= status < 300:
                     return status, answer, response_data # Success
                 else:
                      # Log retry attempt for specific error codes if needed
                     self.logger.warning(f"Attempt {attempt+1} failed with status {status}. Retrying in {self.wait}s...")
                     await asyncio.sleep(self.wait) # Async sleep

             except Exception as e:
                  # Handle exceptions during preparation or sending
                  self.logger.error(f"Error in generate (attempt {attempt+1}): {e}")
                  if attempt >= self.retry:
                      return 500, f"Failed after {self.retry+1} attempts: {e}", None
                  await asyncio.sleep(self.wait) # Wait before retrying on exception

         # If all retries fail
         return 500, f"{self.fail_msg} Failed after {self.retry + 1} attempts.", None


    # --- Batch request method (async) ---
    async def generate_batch(self, batch_inputs, **kwargs) -> list:
        """
        Generates responses for a batch of inputs concurrently using asyncio.gather.

        Args:
            batch_inputs (list): A list where each element is a valid input
                                 for `generate`.
            **kwargs: Additional keyword arguments passed to each API call.

        Returns:
            list: A list of tuples, where each tuple corresponds to an input
                  in the batch and contains (status_code, answer, response_data).
                  The order matches the order of `batch_inputs`.
                  Exceptions during individual calls are returned as part of the tuple.
        """
        if not batch_inputs:
            return []

        session = await self._ensure_session() # Ensure session is ready

        tasks = []
        for single_input in batch_inputs:
            # Create an asyncio task for each generate call
            # Pass kwargs down
            task = asyncio.create_task(self.generate(single_input, **kwargs))
            tasks.append(task)

        if self.verbose:
             self.logger.info(f"Starting concurrent generation for {len(batch_inputs)} inputs.")

        # Run all tasks concurrently and gather results
        # return_exceptions=True ensures that if one task fails, others continue,
        # and the exception object is returned in place of the result.
        results_or_exceptions = await asyncio.gather(*tasks, return_exceptions=True)

        if self.verbose:
             self.logger.info(f"Concurrent generation finished for {len(batch_inputs)} inputs.")

        # Process results to maintain the desired output format
        final_results = []
        for i, res_or_exc in enumerate(results_or_exceptions):
            if isinstance(res_or_exc, Exception):
                self.logger.error(f"Input {i+1}/{len(batch_inputs)} resulted in an exception: {res_or_exc}")
                # Format exception into the standard tuple format
                final_results.append((-1, f"Task Exception: {res_or_exc}", None)) # Use -1 for task errors
            else:
                # Result is already in the (status, answer, data) format from generate()
                final_results.append(res_or_exc)
                if self.verbose:
                     status, answer, _ = res_or_exc
                     short_answer = str(answer)[:70] + '...' if len(str(answer)) > 70 else str(answer)
                     self.logger.debug(f"Input {i+1}/{len(batch_inputs)} completed. Status: {status}, Answer: '{short_answer}'")


        return final_results


    # --- Token calculation methods remain synchronous ---
    # WARNING: These can block the event loop. Consider run_in_executor if they become bottlenecks.
    def get_image_token_len(self, img_path, detail='low'):
        # (Implementation remains the same)
        import math
        if detail == 'low': return 85
        try:
            # Sync file I/O
            im = Image.open(img_path)
            width, height = im.size
            if width == 0 or height == 0: return 85
            h = math.ceil(height / 512)
            w = math.ceil(width / 512)
            total = 85 + 170 * h * w
            return total
        except Exception as e:
             self.logger.warning(f"Could not calculate image token length for {img_path}: {e}")
             return 85

    def get_token_len(self, inputs) -> int:
         # (Implementation remains the same, using tiktoken)
         # Tiktoken is CPU bound, might block loop if inputs are huge
         import tiktoken
         try:
             if not hasattr(self, '_tokenizer_cache') or self.model not in self._tokenizer_cache:
                 if not hasattr(self, '_tokenizer_cache'): self._tokenizer_cache = {}
                 try: self._tokenizer_cache[self.model] = tiktoken.encoding_for_model(self.model)
                 except KeyError:
                      self.logger.warning(f"Model {self.model} not found in tiktoken, using 'cl100k_base'.")
                      self._tokenizer_cache[self.model] = tiktoken.get_encoding("cl100k_base")
             enc = self._tokenizer_cache[self.model]
         except Exception as err:
             self.logger.error(f"Failed to get tiktoken encoder for model {self.model}: {err}. Returning 0.")
             return 0

         if not isinstance(inputs, list): return 0 # Basic type check

         tot = 0
         # Simplified token calculation logic (adjust as needed for accuracy)
         for item in inputs:
             if not isinstance(item, dict): continue
             if 'role' in item and 'content' in item:
                  tot += 4 # Approx role tokens
                  content = item['content']
                  if isinstance(content, list): tot += self.get_token_len(content)
                  elif isinstance(content, str): tot += len(enc.encode(content))
             elif 'type' in item:
                 if item['type'] == 'text' and 'value' in item: tot += len(enc.encode(item['value']))
                 elif item['type'] == 'image' and 'value' in item: tot += self.get_image_token_len(item['value'], detail=self.img_detail)
                 tot += 4 # Approx type/value structure tokens
         tot += 3 # Approx message boundary tokens
         return tot


class GPT4V(OpenAIWrapper):

    def generate(self, message, dataset=None):
        return super(GPT4V, self).generate(message)



import json
import os
import subprocess
import time
from datetime import datetime
from openai import OpenAI
import tiktoken
from typing import List, Any, Dict, Union, Tuple, Callable
from tenacity import retry, wait_random_exponential, stop_after_attempt, RetryError

from Brain_modules.image_vision import ImageVision
from Brain_modules.tool_call_functions.web_research import WebResearchTool
from Brain_modules.tool_call_functions.call_expert import call_expert
from Brain_modules.tool_call_functions.file_directory_manager import file_directory_manager
from Brain_modules.define_tools import tools

MAX_TOKENS_PER_MINUTE = 5500
MAX_RETRIES = 3

def get_current_datetime():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class LLM_API_Calls:
    def __init__(self):
        self.client = None
        self.model = None
        self.current_api_provider = os.getenv('DEFAULT_API_PROVIDER', 'ollama')
        self.setup_client()
        self.image_vision = ImageVision()
        self.chat_history = []
        self.max_tokens = 4000
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.web_research_tool = WebResearchTool()
        self.tokens_used = 0
        self.start_time = time.time()
        self.available_functions = {
            "run_local_command": self.run_local_command,
            "web_research": self.web_research_tool.web_research,
            "analyze_image": self.analyze_image,
            "call_expert": call_expert,
            "file_directory_manager": file_directory_manager
        }
        self.rate_limit_remaining = MAX_TOKENS_PER_MINUTE
        self.rate_limit_reset = time.time() + 60

    def setup_client(self):
        try:
            self.client = self.choose_API_provider()
        except Exception as e:
            print(f"Error setting up client: {e}")
            raise

    def choose_API_provider(self):
        if self.current_api_provider == "OpenAI":
            return self.setup_openai_client()
        elif self.current_api_provider == "ollama":
            return self.setup_ollama_client()
        elif self.current_api_provider == "Groq":
            return self.setup_groq_client()
        else:
            raise ValueError(f"Unsupported LLM Provider: {self.current_api_provider}")

    def setup_openai_client(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        self.model = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
        return OpenAI(api_key=api_key)

    def setup_ollama_client(self):
        self.model = os.environ.get("OLLAMA_MODEL", "llama3:instruct")
        return OpenAI(base_url="http://10.11.1.18:11434/v1", api_key="ollama")

    def setup_groq_client(self):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Groq API key not found in environment variables")
        self.model = os.environ.get("GROQ_MODEL", "llama3-8b-8192")
        return OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)

    def update_api_provider(self, provider):
        self.current_api_provider = provider
        self.setup_client()
        
    def count_tokens(self, text):
        return len(self.encoding.encode(str(text)))

    def truncate_text(self, text, max_tokens):
        tokens = self.encoding.encode(str(text))
        return self.encoding.decode(tokens[:max_tokens]) if len(tokens) > max_tokens else text

    @retry(wait=wait_random_exponential(multiplier=1, max=90), stop=stop_after_attempt(MAX_RETRIES))
    def run_local_command(self, command, progress_callback=None):
        if progress_callback:
            progress_callback(f"Executing local command: {command}")
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
            output = result.stdout
            if progress_callback:
                progress_callback(f"Local command executed successfully")
            return {"command": command, "output": output, "datetime": get_current_datetime()}
        except subprocess.CalledProcessError as e:
            if progress_callback:
                progress_callback(f"Error executing local command: {e.stderr}")
            return {"command": command, "error": f"Command execution failed: {e.stderr}", "datetime": get_current_datetime()}
        except Exception as e:
            if progress_callback:
                progress_callback(f"Unexpected error during local command execution: {str(e)}")
            return {"command": command, "error": str(e), "datetime": get_current_datetime()}

    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(MAX_RETRIES))
    def analyze_image(self, image_url, progress_callback=None):
        if progress_callback:
            progress_callback(f"Analyzing image: {image_url}")
        try:
            description = self.image_vision.analyze_image(image_url)
            if progress_callback:
                progress_callback("Image analysis completed")
            return {"image_url": image_url, "description": description, "datetime": get_current_datetime()}
        except Exception as e:
            if progress_callback:
                progress_callback(f"Error during image analysis: {str(e)}")
            return {"error": str(e), "datetime": get_current_datetime()}

    def chat(self, prompt: str, system_message: str, tools: List[dict], progress_callback: Callable[[str], None] = None) -> Tuple[str, List[Any]]:
        try:
            return self._chat_loop(prompt, system_message, tools, progress_callback)
        except RetryError as e:
            error_message = f"Failed to get a response after {MAX_RETRIES} attempts: {str(e)}"
            if progress_callback:
                progress_callback(error_message)
            return error_message, []
        except Exception as e:
            error_message = f"Unexpected error in chat: {str(e)}"
            if progress_callback:
                progress_callback(error_message)
            return error_message, []

    def _construct_system_message(self) -> str:
        return """You are an AI assistant designed to provide helpful and informative responses. 
        Always respond in natural language, avoiding JSON or any other structured format in your final responses. 
        If you use any tools or perform any actions, incorporate the results into your response naturally. 
        Ensure your final response is a coherent paragraph or set of paragraphs that directly addresses the user's query or request."""

    def _chat_loop(self, prompt: str, system_message: str, tools: List[dict], progress_callback: Callable[[str], None] = None) -> Tuple[str, List[Any]]:
        messages = [{"role": "system", "content": self._construct_system_message()}]
        messages.extend(self.chat_history[-3:])
        
        user_prompt = f"{prompt}\n\nRemember to respond in natural language, not in JSON or any other structured format."
        messages.append({"role": "user", "content": user_prompt if len(user_prompt) < 1000 else user_prompt[:1000] + "... (truncated)"})

        response = self._chat_with_retry(messages, tools, progress_callback)
        
        content = response.get("content", "")
        tool_calls = response.get("tool_calls", [])

        if tool_calls:
            tool_responses = self._process_tool_calls(tool_calls, progress_callback)
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "function", "name": "tool_response", "content": json.dumps(tool_responses)})
            
            reflection_prompt = self._generate_reflection_prompt(prompt, content, tool_responses)
            messages.append({"role": "user", "content": reflection_prompt})
            
            final_response = self._chat_with_retry(messages, tools, progress_callback)
            content = final_response.get("content", "")

        self.chat_history.extend(messages[1:])  # Add all new messages except the system message
        return content, tool_calls

    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(MAX_RETRIES))
    def _chat_with_retry(self, messages: List[Dict[str, str]], tools: List[dict], progress_callback: Callable[[str], None] = None) -> Dict[str, Any]:
        self.reset_token_usage()

        if progress_callback:
            progress_callback("Sending request to language model")

        try:
            self.check_rate_limit()
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=self.max_tokens
            )
            
            self.update_rate_limit(getattr(response, 'headers', {}))
            response_message = response.choices[0].message

            if progress_callback:
                progress_callback("Received response from language model")

            content = response_message.content or ""
            tool_calls = response_message.tool_calls or []
            self.update_token_usage(messages, content)

            return {"content": content, "tool_calls": tool_calls}

        except Exception as e:
            error_message = f"Error in chat: {str(e)}"
            if progress_callback:
                progress_callback(f"Error occurred: {error_message}")
            print(error_message)
            raise

    def _process_tool_calls(self, tool_calls, progress_callback=None):
        tool_responses = []
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = self.available_functions.get(function_name)
            if function_to_call:
                try:
                    function_args = json.loads(tool_call.function.arguments)
                    function_response = function_to_call(**function_args, progress_callback=progress_callback)
                    tool_responses.append({
                        "tool_name": function_name,
                        "tool_response": function_response
                    })
                except Exception as e:
                    if progress_callback:
                        progress_callback(f"Error in {function_name}: {str(e)}")
                    tool_responses.append({
                        "tool_name": function_name,
                        "tool_response": {"error": str(e)}
                    })
            else:
                if progress_callback:
                    progress_callback(f"Function {function_name} not found")
        return tool_responses

    def _generate_reflection_prompt(self, initial_prompt, initial_response, tool_responses):
        tool_results = "\n".join([f"{resp['tool_name']}: {json.dumps(resp['tool_response'])}" for resp in tool_responses])
        reflection_prompt = f"""
        Initial user input: {initial_prompt}
       
        Your initial response: {initial_response}
        Tool usage results: {tool_results}
        
        Based on the initial user input, your initial response, and the results from the tools used, 
        please provide a comprehensive response. Consider how the tool results affect your understanding 
        of the user's request and how you can best address their needs. If you have enough information 
        to answer the user's query, provide a final response. If not, you may use additional tools or 
        ask for clarification.

        Remember to respond in natural language, not in JSON or any other structured format.
        Your reflection and response:
        """
        return self.truncate_text(reflection_prompt, self.max_tokens)

    def update_token_usage(self, messages, response):
        tokens_used = sum(self.count_tokens(msg["content"]) for msg in messages) + self.count_tokens(response)
        self.tokens_used += tokens_used
        self.rate_limit_remaining -= tokens_used

    def reset_token_usage(self):
        current_time = time.time()
        if current_time >= self.rate_limit_reset:
            self.tokens_used = 0
            self.rate_limit_remaining = MAX_TOKENS_PER_MINUTE
            self.rate_limit_reset = current_time + 60

    def check_rate_limit(self):
        if self.rate_limit_remaining <= 0:
            sleep_time = max(0, self.rate_limit_reset - time.time())
            time.sleep(sleep_time)
            self.reset_token_usage()

    def update_rate_limit(self, headers):
        remaining = headers.get('X-RateLimit-Remaining')
        reset = headers.get('X-RateLimit-Reset')
        
        if remaining is not None:
            try:
                self.rate_limit_remaining = int(remaining)
            except ValueError:
                self.rate_limit_remaining = MAX_TOKENS_PER_MINUTE
        
        if reset is not None:
            try:
                self.rate_limit_reset = float(reset)
            except ValueError:
                self.rate_limit_reset = time.time() + 60

llm_api_calls = LLM_API_Calls()

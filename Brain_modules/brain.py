import json
import time
from typing import Dict, Any, Callable, Tuple, List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from Brain_modules.llm_api_calls import llm_api_calls, tools
from Brain_modules.memory_utils import generate_embedding, add_to_memory, retrieve_relevant_memory
from Brain_modules.sentiment_analysis import analyze_sentiment
from Brain_modules.lobes_processing import LobesProcessing
from Brain_modules.final_agent_persona import FinalAgentPersona

class Brain:
    def __init__(self, progress_callback: Callable[[str], None], collection, collection_size):
        self.progress_callback = progress_callback
        self.collection = collection
        self.collection_size = collection_size
        self._initialize()

    def _initialize(self):
        self._log_progress("Initializing Brain")
        self.tts_enabled = True
        self.lobes_processing = LobesProcessing()
        self.embeddings_model = "mxbai-embed-large"
        self.chat_histories = {}
        self.last_response = ""
        self._log_progress("Brain initialization completed")

    def toggle_tts(self):
        try:
            self.tts_enabled = not self.tts_enabled
            status = "enabled" if self.tts_enabled else "disabled"
            self._log_progress(f"TTS toggled to {status}")
            return status
        except Exception as e:
            error_message = f"Error toggling TTS: {str(e)}"
            self._log_progress(error_message)
            raise

    def process_input(self, user_input: str, session_id: str) -> str:
        try:
            self._log_progress("Initiating cognitive processes...")
            initial_response, tool_calls = self._get_initial_response(user_input, session_id)
            lobe_responses = self._process_lobes(user_input, initial_response)
            memory_context = self._integrate_memory(user_input, initial_response, lobe_responses)
            sentiment = analyze_sentiment(user_input)
            
            tool_responses = self._process_tool_calls(tool_calls) if tool_calls else None
            final_response = self._generate_final_response(user_input, initial_response, lobe_responses, memory_context, sentiment, tool_responses, session_id)
            
            self._log_progress("Cognitive processing complete. Formulating response...")
            return final_response
        except Exception as e:
            error_message = f"Cognitive error encountered: {e} at {time.strftime('%Y-%m-%d %H:%M:%S')}"
            self._log_progress(error_message)
            return "An unexpected error occurred while processing your request. I'll do my best to address your needs based on the information available."
    
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, max=10), retry=retry_if_exception_type(Exception))
    def _get_initial_response(self, combined_input: str, session_id: str) -> Tuple[str, List[Any]]:
        self._log_progress("Initiating primary language model response...")
        initial_prompt = self._construct_initial_prompt(combined_input)
        system_message = self._construct_system_message()
        response, tool_calls = llm_api_calls.chat(initial_prompt, system_message, tools, progress_callback=self.progress_callback)
        if not response or len(response.strip()) == 0:
            raise ValueError("Empty response received from LLM")
        self._update_chat_history(session_id, {"role": "user", "content": combined_input})
        self._update_chat_history(session_id, {"role": "assistant", "content": response})
        return response, tool_calls

    def _process_tool_calls(self, tool_calls):
        tool_responses = {}
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = llm_api_calls.available_functions.get(function_name)
            if function_to_call:
                try:
                    function_args = json.loads(tool_call.function.arguments)
                    function_response = function_to_call(**function_args, progress_callback=self.progress_callback)
                    tool_responses[function_name] = function_response
                except Exception as e:
                    if self.progress_callback:
                        self.progress_callback(f"Error in {function_name}: {str(e)}")
                    tool_responses[function_name] = {"error": str(e)}
        return tool_responses

    def _process_lobes(self, user_input: str, initial_response: str) -> Dict[str, Any]:
        self._log_progress("Processing lobes...")
        combined_input = f"{user_input}\n{initial_response}\n"
        return self.lobes_processing.process_all_lobes(combined_input)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10), retry=retry_if_exception_type(Exception))
    def _integrate_memory(self, user_input: str, initial_response: str, lobe_responses: Dict[str, Any]) -> str:
        self._log_progress("Integrating memory and context...")
        combined_input = f"{user_input}\n{initial_response}\n{json.dumps(lobe_responses)}\n"
        embedding = generate_embedding(combined_input, self.embeddings_model, self.collection, self.collection_size)
        add_to_memory(combined_input, self.embeddings_model, self.collection, self.collection_size)
        self.collection_size += 1
        relevant_memory = retrieve_relevant_memory(embedding, self.collection)

        return " ".join(str(item) for item in relevant_memory if item is not None)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, max=10), retry=retry_if_exception_type(Exception))
    def _generate_final_response(self, user_input: str, initial_response: str, lobe_responses: Dict[str, Any], memory_context: str, sentiment: Dict[str, float], tool_responses: Dict[str, Any], session_id: str) -> str:
        self._log_progress("Generating final response...")
        context = self._construct_final_prompt(user_input, initial_response, lobe_responses, memory_context, sentiment, tool_responses)
        system_message = self._construct_system_message()
        final_response, _ = llm_api_calls.chat(context, system_message, tools, progress_callback=self.progress_callback)

        if not final_response or len(final_response.strip()) == 0:
            raise ValueError("Empty final response received from LLM")
        self.last_response = final_response
        self._update_chat_history(session_id, {"role": "user", "content": user_input})
        self._update_chat_history(session_id, {"role": "assistant", "content": final_response})
        return final_response

    def _construct_system_message(self) -> str:
        return f"""You are {FinalAgentPersona.name}, an all-knowing AI assistant with tools to use to its advantage. {FinalAgentPersona.description} 
        You have access to a vast knowledge base and various tools to assist you. Always be proactive, 
        inferring the user's needs and taking action without asking clarifying questions. Be confident, 
        direct, and provide comprehensive responses. If you're unsure about something, make a reasonable 
        assumption and proceed with the most likely course of action."""

    def _construct_initial_prompt(self, combined_input: str) -> str:
        return f"""
        Analyze the following user input and context to generate a proactive and comprehensive response. 
        Your goal is to infer the user's needs and provide a helpful, specific, and engaging response 
        that addresses those needs without asking clarifying questions.

        Input: "{combined_input}"

        Your task is to:
        1. Understand the request and its context thoroughly.
        2. Infer any unstated needs or implications from the user's input.
        3. Provide a direct, helpful, and comprehensive response that addresses both stated and inferred needs.
        4. Use any necessary tools proactively to gather information or perform actions that will benefit the user.
        5. If multiple interpretations are possible, choose the most likely one and proceed confidently.

        Remember, you have access to a vast knowledge base and various tools. Use them proactively to 
        provide the best possible response. Don't ask the user for clarification; instead, make informed 
        assumptions and act on them.

        Respond with your analysis, any tool calls you deem necessary, or your initial response to yourself as your thoughts as this stage doesnt have you talk to user yet..
        tool list: {json.dumps(tools, indent=2)}

        Your response:
        """

    def _construct_final_prompt(self, user_input: str, initial_response: str, lobe_responses: Dict[str, Any], memory_context: str, sentiment: Dict[str, float], tool_responses: Dict[str, Any]) -> str:
        tool_results = json.dumps(tool_responses, indent=2) if tool_responses else "No tools were used."
        return f"""
        Synthesize the following information to formulate a comprehensive and proactive response:

        User Input: "{user_input}"

        Initial Response: {initial_response}

        Tool Use Results: {tool_results}

        Lobe Processing Results:
        {json.dumps(lobe_responses, indent=2)}

        Relevant Memory Context: {memory_context}

        Detected Sentiment: Polarity: {sentiment['polarity']}, Subjectivity: {sentiment['subjectivity']}

        Based on this information, generate a response that addresses the user's input comprehensively and proactively. Your response should:

        1. Directly address the user's main point and any inferred needs
        2. Incorporate relevant insights from the lobe processing results and memory context
        3. Integrate the results from any tools that were used
        4. Propose next steps or additional actions that would benefit the user
        5. Be confident, direct, and avoid asking clarifying questions

        Ensure that all parts contribute to a unified, helpful, and proactive answer. If you're unsure about 
        any aspect, make a reasonable assumption and proceed with the most likely course of action.

        Your response must be natural and not include any other wrappers or instructions, only the response itself.
        """

    def _log_progress(self, message: str):
        self.progress_callback(f"{message} at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    def _update_chat_history(self, session_id: str, message: Dict[str, str]):
        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = []
        self.chat_histories[session_id].append(message)
        # Keep only the last 10 messages
        self.chat_histories[session_id] = self.chat_histories[session_id][-10:]

    def get_chat_history(self, session_id: str):
        return self.chat_histories.get(session_id, [])

    def get_detailed_info(self):
        try:
            detailed_info = {
                "tts_enabled": self.tts_enabled,
                "embeddings_model": self.embeddings_model,
                "collection_size": self.collection_size,
                "last_response": self.last_response
            }
            return json.dumps(detailed_info, indent=2)
        except Exception as e:
            return f"Error retrieving detailed info: {str(e)} at {time.strftime('%Y-%m-%d %H:%M:%S')}"
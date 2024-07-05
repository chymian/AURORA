import json
import time
import re
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
        self.web_research_cache = {}
        self._initialize()

    def _initialize(self):
        self._log_progress("Initializing Brain")
        self.tts_enabled = True
        self.lobes_processing = LobesProcessing()
        self.embeddings_model = "mxbai-embed-large"
        self.chat_histories = {}
        self.last_response = ""
        self.context = {"Conversation History": ""}
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
            tasks = self._break_down_tasks(user_input)
            final_response = ""

            for task in tasks:
                task_response = self._process_single_task(task, session_id)
                self._update_context(task, task_response)
                final_response += task_response + "\n\n"
                
                self._add_to_memory(f"Task: {task}\nResponse: {task_response}")

            self._log_progress("All tasks completed. Generating final response...")
            final_response = self._generate_final_response(final_response, user_input)
            self._update_context("Final Response", final_response)
            self._add_to_memory(f"User Input: {user_input}\nFinal Response: {final_response}")

            return final_response
        except Exception as e:
            error_message = f"Error encountered: {e} at {time.strftime('%Y-%m-%d %H:%M:%S')}"
            self._log_progress(error_message)
            return f"I apologize, but an error occurred while processing your request. I'll do my best to assist you based on the information available. Error details: {str(e)}"

    def _break_down_tasks(self, user_input: str) -> List[str]:
        self._log_progress("Breaking down tasks...")
        relevant_memory = self._retrieve_relevant_memory(user_input)
        
        prompt = f"""
        Analyze the following user input and break it down into distinct tasks:

        User Input: {user_input}

        Relevant Context:
        {relevant_memory}

        Provide the tasks as a Python list of strings. If there's only one task, still use the list format.
        Example: ["Task 1", "Task 2"] or ["Single Task"]
        """
        response, _ = llm_api_calls.chat(prompt, self._construct_system_message(), tools, progress_callback=self.progress_callback)
        
        tasks = self._parse_tasks(response)
        self._log_progress(f"Identified {len(tasks)} tasks")
        return tasks

    def _parse_tasks(self, response: str) -> List[str]:
        try:
            tasks = json.loads(response)
            if isinstance(tasks, list) and all(isinstance(item, str) for item in tasks):
                return tasks
        except json.JSONDecodeError:
            pass

        try:
            tasks = eval(response)
            if isinstance(tasks, list) and all(isinstance(item, str) for item in tasks):
                return tasks
        except:
            pass

        tasks = [task.strip() for task in response.split('\n') if task.strip()]
        if tasks:
            return tasks

        return [response.strip()]

    def _process_single_task(self, task: str, session_id: str) -> str:
        self._log_progress(f"Processing task: {task}")
        relevant_memory = self._retrieve_relevant_memory(task)
        
        if "search" in task.lower():
            if task in self.web_research_cache:
                web_results = self.web_research_cache[task]
                self._log_progress(f"Using cached results for task: {task}")
            else:
                web_results = self._perform_web_research(task)
                self.web_research_cache[task] = web_results
                self._log_progress(f"Cached results for task: {task}")

            task_response = self._generate_task_response(web_results, session_id)
        else:
            initial_response, tool_calls = self._get_initial_response(task, relevant_memory, session_id)
            lobe_responses = self._process_lobes(task, initial_response)
            sentiment = analyze_sentiment(task)
            consolidated_info = self._consolidate_information(task, initial_response, lobe_responses, sentiment, None, relevant_memory)
            task_response = self._generate_task_response(consolidated_info, session_id)
        
        return task_response

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10), retry=retry_if_exception_type(Exception))
    def _get_initial_response(self, task: str, relevant_memory: str, session_id: str) -> Tuple[str, List[Any]]:
        self._log_progress("Generating initial response...")
        initial_prompt = self._construct_initial_prompt(task, relevant_memory)
        system_message = self._construct_system_message()
        response, tool_calls = llm_api_calls.chat(initial_prompt, system_message, tools, progress_callback=self.progress_callback)
        if not response or len(response.strip()) == 0:
            raise ValueError("Empty response received from LLM")
        self._update_chat_history(session_id, {"role": "user", "content": task})
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
                    self.progress_callback(f"Error in {function_name}: {str(e)}")
                    tool_responses[function_name] = {"error": str(e)}
        return tool_responses

    def _process_lobes(self, task: str, initial_response: str) -> Dict[str, Any]:
        self._log_progress("Processing lobes...")
        combined_input = f"{task}\n{initial_response}\n"
        return self.lobes_processing.process_all_lobes(combined_input)

    def _retrieve_relevant_memory(self, query: str) -> str:
        embedding = generate_embedding(query, self.embeddings_model, self.collection, self.collection_size)
        relevant_memory = retrieve_relevant_memory(embedding, self.collection)
        return " ".join(str(item) for item in relevant_memory if item is not None)

    def _consolidate_information(self, task: str, initial_response: str, lobe_responses: Dict[str, Any], 
                                 sentiment: Dict[str, float], tool_responses: Dict[str, Any],
                                 relevant_memory: str) -> Dict[str, Any]:
        return {
            "task": task,
            "initial_response": initial_response,
            "lobe_responses": lobe_responses,
            "sentiment": sentiment,
            "tool_responses": tool_responses,
            "current_context": self.context,
            "relevant_memory": relevant_memory
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10), retry=retry_if_exception_type(Exception))
    def _generate_task_response(self, consolidated_info: Dict[str, Any], session_id: str) -> str:
        self._log_progress("Generating task response...")
        context = self._construct_task_prompt(consolidated_info)
        system_message = self._construct_system_message()
        task_response, _ = llm_api_calls.chat(context, system_message, tools, progress_callback=self.progress_callback)

        if not task_response or len(task_response.strip()) == 0:
            raise ValueError("Empty task response received from LLM")
        
        self.last_response = task_response
        self._update_chat_history(session_id, {"role": "user", "content": consolidated_info["task"]})
        self._update_chat_history(session_id, {"role": "assistant", "content": task_response})
        return task_response

    def _update_context(self, input_text: str, response: str):
        summary = self._generate_context_summary(input_text, response)
        self.context["Conversation History"] += f"\n{summary}"

    def _generate_context_summary(self, input_text: str, response: str) -> str:
        prompt = f"""
        Summarize the key points from the following interaction in 1-2 sentences:

        Input: {input_text}
        Response: {response}

        Focus on the most important information and any decisions or actions taken.
        """
        summary, _ = llm_api_calls.chat(prompt, self._construct_system_message(), tools, progress_callback=self.progress_callback)
        return summary

    def _generate_final_response(self, task_responses: str, original_input: str) -> str:
        relevant_memory = self._retrieve_relevant_memory(original_input)
        
        prompt = f"""
        Create a comprehensive and coherent response based on the following information:

        Original User Input: {original_input}

        Task Responses:
        {task_responses}

        Conversation History:
        {self.context["Conversation History"]}

        Relevant Memory:
        {relevant_memory}

        Your response should:
        1. Directly address the user's original input
        2. Incorporate insights from all completed tasks
        3. Maintain a consistent tone and style
        4. Be clear, concise, and engaging
        5. Propose any necessary follow-up actions or questions

        Provide this response in a natural, conversational manner.
        """
        final_response, _ = llm_api_calls.chat(prompt, self._construct_system_message(), tools, progress_callback=self.progress_callback)
        return final_response

    def _construct_system_message(self) -> str:
        return f"""You are {FinalAgentPersona.name}, an advanced AI assistant capable of handling a wide variety of tasks. {FinalAgentPersona.description} 
        You have access to a vast knowledge base and various tools to assist you. Always be proactive, 
        inferring the user's needs and taking action without asking unnecessary clarifying questions. Be confident, 
        direct, and provide comprehensive responses. If you're unsure about something, make a reasonable 
        assumption and proceed with the most likely course of action. Maintain context throughout the conversation
        and use it to inform your responses and decisions."""

    def _construct_initial_prompt(self, task: str, relevant_memory: str) -> str:
        return f"""
        Analyze the following task and provide an initial response:

        Task: "{task}"

        Relevant Context:
        {relevant_memory}

        Current Conversation History:
        {self.context["Conversation History"]}

        Your response should:
        1. Demonstrate understanding of the task
        2. Incorporate relevant context and conversation history
        3. Identify any necessary tools or actions needed to complete the task
        4. Provide an initial approach or response to the task

        If you need to use any tools, call them directly. Otherwise, provide your initial thoughts and approach.
        """

    def _construct_task_prompt(self, consolidated_info: Dict[str, Any]) -> str:
        tool_results = json.dumps(consolidated_info.get("tool_responses", {}), indent=2) if consolidated_info.get("tool_responses") else "No tools were used."
        return f"""
        Provide a comprehensive response to the following task, considering all available information:

        Task: "{consolidated_info['task']}"

        Initial Response: {consolidated_info['initial_response']}

        Tool Use Results: {tool_results}

        Lobe Processing Results:
        {json.dumps(consolidated_info['lobe_responses'], indent=2)}

        Detected Sentiment: Polarity: {consolidated_info['sentiment']['polarity']}, Subjectivity: {consolidated_info['sentiment']['subjectivity']}

        Current Conversation History:
        {consolidated_info['current_context']['Conversation History']}

        Relevant Memory:
        {consolidated_info['relevant_memory']}

        Your response should:
        1. Directly address the task at hand
        2. Incorporate insights from the lobe processing, memory context, and conversation history
        3. Integrate the results from any tools that were used
        4. Consider the detected sentiment in your response tone
        5. Propose any necessary follow-up actions or questions

        Provide a clear, detailed, and actionable response that best assists the user.
        """

    def _log_progress(self, message: str):
        self.progress_callback(f"{message} at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    def _update_chat_history(self, session_id: str, message: Dict[str, str]):
        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = []
        self.chat_histories[session_id].append(message)
        self.chat_histories[session_id] = self.chat_histories[session_id][-10:]

    def _add_to_memory(self, text: str):
        try:
            add_to_memory(text, self.embeddings_model, self.collection, self.collection_size)
            self.collection_size += 1
        except Exception as e:
            self._log_progress(f"Error adding to memory: {str(e)}")

    def get_chat_history(self, session_id: str):
        return self.chat_histories.get(session_id, [])

    def get_detailed_info(self):
        try:
            detailed_info = {
                "tts_enabled": self.tts_enabled,
                "embeddings_model": self.embeddings_model,
                "collection_size": self.collection_size,
                "last_response": self.last_response,
                "current_context": self.context
            }
            return json.dumps(detailed_info, indent=2)
        except Exception as e:
            return f"Error retrieving detailed info: {str(e)} at {time.strftime('%Y-%m-%d %H:%M:%S')}"

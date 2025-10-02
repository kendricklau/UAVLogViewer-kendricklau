from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_MODEL = "gpt-4o-mini"
GENERAL_SYSTEM_PROMPT = """
    You are an expert in ArduPilot operations and vehicle diagnosis.
    You are the general expert. Focus on general flight data and related flight indicators.
    Input has immutable flight data you must base and parse your response on.
    Ensure you address the user question first, and then elaborate. Don't be too verbose.
"""

SUMMARIZE_FOR_USER_SYSTEM_PROMPT = """
    You are the summarize for user expert. Focus on summarizing the context and answering the question for the user.
    Respond in a concise human readable format and feel free to be story telling. Use SI units and abbreviations where appropriate.
    summary of issues or findings with timestamps associated with each of them. Each should be its own paragraph and comes with a diagnostic and suggested cause.
    Ensure you address the user question first, and then elaborate. No need to mention every expert explicitly, just a quick summary of what each expert found.
    This is an executive summary, so don't be too verbose. Do not wrap in markdown code blocks or add any other text.
    At the end, add a list of relevant timestamps and one sentence of each event. This can be used for the user to find easily in the flight plot. Keep it concise.
"""
PLANNER_SYSTEM_PROMPT = f"""
    You are an expert in ArduPilot operations and vehicle diagnosis.
    Given a question, decide the following and respond as a compact JSON object with the keys:
    "requested_time_windows": [(timestamp_ms, window_ms), ...], array of a tuple pair of single point timestamps and a before and after window size max of 100ms to request from those timestamps. You can always default to tuple (0, 10000000) if you need the full log or no timestamps identified. Always in milliseconds and always include at least one tuple.
    "requested_experts": array of experts to call drawn from ["attitude","gps","ekf","parameters"], ok to not call any experts if the question is not related to any of the experts.
    Return ONLY valid JSON. Do not wrap in markdown code blocks or add any other text.
"""
EXPERT_SYSTEM_PROMPT = {
    "attitude": """
        You are an expert in ArduPilot operations and vehicle diagnosis. Take into account the provided evidence and diagnostics to answer the question.
        You are Attitude Expert. Focus on ATT messages and related attitude indicators.
        Input has immutable flight data you must base and parse your response on. Ensure you're converging on available attitude data from available ardupilot rag docs
        "evidence": array of evidence citing the source from rag docs, flight data, and your own analysis.
        "diagnostics": dictionary of diagnostics from the expert.
        "suggested_cause": suggested cause of the issue.
        Return ONLY valid JSON. Do not wrap in markdown code blocks or add any other text.
    """,
    "gps": """
        You are an expert in ArduPilot operations and vehicle diagnosis. Take into account the provided evidence and diagnostics to answer the question.
        You are GPS Expert. Focus on GPS messages and related GPS indicators.
        Input has immutable flight data you must base and parse your response on. Ensure you're converging on available gps data from available ardupilot rag docs
        "evidence": array of evidence citing the source from rag docs, flight data, and your own analysis.
        "diagnostics": dictionary of diagnostics from the expert.
        "suggested_cause": suggested cause of the issue.
        Return ONLY valid JSON. Do not wrap in markdown code blocks or add any other text.
    """,
    "ekf": """
        You are an expert in ArduPilot operations and vehicle diagnosis. Take into account the provided evidence and diagnostics to answer the question.
        You are the EKF expert. Focus on XKF and EKF health indicators.
        Input has immutable flight data you must base and parse your response on. Ensure you're converging on available ekf data from available ardupilot rag docs
        "evidence": array of evidence citing the source from rag docs, flight data, and your own analysis.
        "diagnostics": dictionary of diagnostics from the expert.
        "suggested_cause": suggested cause of the issue.
        Return ONLY valid JSON. Do not wrap in markdown code blocks or add any other text.
    """,
    "parameters": """
        You are an expert in ArduPilot operations and vehicle diagnosis. Take into account the provided evidence and diagnostics to answer the question.
        You are the parameters expert. Concentrate on PARM messages and configuration anomalies.
        Input has immutable flight data you must base and parse your response on. Ensure you're converging on available parameters data from available ardupilot rag docs
        "evidence": array of evidence citing the source from rag docs, flight data, and your own analysis.
        "diagnostics": dictionary of diagnostics from the expert.
        "suggested_cause": suggested cause of the issue.
        Return ONLY valid JSON. Do not wrap in markdown code blocks or add any other text.
    """,
}

INTEGRATION_SYSTEM_PROMPT = """
    You are an expert in ArduPilot operations and vehicle diagnosis.
    You are the integration expert. 
    Ensure you're reasoning and cross-analyzing between experts analysis and your own analysis. Do not modify the expert data.
    Hypothesize about any patterns or correlations between the experts analysis and your own analysis. Do not modify the expert data.
    Help the user solve the problem or provide additional context that might not be obvious. Do not modify the expert data.
    "evidence": array of evidence citing the source from rag docs, flight data, and your own analysis.
    "diagnostics": dictionary of diagnostics from the expert.
    "suggested_cause": suggested cause of the issue.
    Ensure we're converging on actually answering the original question.
    Return ONLY valid JSON. Do not wrap in markdown code blocks or add any other text.
"""
def _append_agent_chat_to_history(log_id, agent_name, question, answer):
    if not log_id:
        return

    rag_file = f"data/{log_id}_rag.json"
    if not os.path.exists(rag_file):
        return

    try:
        with open(rag_file, "r") as f:
            rag_data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return

    documents = rag_data.get("documents")
    if not isinstance(documents, list):
        return

    timestamp = datetime.now().isoformat()
    question_text = str(question)
    answer_text = str(answer)
    entry = (
        f"\n\n--- Agent Chat Entry {timestamp} [{agent_name}] ---\n"
        f"Question: {question_text}\n"
        f"Response: {answer_text}"
    )

    chat_doc = None
    for doc in documents:
        if doc.get("document_type") == "chat_history":
            chat_doc = doc
            break

    if chat_doc is None:
        chat_doc = {
            "document_id": f"{log_id}_chat_history",
            "document_type": "chat_history",
            "title": f"Chat History - {log_id}",
            "content": f"Chat History for Flight Log {log_id}{entry}",
            "metadata": {
                "created_at": timestamp,
                "last_updated": timestamp,
                "message_count": 1,
                "log_id": log_id
            }
        }
        documents.append(chat_doc)
    else:
        existing_content = chat_doc.get("content")
        if not isinstance(existing_content, str):
            existing_content = f"Chat History for Flight Log {log_id}"
        chat_doc["content"] = existing_content + entry
        metadata = chat_doc.setdefault("metadata", {})
        metadata.setdefault("created_at", timestamp)
        metadata["last_updated"] = timestamp
        metadata["message_count"] = metadata.get("message_count", 0) + 1

    try:
        with open(rag_file, "w") as f:
            json.dump(rag_data, f, indent=2)
    except OSError:
        return
# Token budgeting to keep prompts and responses within model limits.
MAX_INPUT_TOKENS = 20000
MAX_OUTPUT_TOKENS = 1024
AVG_CHARS_PER_TOKEN = 4
class ChatAgent:
    def __init__(self, log_id=None):
        self.client = OpenAI()
        self.log_id = log_id

    def _safe_parse_json(self, raw_content):
        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError:
            return {"raw": raw_content}
        return parsed if isinstance(parsed, (dict, list)) else {"raw": parsed}

    def _json_dump(self, data):
        if isinstance(data, (dict, list)):
            return json.dumps(data, indent=2)
        return str(data)

    def _truncate_text(self, text: str, max_tokens: int):
        if not text or max_tokens <= 0:
            return text
        max_chars = max_tokens * AVG_CHARS_PER_TOKEN
        if len(text) <= max_chars:
            return text
        if max_chars <= 3:
            return text[:max_chars]
        return text[:max_chars - 3] + "..."

    def ask(self, question: str):
        prompt = self._truncate_text(question, MAX_INPUT_TOKENS)
        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        content = response.choices[0].message.content
        _append_agent_chat_to_history(self.log_id, "ask", question, content)
        return content
    
    def planner(self, question: str):
        context, _ = self.get_context_with_chat_history(self.log_id)
        context = self._truncate_text(context, MAX_INPUT_TOKENS)
        prompt_question = self._truncate_text(question, MAX_INPUT_TOKENS)
        
        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {prompt_question}\n\nContext:\n{context}"}
            ],
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        planner_response_content = response.choices[0].message.content
        _append_agent_chat_to_history(self.log_id, "planner", question, planner_response_content)
        print("planner_response_content", planner_response_content)
        return planner_response_content
    
    def call_general(self, question: str):
        rag_docs = self.get_rag_docs(self.log_id)
        context = "\n\n".join([f"Title: {doc.get('title', '')}\nContent: {doc.get('content', '')}" for doc in rag_docs])
        context = self._truncate_text(context, MAX_INPUT_TOKENS)
        prompt_question = self._truncate_text(question, MAX_INPUT_TOKENS)
        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": GENERAL_SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {prompt_question}\n\nContext:\n{context}"}
            ],
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        content = response.choices[0].message.content
        _append_agent_chat_to_history(self.log_id, "general", question, content)
        return content
    
    def call_summarizeForUser(self, question, integration_result):
        """Summarize the integration result for the user"""
        rag_docs = self.get_rag_docs(self.log_id)
        context = "\n\n".join([f"Title: {doc.get('title', '')}\nContent: {doc.get('content', '')}" for doc in rag_docs])
        context = self._truncate_text(context, MAX_INPUT_TOKENS)
        prompt_question = self._truncate_text(question, MAX_INPUT_TOKENS)
        
        messages = [
            {"role": "system", "content": f"{SUMMARIZE_FOR_USER_SYSTEM_PROMPT}\n\nContext:\n{context}"},
            {"role": "user", "content": f"Question: {prompt_question}\n\nIntegration Result: {integration_result}"}
        ]
        
        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        
        content = response.choices[0].message.content
        _append_agent_chat_to_history(self.log_id, "summarizeForUser", question, content)
        return content

    

    def call_expert(self, question: str, expert: str, timestamp_ms: list):
        rag_docs = self.get_rag_docs(self.log_id, expert)
        
        # Define mapping for expert-specific signals to filter flight_data
        EXPERT_MESSAGE_MAPPING = {
            "attitude": "ATT",
            "gps": "GPS[0]", # Include common GPS message types
            "ekf": "XKQ[0]", # Include EKF and its common variants
            "parameters": "PARM"
        }
        MESSAGE_SIGNAL_MAPPING = {
            "ATT": ["DesRoll",
                    "Roll",
                    "DesPitch",
                    "Pitch",
                    "DesYaw",
                    "Yaw",
                    "ErrRP",
                    "ErrYaw",
                    "AEKF"],
            "GPS[0]": ["I", 
                    "Status",
                    "GMS",
                    "GWk",
                    "NSats",
                    "HDop",
                    "Lat",
                    "Lng",
                    "Alt",
                    "Spd",
                    "GCrs",
                    "VZ",
                    "Yaw",
                    "U"],
            "XKQ[0]": ["C",
                    "Q1",
                    "Q2",
                    "Q3",
                    "Q4"],
            "XKQ[1]": ["C",
                    "Q1",
                    "Q2",
                    "Q3",
                    "Q4"],
            "XKQ[2]": ["C",
                    "Q1",
                    "Q2",
                    "Q3",
                    "Q4"],
            "PARM": ["Name",
                    "Value",
                    "Default"]
        }

        # Generate list of signals to query data for
        relevant_signals = []
        message_signals = [EXPERT_MESSAGE_MAPPING[expert]]
        for message in message_signals:
            relevant_signals.extend(MESSAGE_SIGNAL_MAPPING[message])
        relevant_signals = list(set(relevant_signals))
        return_data = []

        for timestamp, window_ms in timestamp_ms:
            if timestamp == 0 and window_ms == 10000000 or timestamp_ms == []:
                flight_data = self.get_all_flight_data()
            else:
                flight_data = self.get_flight_data_v2(timestamp, relevant_signals, window_ms)
            flight_data_str = self._json_dump(flight_data)
            flight_data_str = self._truncate_text(flight_data_str, MAX_INPUT_TOKENS)
            context = "\n\n".join([f"Title: {doc.get('title', '')}\nContent: {doc.get('content', '')}" for doc in rag_docs])
            context = self._truncate_text(context, MAX_INPUT_TOKENS)
            prompt_question = self._truncate_text(question, MAX_INPUT_TOKENS)
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": EXPERT_SYSTEM_PROMPT[expert]},
                    {"role": "user", "content": f"Question: {prompt_question}\n\nContext:\n{context}\n\nFlight Data:\n{flight_data_str}"}
                ],
                max_tokens=MAX_OUTPUT_TOKENS,
            )
            response_text = response.choices[0].message.content
            return_data.append(self._safe_parse_json(response_text))
            _append_agent_chat_to_history(self.log_id, f"expert:{expert}", question, response_text)
        
        return return_data
        

    def call_integration(self, question: str, expert_responses: dict):
        rag_docs = self.get_rag_docs(self.log_id)
        context = "\n\n".join([f"Title: {doc.get('title', '')}\nContent: {doc.get('content', '')}" for doc in rag_docs])
        context = self._truncate_text(context, MAX_INPUT_TOKENS)
        prompt_question = self._truncate_text(question, MAX_INPUT_TOKENS)
        
        # Prepare expert payload from the initial expert_responses
        expert_payload_sections = []
        expert_count = max(1, len(expert_responses))
        per_expert_budget = max(1, MAX_INPUT_TOKENS // expert_count)
        for expert_name, payload in expert_responses.items():
            payload_text = self._truncate_text(self._json_dump(payload), per_expert_budget)
            expert_payload_sections.append(f"{expert_name}: {payload_text}")
        expert_payload = self._truncate_text("\n".join(expert_payload_sections), MAX_INPUT_TOKENS)
        
        # Call the integration agent once
        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": INTEGRATION_SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {prompt_question}\n\nExpert Responses:\n{expert_payload}"}
            ],
            max_tokens=MAX_OUTPUT_TOKENS,
        )

        content = response.choices[0].message.content
        parsed_response = self._safe_parse_json(content)
        
        # Append integration agent's chat to history
        _append_agent_chat_to_history(self.log_id, "integration", question, content)

        return parsed_response

    def debug_chatbot(self, question, log_window=None):
        expert_resp = {}
        
        planner_resp = self.planner(question)
        planner_resp = self._safe_parse_json(planner_resp)
        if "requested_time_windows" not in planner_resp:
            planner_resp["requested_time_windows"] = [(0, 10000000)]
        if "requested_experts" not in planner_resp:
            expert_resp["general"] = self.call_general(question)
        else:
            for expert in planner_resp["requested_experts"]:
                expert_resp[expert] = self.call_expert(question, expert, planner_resp["requested_time_windows"])
        integration_resp = self.call_integration(question, expert_resp)
        final_resp = self.call_summarizeForUser(question, integration_resp)
        return final_resp

    def get_rag_docs(self, log_id, document_type=None):
        with open(f'data/{log_id}_rag.json', 'r') as f:
            docs = json.load(f)['documents']
        
        if document_type:
            import re
            # Filter documents that match the document_type pattern
            filtered_docs = []
            for doc in docs:
                if 'document_type' in doc and re.search(document_type, doc['document_type'], re.IGNORECASE) or doc['document_type'] == 'reference':
                    filtered_docs.append(doc)
            
            return filtered_docs
        
        return docs

    def get_chat_history(self, log_id):
        """Get chat history from single chat_history document"""
        try:
            with open(f'data/{log_id}_rag.json', 'r') as f:
                rag_data = json.load(f)
            
            # Find chat_history document
            chat_doc = None
            for doc in rag_data["documents"]:
                if doc.get("document_type") == "chat_history":
                    chat_doc = doc
                    break
            
            return chat_doc if chat_doc else None
            
        except Exception as e:
            print(f"Error getting chat history: {e}")
            return None

    def get_context_with_chat_history(self, log_id, document_type=None):
        """Get RAG docs with chat history included"""
        # Get regular RAG docs
        rag_docs = self.get_rag_docs(log_id, document_type)
        
        # Get chat history document
        chat_doc = self.get_chat_history(log_id)
        
        # Combine and format
        all_docs = rag_docs
        if chat_doc:
            all_docs.append(chat_doc)
        
        context = "\n\n".join([
            f"Title: {doc.get('title', '')}\nContent: {doc.get('content', '')}" 
            for doc in all_docs
        ])
        
        return context, all_docs

    def get_flight_data_v2(self, timestamp_ms: float, signals: list = None, window_ms: int = 10):
        max_data = 400
        try:
            with open(f'data/{self.log_id}.json', 'r') as f:
                log_data = json.load(f)
        except FileNotFoundError:
            return {"error": f"Log file {self.log_id}.json not found"}
        
        # grab the relevant time window
        return_data = []
        messages = log_data["time_series_data"].keys()
        for message in messages:
            for signal in signals:
                if signal in log_data["time_series_data"][message]["data"].keys():
                    start_time = float(log_data["time_series_data"][message]["time_range"]["start"])
                    end_time = float(log_data["time_series_data"][message]["time_range"]["end"])
                    
                    if start_time <= float(timestamp_ms) < end_time:
                        log_index = [
                            i for i, t in enumerate(log_data["time_series_data"][message]["data"]["time_boot_ms"])
                            if abs(float(t) - float(timestamp_ms)) <= window_ms / 2
                        ]
                        for i in log_index:
                            return_data.append({"tsd": log_data["time_series_data"][message]["data"]["time_boot_ms"][i], signal: log_data["time_series_data"][message]["data"][signal][i]})
                    else:
                        return {"error": f"Signal {signal} not found in message {message} during {timestamp_ms}"}
                
        if len(return_data) > max_data:
            step = len(return_data) // max_data
            return_data = return_data[::step]
        else:   
            return_data = [return_data]
        metadata = []
        metadata.append({"flight_summary": log_data["flight_summary"]})
        metadata.append({"parameters": log_data["parameters"]["changeArray"]})
        metadata.append({"default_parameters": log_data["default_parameters"]})
        return_data.append({"metadata": metadata})
        return return_data

    def get_all_flight_data(self):
        max_data = 400
        try:
            with open(f'data/{self.log_id}.json', 'r') as f:
                log_data = json.load(f)
        except FileNotFoundError:
            return {"error": f"Log file {self.log_id}.json not found"}
        
        # grab the relevant time window
        return_data = []
        messages = log_data["time_series_data"].keys()
        for message in messages:
            for signal in log_data["time_series_data"][message]["data"].keys():
                return_data.append({"tsd": log_data["time_series_data"][message]["data"]["time_boot_ms"], signal: log_data["time_series_data"][message]["data"][signal]})
                
        if len(return_data) > max_data:
            step = len(return_data) // max_data
            return_data = return_data[::step]
        else:   
            return_data = [return_data]
        metadata = []
        metadata.append({"flight_summary": log_data["flight_summary"]})
        metadata.append({"parameters": log_data["parameters"]["changeArray"]})
        metadata.append({"default_parameters": log_data["default_parameters"]})
        return_data.append({"metadata": metadata})
        return return_data
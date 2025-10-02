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
"""
PLANNER_SYSTEM_PROMPT = f"""
    You are an expert in ArduPilot operations and vehicle diagnosis.
    Given a question, decide the following and respond as a compact JSON object with the keys:
    "requested_time_windows": array of single point timestamps to request from those timestamps. Always in milliseconds.
    "requested_experts": array of experts to call drawn from ["attitude","gps","ekf","parameters"], ok to not call any experts if the question is not related to any of the experts.
    Return ONLY valid JSON. Do not wrap in markdown code blocks or add any other text.
"""
        
EXPERT_SYSTEM_PROMPT = {
    "attitude": """
        You are an expert in ArduPilot operations and vehicle diagnosis.
        You are Attitude Expert. Focus on ATT messages and related attitude indicators.
        Input has immutable flight data you must base and parse your response on. Ensure you're converging on available attitude data from available ardupilot rag docs
        Return Only valid JSON: [{timestamp, signal name, signal value, start_ts, end_ts}], diagnostics: {...}, suggested_cause: ""}
        Return ONLY valid JSON. Do not wrap in markdown code blocks or add any other text.
    """,
    "gps": """
        You are an expert in ArduPilot operations and vehicle diagnosis.
        You are GPS Expert. Focus on GPS messages and related GPS indicators.
        Input has immutable flight data you must base and parse your response on. Ensure you're converging on available gps data from available ardupilot rag docs
        Return Only valid JSON: [{timestamp, signal name, signal value, start_ts, end_ts}], suggested_cause: ""}
        Return ONLY valid JSON. Do not wrap in markdown code blocks or add any other text.
    """,
    "ekf": """
        You are an expert in ArduPilot operations and vehicle diagnosis.
        You are the EKF expert. Focus on XKF and EKF health indicators.
        Input has immutable flight data you must base and parse your response on. Ensure you're converging on available ekf data from available ardupilot rag docs
        Return Only valid JSON: [{timestamp, signal name, signal value, start_ts, end_ts}], diagnostics: {...}, suggested_cause: ""}
        Return ONLY valid JSON. Do not wrap in markdown code blocks or add any other text.
    """,
    "parameters": """
        You are an expert in ArduPilot operations and vehicle diagnosis.
        You are the parameters expert. Concentrate on PARM messages and configuration anomalies.
        Input has immutable flight data you must base and parse your response on. Ensure you're converging on available parameters data from available ardupilot rag docs
        Return Only valid JSON: [{timestamp, signal name, signal value, start_ts, end_ts}], diagnostics: {...}, suggested_cause: ""}
        Return ONLY valid JSON. Do not wrap in markdown code blocks or add any other text.
    """,
}

INTEGRATION_SYSTEM_PROMPT = """
    You are an expert in ArduPilot operations and vehicle diagnosis.
    You are the integration expert. 
    Ensure you're reasoning and cross-analyzing between experts analysis and your own analysis. Do not modify the expert data.
    Hypothesize about any patterns or correlations between the experts analysis and your own analysis. Do not modify the expert data.
    Help the user solve the problem or provide additional context that might not be obvious. Do not modify the expert data.
    timestamps: array of timestamps from the experts,
    findings: array of findings from the experts,
    diagnostics: dictionary of diagnostics from the experts
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
        
        return planner_response_content

    def call_expert(self, question: str, expert: str, flight_data: dict):
        rag_docs = self.get_rag_docs(self.log_id, expert)
        
        # Define mapping for expert-specific signals to filter flight_data
        EXPERT_SIGNAL_MAPPING = {
            "attitude": ["ATT"],
            "gps": ["GPS", "GPS[0]", "GPS[1]"], # Include common GPS message types
            "ekf": ["XKF", "XKF0", "XKF1", "XKF2", "EKF"], # Include EKF and its common variants
            "parameters": ["PARM"]
        }

        # Filter flight_data to only include signals relevant to the current expert
        filtered_flight_data = {}
        if "signals" in flight_data and expert in EXPERT_SIGNAL_MAPPING:
            relevant_signals = EXPERT_SIGNAL_MAPPING[expert]
            filtered_signals = {
                sig_type: sig_data
                for sig_type, sig_data in flight_data["signals"].items()
                if sig_type in relevant_signals
            }
            if filtered_signals:
                filtered_flight_data["signals"] = filtered_signals
        
        # Include other top-level keys that might be generally useful, like anomalies or context
        for key in ["anomalies", "context"]:
            if key in flight_data:
                filtered_flight_data[key] = flight_data[key]
        
        # Use filtered data if it's not empty, otherwise fall back to original flight_data
        # This ensures that if an expert has no specific signal mapping or if flight_data
        # doesn't contain a 'signals' key, the original data is still passed.
        flight_data_to_dump = filtered_flight_data if filtered_flight_data else flight_data
        flight_data_json_str = self._json_dump(flight_data_to_dump)
        # Re-generate flight_data_json_str with the filtered (or original) data
        flight_data_json_str = self._json_dump(flight_data_to_dump)
        flight_data_str = f"\n\nFlight Data:\n{flight_data_json_str}"
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
        _append_agent_chat_to_history(self.log_id, f"expert:{expert}", question, response_text)
        return self._safe_parse_json(response_text)

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
        flight_data = {}
        outputs = {}
        planner_response = self.planner(question)
        planner_response = self._safe_parse_json(planner_response)
        experts = planner_response.get("requested_experts", [])
        time_windows = planner_response.get("requested_time_windows", [])
        
        # Convert time windows to numbers
        for time_window in time_windows:
            try:
                # Handle both string timestamps and objects with timestamp/duration
                if isinstance(time_window, str):
                    timestamp_ms = float(time_window)
                    flight_data[time_window] = self.get_flight_data(timestamp_ms, None, 1)
                else:
                    timestamp_ms = float(time_window)
                    flight_data[str(timestamp_ms)] = self.get_flight_data(timestamp_ms, None, 1)
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not convert timestamp '{time_window}' to number: {e}")
                continue
        if experts != []:
            for expert in experts:
                outputs[expert] = self.call_expert(question, expert, flight_data)
            final = self.call_integration(question, outputs)
            final_str = self._json_dump(final) if isinstance(final, dict) else str(final)
            final = self.call_summarizeForUser(question, final_str)
        else:
            final = self.call_general(question)
        return self._json_dump(final)

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

    def get_flight_data(self, timestamp_ms: float, signals: list = None, window_sec: int = 10):
        """
        Get flight data around a timestamp for LLM analysis.
        
        Args:
            timestamp_ms: Target timestamp in milliseconds
            signals: List of signals ['ATT', 'GPS', 'XKQ', 'PARM'] (default: all)
            window_sec: Time window in seconds (default: 10)
            
        Returns:
            dict: Formatted data for LLM analysis
        """
        if not self.log_id:
            return {"error": "No log_id specified"}
        
        try:
            with open(f'data/{self.log_id}.json', 'r') as f:
                log_data = json.load(f)
        except FileNotFoundError:
            return {"error": f"Log file {self.log_id}.json not found"}
        
        if signals is None:
            signals = ['ATT', 'GPS', 'XKQ', 'PARM']
        
        # Calculate time window
        start_ms = timestamp_ms - window_sec * 500
        end_ms = timestamp_ms + window_sec * 500
        
        result = {
            "timestamp": timestamp_ms,
            "window_seconds": window_sec,
            "signals": {},
            "anomalies": [],
            "context": {}
        }
        
        time_series = log_data.get('time_series_data', {})
        
        # Extract data for each signal
        for signal in signals:
            # Handle GPS instances
            if signal == 'GPS':
                data = None
                for i in range(3):
                    if f'GPS[{i}]' in time_series:
                        data = time_series[f'GPS[{i}]']
                        break
            else:
                data = time_series.get(signal)
            
            if not data or 'data' not in data:
                continue
            
            times = data['data'].get('time_boot_ms', [])
            if not times:
                continue
            
            # Find samples in time window
            indices = [i for i, t in enumerate(times) if start_ms <= t <= end_ms]
            if not indices:
                continue
            
            # Extract field data
            signal_data = {
                "samples": len(indices),
                "time_range": [times[indices[0]], times[indices[-1]]],
                "fields": {}
            }
            
            for field, values in data['data'].items():
                if field == 'time_boot_ms' or not isinstance(values, list):
                    continue
                
                window_values = [values[i] for i in indices if i < len(values)]
                if window_values and isinstance(window_values[0], (int, float)):
                    signal_data["fields"][field] = {
                        "values": window_values[:100],  # Limit for LLM
                        "min": min(window_values),
                        "max": max(window_values),
                        "avg": sum(window_values) / len(window_values)
                    }
            
            result["signals"][signal] = signal_data
        
        # Add flight context
        flight_summary = log_data.get('flight_summary', {})
        modes = flight_summary.get('modes', [])
        events = flight_summary.get('events', [])
        messages = flight_summary.get('text_messages', [])
        
        # Find current mode
        current_mode = "Unknown"
        for mode_time, mode in modes:
            if mode_time <= timestamp_ms:
                current_mode = mode
            else:
                break
        
        # Find events in window
        window_events = []
        for event in events:
            if len(event) >= 2 and start_ms <= event[0] <= end_ms:
                window_events.append({"time": event[0], "event": event[1]})
        
        # Find messages in window
        window_messages = []
        for msg in messages:
            if len(msg) >= 3 and start_ms <= msg[0] <= end_ms:
                window_messages.append({"time": msg[0], "message": msg[2]})
        
        result["context"] = {
            "mode": current_mode,
            "events": window_events,
            "messages": window_messages
        }
        
        # Simple anomaly detection
        result["anomalies"] = self._detect_simple_anomalies(result)
        
        return result
    
    def _detect_simple_anomalies(self, data):
        """Simple anomaly detection"""
        anomalies = []
        
        # Check attitude errors
        att = data.get("signals", {}).get("ATT", {})
        if att and "ErrRP" in att.get("fields", {}):
            max_error = att["fields"]["ErrRP"]["max"]
            if max_error > 10:
                anomalies.append(f"Large attitude error: {max_error:.1f}°")
        
        # Check GPS quality
        gps = data.get("signals", {}).get("GPS", {})
        if gps:
            if "NSats" in gps.get("fields", {}):
                min_sats = gps["fields"]["NSats"]["min"]
                if min_sats < 6:
                    anomalies.append(f"Low GPS satellites: {min_sats}")
            
            if "HDop" in gps.get("fields", {}):
                max_hdop = gps["fields"]["HDop"]["max"]
                if max_hdop > 2.0:
                    anomalies.append(f"High GPS HDOP: {max_hdop:.1f}")
        
        # Check for error messages
        for msg in data.get("context", {}).get("messages", []):
            if any(word in msg["message"].lower() for word in ["error", "fail", "fault"]):
                anomalies.append(f"System message: {msg['message']}")
        
        return anomalies

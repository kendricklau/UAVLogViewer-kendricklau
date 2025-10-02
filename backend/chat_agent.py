from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_MODEL = "gpt-4o-mini"
SUMMARIZE_FOR_USER_SYSTEM_PROMPT = """
    You are the summarize for user expert. Focus on summarizing the context and answering the question for the user.
    Respond in a concise human readable format: "summary", "key_findings", "confidence", "follow_ups".
"""
PLANNER_SYSTEM_PROMPT = """
    You are an expert in ArduPilot operations and vehicle diagnosis.
    Given a question and contextual snippets, decide whether the answer can be produced directly or which specialist agents to consult.
    Respond as a compact JSON object with the keys:
      "analysis": brief reasoning that references the context titles when relevant,
      "selected_experts": array of experts to call drawn from ["attitude","gps","ekf","parameters"],
      "direct_answer_ok": boolean indicating if the overview agent alone is sufficient.
    Do not include any additional text outside the JSON object.
"""
        
EXPERT_SYSTEM_PROMPT = {
    "attitude": """
        You are the attitude expert. Focus on ATT messages and related attitude indicators.
        Summarize findings in strict JSON with keys:
          "summary": short description of the key attitude insights,
          "key_findings": array of bullet strings,
          "confidence": value between 0 and 1,
          "follow_ups": array of suggested checks or log segments to inspect next.
        Exclude any text outside the JSON response.
    """,
    "gps": """
        You are the GPS expert. Focus on GPS messages and navigation consistency.
        Return strict JSON with keys: "summary", "key_findings", "confidence", "follow_ups" as described for other experts.
        No additional narration outside the JSON object.
    """,
    "ekf": """
        You are the EKF expert. Focus on XKF and EKF health indicators.
        Output strict JSON with "summary", "key_findings", "confidence", "follow_ups" (array).
        Do not add extra prose outside the JSON object.
    """,
    "parameters": """
        You are the parameters expert. Concentrate on PARM messages and configuration anomalies.
        Respond in JSON with keys: "summary", "key_findings", "confidence", "follow_ups".
        The response must be valid JSON only.
    """,
}

INTEGRATION_SYSTEM_PROMPT = """
    You are the integration expert. Combine the structured outputs from the other experts, the question, and the shared context to answer the user's question while assessing whether more specialists are needed.
    Produce strict JSON with keys:
      "answer": concise response to the question,
      "evidence": array of strings citing which expert/context elements support the answer,
      "additional_experts": array of extra expert names to consult next, chosen from ["attitude","gps","ekf","parameters"] and excluding any already present in Expert Responses (use [] when none are needed),
      "risks": optional array highlighting uncertainties or missing data.
    Only output JSON, no additional commentary.
"""

OVERVIEW_SYSTEM_PROMPT = """
    You are the overview expert. Provide a high-level flight overview using strict JSON with keys:
      "summary": short narrative,
      "notable_events": array of key bullet points,
      "data_gaps": array of missing or uncertain aspects (allow empty array).
    Do not include any content outside the JSON object.
"""
# TODO: Make it so that we're not sending the whole rag context each time, and ensuring we keep track of previous chats, even from the agents not outputting to the user.
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
MAX_INPUT_TOKENS = 6000
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
    
    def call_summarizeForUser(self, question: str):
        rag_docs = self.get_rag_docs(self.log_id)
        context = "\n\n".join([f"Title: {doc.get('title', '')}\nContent: {doc.get('content', '')}" for doc in rag_docs])
        context = self._truncate_text(context, MAX_INPUT_TOKENS)
        prompt_question = self._truncate_text(question, MAX_INPUT_TOKENS)
        
        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SUMMARIZE_FOR_USER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_question}
            ],
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        content = response.choices[0].message.content
        _append_agent_chat_to_history(self.log_id, "summarizeForUser", question, content)
        return content

    def call_overview(self, question: str):
        rag_docs = self.get_rag_docs(self.log_id)
        context = "\n\n".join([f"Title: {doc.get('title', '')}\nContent: {doc.get('content', '')}" for doc in rag_docs])
        context = self._truncate_text(context, MAX_INPUT_TOKENS)
        prompt_question = self._truncate_text(question, MAX_INPUT_TOKENS)
        
        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": OVERVIEW_SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {prompt_question}\n\nContext:\n{context}"}
            ],
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        response_text = response.choices[0].message.content
        _append_agent_chat_to_history(self.log_id, "overview", question, response_text)
        return self._safe_parse_json(response_text)

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

        # Define the known experts based on EXPERT_SYSTEM_PROMPT keys.
        known_experts = list(EXPERT_SYSTEM_PROMPT.keys())

        identified_experts = []
        for expert_name in known_experts:
            if expert_name in planner_response_content.lower():
                identified_experts.append(expert_name)
        
        return identified_experts

    def call_expert(self, question: str, expert: str):
        rag_docs = self.get_rag_docs(self.log_id, expert)
        context = "\n\n".join([f"Title: {doc.get('title', '')}\nContent: {doc.get('content', '')}" for doc in rag_docs])
        context = self._truncate_text(context, MAX_INPUT_TOKENS)
        prompt_question = self._truncate_text(question, MAX_INPUT_TOKENS)
        
        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": EXPERT_SYSTEM_PROMPT[expert]},
                {"role": "user", "content": f"Question: {prompt_question}\n\nContext:\n{context}"}
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
        known_experts = set(EXPERT_SYSTEM_PROMPT.keys())
        collected_responses = dict(expert_responses)
        max_iterations = len(known_experts) + 1
        last_parsed = None

        for _ in range(max_iterations):
            expert_payload_sections = []
            expert_count = max(1, len(collected_responses))
            per_expert_budget = max(1, MAX_INPUT_TOKENS // expert_count)
            for expert_name, payload in collected_responses.items():
                payload_text = self._truncate_text(self._json_dump(payload), per_expert_budget)
                expert_payload_sections.append(f"{expert_name}: {payload_text}")
            expert_payload = self._truncate_text("\n".join(expert_payload_sections), MAX_INPUT_TOKENS)
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": INTEGRATION_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Question: {prompt_question}\n\nContext:\n{context}\n\nExpert Responses:\n{expert_payload}"}
                ],
                max_tokens=MAX_OUTPUT_TOKENS,
            )

            content = response.choices[0].message.content
            last_parsed = self._safe_parse_json(content)

            if not isinstance(last_parsed, dict):
                return last_parsed

            requested = last_parsed.get("additional_experts") or []
            if not isinstance(requested, list):
                return last_parsed

            new_experts = []
            for expert_name in requested:
                if not isinstance(expert_name, str):
                    continue
                normalized = expert_name.strip().lower()
                if (
                    not normalized
                    or normalized not in known_experts
                    or normalized in collected_responses
                ):
                    continue
                new_experts.append(normalized)

            if not new_experts:
                return last_parsed

            for expert_name in new_experts:
                collected_responses[expert_name] = self.call_expert(question, expert_name)

        return last_parsed

    def debug_chatbot(self, question, log_window=None):
        experts = self.planner(question)
        outputs = {}
        if not experts:
            summarize_for_user_response = self.call_summarizeForUser(question)
            return self._json_dump(summarize_for_user_response)
        for expert_name in experts:
            outputs[expert_name] = self.call_expert(question, expert_name)
        final = self.call_integration(question, outputs)
        final_str = self._json_dump(final) if isinstance(final, dict) else str(final)
        final = self.call_summarizeForUser(final_str)
        return self._json_dump(final)

    def get_rag_docs(self, log_id, document_type=None):
        with open(f'data/{log_id}_rag.json', 'r') as f:
            docs = json.load(f)['documents']
        
        if document_type:
            import re
            # Filter documents that match the document_type pattern
            filtered_docs = []
            for doc in docs:
                if 'document_type' in doc and re.search(document_type, doc['document_type'], re.IGNORECASE):
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

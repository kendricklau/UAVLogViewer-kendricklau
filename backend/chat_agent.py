from openai import OpenAI   
import json
from dotenv import load_dotenv

load_dotenv()

OPENAI_MODEL = "gpt-4o-mini"

PLANNER_SYSTEM_PROMPT = """
    You are an expert in ArduPilot. You are given a question and a context. You need to answer the question based on the context.
    You will first analyze the question and then decide which expert to call. If we can figure out the answer from the context, you will not call an expert.
    You will then integrate the responses of the experts and return the final answer.
    The experts are: attitude, gps, ekf, and parameters agents
"""
        
EXPERT_SYSTEM_PROMPT = {
    "attitude": "You are the attitude expert. Focus on the ATT messages",
    "gps": "You are the gps expert. Focus on the GPS messages",
    "ekf": "You are the ekf expert. Focus on the XKQ messages",
    "parameters": "You are the parameters expert. Focus on the PARM messages",
}

INTEGRATION_SYSTEM_PROMPT = """
    You are the integration expert. You will combine the attitude, gps, ekf, and parameters expert responses to answer the question.
"""

OVERVIEW_SYSTEM_PROMPT = """
    You are the overview expert. You will provide a high-level overview of the flight based on the context.
"""

class ChatAgent:
    def __init__(self, log_id=None):
        self.client = OpenAI()
        self.log_id = log_id

    def ask(self, question: str):
        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": question}]
        )
        return response.choices[0].message.content

    def call_overview(self, question: str):
        rag_docs = self.get_rag_docs(self.log_id)
        context = "\n\n".join([f"Title: {doc.get('title', '')}\nContent: {doc.get('content', '')}" for doc in rag_docs])
        
        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": OVERVIEW_SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"}
            ]
        )
        return response.choices[0].message.content

    def planner(self, question: str):
        rag_docs = self.get_rag_docs(self.log_id)
        context = "\n\n".join([f"Title: {doc.get('title', '')}\nContent: {doc.get('content', '')}" for doc in rag_docs])
        
        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"}
            ]
        )
        planner_response_content = response.choices[0].message.content

        # Define the known experts based on EXPERT_SYSTEM_PROMPT keys.
        # This dictionary is available in the global scope of this module.
        known_experts = list(EXPERT_SYSTEM_PROMPT.keys())

        identified_experts = []
        for expert_name in known_experts:
            if expert_name in planner_response_content.lower():
                identified_experts.append(expert_name)
        
        return identified_experts

    def call_expert(self, question: str, expert: str):
        rag_docs = self.get_rag_docs(self.log_id, expert)
        context = "\n\n".join([f"Title: {doc.get('title', '')}\nContent: {doc.get('content', '')}" for doc in rag_docs])
        
        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": EXPERT_SYSTEM_PROMPT[expert]},
                {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"}
            ]
        )
        return response.choices[0].message.content

    def call_integration(self, question: str, expert_responses: list):
        rag_docs = self.get_rag_docs(self.log_id)
        context = "\n\n".join([f"Title: {doc.get('title', '')}\nContent: {doc.get('content', '')}" for doc in rag_docs])
        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": INTEGRATION_SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}\n\nExpert Responses:\n{expert_responses}"}
            ]
        )
        # TODO: Integration agent should be able to call other experts based on the analysis. Need to add system prompt to reason if we need to talk to any other experts or look at specific logs to see if we can figure out what issues the rest of the vehicle might be able to tell us
        return response.choices[0].message.content

    def debug_chatbot(self, question, log_window):
        experts = self.planner(question)
        outputs = {}
        if experts == []:
            return self.call_overview(question)
        for e in experts:
            outputs[e] = self.call_expert(question, e)
        final = self.call_integration(question, outputs)
        return final

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
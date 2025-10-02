from chat_agent import ChatAgent
import json
import os

def test_agent_flow():
    """Test the full agent flow by calling debug_chatbot and logging each step"""
    # Get available log
    data_dir = "data"
    rag_files = [f for f in os.listdir(data_dir) if f.endswith('_rag.json')]
    
    if not rag_files:
        print("No RAG files found. Please upload a log file first.")
        return
    
    log_id = rag_files[0].replace('_rag.json', '')
    print(f"Testing agent flow with log ID: {log_id}")
    
    # Create a simple wrapper that logs each step
    class LoggingAgent(ChatAgent):
        def planner(self, question: str):
            print(f"\n{'='*60}")
            print("STEP: PLANNER")
            print(f"{'='*60}")
            print(f"Question: {question}")
            
            result = super().planner(question)
            print(f"Response: {result}")
            return result
        
        def call_expert(self, question: str, expert: str, flight_data: dict):
            print(f"\n{'='*60}")
            print(f"STEP: EXPERT_{expert.upper()}")
            print(f"{'='*60}")
            print(f"Question: {question}")
            print(f"Expert: {expert}")
            print(f"Flight data keys: {list(flight_data.keys()) if flight_data else 'None'}")
            
            result = super().call_expert(question, expert, flight_data)
            print(f"Response: {json.dumps(result, indent=2)}")
            return result
        
        def call_integration(self, question: str, expert_responses: dict):
            print(f"\n{'='*60}")
            print("STEP: INTEGRATION")
            print(f"{'='*60}")
            print(f"Question: {question}")
            print(f"Expert responses keys: {list(expert_responses.keys())}")
            
            result = super().call_integration(question, expert_responses)
            print(f"Response: {json.dumps(result, indent=2)}")
            return result
        
        def call_summarizeForUser(self, question: str):
            print(f"\n{'='*60}")
            print("STEP: SUMMARIZE_FOR_USER")
            print(f"{'='*60}")
            print(f"Question: {question}")
            
            result = super().call_summarizeForUser(question)
            print(f"Response: {result}")
            return result
    
    # Test with one question
    agent = LoggingAgent(log_id=log_id)
    question = "What happened at 100 seconds into the flight?"
    
    print(f"TESTING QUESTION: {question}")
    print(f"LOG ID: {log_id}")
    
    # Run the full flow
    result = agent.debug_chatbot(question)
    
    print(f"\n{'='*60}")
    print("FINAL RESULT")
    print(f"{'='*60}")
    print(result)

if __name__ == "__main__":
    test_agent_flow()

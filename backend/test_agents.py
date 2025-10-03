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
            print(f"Raw Response: {result}")
            
            # Parse the JSON response for better display
            try:
                parsed_result = self._safe_parse_json(result)
                print(f"Parsed Response: {json.dumps(parsed_result, indent=2)}")
            except Exception as e:
                print(f"Error parsing planner response: {e}")
            
            return result
        
        def call_expert(self, question: str, expert: str, timestamp_ms: list):
            print(f"\n{'='*60}")
            print(f"STEP: EXPERT_{expert.upper()}")
            print(f"{'='*60}")
            print(f"Question: {question}")
            print(f"Expert: {expert}")
            print(f"Timestamps: {timestamp_ms}")
            
            result = super().call_expert(question, expert, timestamp_ms)
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
        
        def call_summarizeForUser(self, question: str, integration_result: str):
            print(f"\n{'='*60}")
            print("STEP: SUMMARIZE_FOR_USER")
            print(f"{'='*60}")
            print(f"Question: {question}")
            print(f"Integration Result: {integration_result}")
            
            result = super().call_summarizeForUser(question, integration_result)
            print(f"Response: {result}")
            return result
        
        def call_general(self, question: str):
            print(f"\n{'='*60}")
            print("STEP: GENERAL")
            print(f"{'='*60}")
            print(f"Question: {question}")
            
            result = super().call_general(question)
            print(f"Response: {result}")
            return result
    
    # Test with one question
    agent = LoggingAgent(log_id=log_id)
    question = "What was the speed at time 3m45s?"
    
    print(f"TESTING QUESTION: {question}")
    print(f"LOG ID: {log_id}")
    
    # Run the full flow
    result = agent.debug_chatbot(question)
    
    print(f"\n{'='*60}")
    print("FINAL RESULT")
    print(f"{'='*60}")
    print(result)

def test_specific_questions():
    """Test with different types of questions"""
    data_dir = "data"
    rag_files = [f for f in os.listdir(data_dir) if f.endswith('_rag.json')]
    
    if not rag_files:
        print("No RAG files found. Please upload a log file first.")
        return
    
    log_id = rag_files[0].replace('_rag.json', '')
    agent = ChatAgent(log_id=log_id)
    
    test_questions = [
        "What was the roll and pitch at 225000ms?",
        "Show me the GPS data for the entire flight",
        "What happened during the mode changes?",
        "Are there any errors or warnings in the flight?",
        "What was the altitude during takeoff?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}: {question}")
        print(f"{'='*80}")
        
        try:
            result = agent.debug_chatbot(question)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("Running basic agent flow test...")
    test_agent_flow()
    
    print("\n" + "="*80)
    print("Running multiple question tests...")
    test_specific_questions()
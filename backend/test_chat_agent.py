from chat_agent import ChatAgent
import json
import os

def test_chat_agent():
    # Get available log IDs
    data_dir = "data"
    rag_files = [f for f in os.listdir(data_dir) if f.endswith('_rag.json')]
    
    if not rag_files:
        print("No RAG files found. Please upload a log file first.")
        return
    
    # Use the first available log
    log_id = rag_files[0].replace('_rag.json', '')
    print(f"Testing with log ID: {log_id}")
    
    # Initialize chat agent
    agent = ChatAgent(log_id=log_id)
    
    # Test prompts
    test_prompts = [
        "What was the flight duration?",
        "Were there any GPS issues?",
        "Show me attitude control performance",
        "What errors occurred during flight?",
        "Tell me about the vehicle parameters"
    ]
    
    for prompt in test_prompts:
        print(f"\n{'='*50}")
        print(f"Question: {prompt}")
        print(f"{'='*50}")
        
        try:
            response = agent.debug_chatbot(prompt, log_window=None)
            print(f"Answer: {response}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_chat_agent()
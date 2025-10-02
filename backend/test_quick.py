import chat_agent


ChatAgent = chat_agent.ChatAgent()
ChatAgent.log_id = "c450d131-d45c-4bb8-908d-60df8aa9ddbe"

list_of_test_questions = [
    "What was the max speed of the entire flight?"
]
for question in list_of_test_questions:
    full_resp = ChatAgent.debug_chatbot(question)
    print("question", question, "\nfull_resp", full_resp)

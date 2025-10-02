# backend/app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import json
import os
from datetime import datetime
import uuid

from rag_docs_generation import RAGDocsGenerator
from chat_agent import ChatAgent

app = FastAPI(title="UAV Logs Chatbot Backend", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create data directory in backend if it doesn't exist
os.makedirs("data", exist_ok=True)

# Define the LogData model with Pydantic
class LogData(BaseModel):
    filename: str
    logType: str
    metadata: Dict[str, Any]
    messages: Dict[str, Any]
    flightModes: List[Any]  
    events: List[Any]       
    mission: List[Any]      
    vehicle: str
    parameters: Optional[Dict[str, Any]] = None  
    defaultParams: Optional[Dict[str, Any]] = None
    textMessages: List[Any]  
    fences: List[Any]        
    attitudeSources: Dict[str, Any]  
    attitudeSource: str
    timeAttitude: List[Any]  
    timeAttitudeQ: Dict[str, Any]  
    trajectorySources: List[str]
    trajectorySource: str
    trajectories: Dict[str, Any]
    currentTrajectory: List[Any]  
    timeTrajectory: Dict[str, Any]
    lastTime: Optional[int] = None  
    timestamp: str

class ChatRequest(BaseModel):
    log_id: str
    question: str

class ChatResponse(BaseModel):
    answer: str
    log_id: str
    message_id: str

class FlightDataRequest(BaseModel):
    log_id: str
    timestamp_ms: float
    signals: Optional[List[str]] = None
    window_seconds: int = 10

# Define the upload_log_data endpoint
@app.post("/api/logs/upload")
async def upload_log_data(data: dict):
    try:
        log_id = str(uuid.uuid4())
        
        # Save as structured JSON for RAG
        rag_data = {
            "log_id": log_id,
            "filename": data.get("filename", "unknown"),
            "log_type": data.get("logType", "unknown"),
            "vehicle": data.get("vehicle", "unknown"),
            "flight_duration_ms": data.get("lastTime") or 0,
            "upload_time": data.get("timestamp", ""),
            
            # Time series data by message type
            "time_series_data": {
                msg_type: {
                    "fields": list(msg_data.keys()) if isinstance(msg_data, dict) else [],
                    "sample_count": len(msg_data.get("time_boot_ms", [])) if isinstance(msg_data, dict) else 0,
                    "time_range": {
                        "start": min(msg_data.get("time_boot_ms", [0])) if isinstance(msg_data, dict) and msg_data.get("time_boot_ms") else 0,
                        "end": max(msg_data.get("time_boot_ms", [0])) if isinstance(msg_data, dict) and msg_data.get("time_boot_ms") else 0
                    },
                    "data": msg_data
                }
                for msg_type, msg_data in data.get("messages", {}).items()
                if msg_data and isinstance(msg_data, dict) and "time_boot_ms" in msg_data
            },
            
            # Extracted flight data
            "flight_summary": {
                "modes": data.get("flightModes", []),
                "events": data.get("events", []),
                "mission": data.get("mission", []),
                "text_messages": data.get("textMessages", []),
                "fences": data.get("fences", []),
                "attitude_sources": data.get("attitudeSources", {}),
                "trajectory_sources": data.get("trajectorySources", [])
            },
            
            # Parameters
            "parameters": data.get("parameters"),
            "default_parameters": data.get("defaultParams")
        }
        
        # Save to file
        with open(f'data/{log_id}.json', 'w') as f:
            json.dump(rag_data, f, indent=2)
        
        # Generate RAG documents using the RAGDocsGenerator
        rag_generator = RAGDocsGenerator()
        rag_documents = rag_generator.generate_rag_documents(rag_data)
        
        # Save RAG documents
        rag_file = f'data/{log_id}_rag.json'
        with open(rag_file, 'w') as f:
            json.dump({
                "log_id": log_id,
                "documents": rag_documents,
                "created_at": datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"Successfully generated {len(rag_documents)} RAG documents for log {log_id}")
        
        return {
            "status": "success",
            "log_id": log_id,
            "message": "Log data saved for RAG analysis"
        }
        
    except Exception as e:
        print(f"Error processing log data: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/logs/debug")
async def debug_log_data(data: dict):
    print("Received data keys:", list(data.keys()))
    print("Data types:", {k: type(v) for k, v in data.items()})
    return {"received": True, "keys": list(data.keys())}

@app.post("/api/chat/ask", response_model=ChatResponse)
async def ask_chat_with_context(request: ChatRequest):
    try:
        # Check if log_id exists
        rag_file = f'data/{request.log_id}_rag.json'
        if not os.path.exists(rag_file):
            raise HTTPException(status_code=404, detail=f"Log {request.log_id} not found")
        
        # Create chat agent with log context
        agent = ChatAgent(log_id=request.log_id)
        
        # Get response
        answer = agent.debug_chatbot(request.question)
        
        # Generate message ID
        message_id = str(uuid.uuid4())
        
        # Append chat history to RAG documents
        await append_chat_to_rag(request.log_id, request.question, answer, message_id)
        
        return ChatResponse(
            answer=answer, 
            log_id=request.log_id,
            message_id=message_id
        )
    except Exception as e:
        print(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# New function to append chat to single RAG document
async def append_chat_to_rag(log_id: str, question: str, answer: str, message_id: str):
    """Append chat conversation to single chat_history document"""
    rag_file = f'data/{log_id}_rag.json'
    
    try:
        # Load existing RAG documents
        with open(rag_file, 'r') as f:
            rag_data = json.load(f)
        
        # Find existing chat_history document or create new one
        chat_doc = None
        chat_doc_index = None
        
        for i, doc in enumerate(rag_data["documents"]):
            if doc.get("document_type") == "chat_history":
                chat_doc = doc
                chat_doc_index = i
                break
        
        # Create new chat entry
        timestamp = datetime.now().isoformat()
        chat_entry = f"\n\n--- Chat Entry {timestamp} ---\nUser: {question}\nAssistant: {answer}"
        
        if chat_doc is None:
            # Create new chat_history document
            chat_doc = {
                "document_id": f"{log_id}_chat_history",
                "document_type": "chat_history",
                "title": f"Chat History - {log_id}",
                "content": f"Chat History for Flight Log {log_id}{chat_entry}",
                "metadata": {
                    "created_at": timestamp,
                    "last_updated": timestamp,
                    "message_count": 1,
                    "log_id": log_id
                }
            }
            rag_data["documents"].append(chat_doc)
        else:
            # Append to existing chat_history document
            chat_doc["content"] += chat_entry
            chat_doc["metadata"]["last_updated"] = timestamp
            chat_doc["metadata"]["message_count"] = chat_doc["metadata"].get("message_count", 0) + 1
            
            # Update the document in the list
            rag_data["documents"][chat_doc_index] = chat_doc
        
        # Save back to file
        with open(rag_file, 'w') as f:
            json.dump(rag_data, f, indent=2)
            
        print(f"Chat history appended to RAG for log {log_id}")
        
    except Exception as e:
        print(f"Error appending chat to RAG: {e}")
        # Don't fail the chat request if RAG append fails

@app.get("/api/chat/history/{log_id}")
async def get_chat_history(log_id: str):
    """Get chat history from single chat_history document"""
    try:
        rag_file = f'data/{log_id}_rag.json'
        if not os.path.exists(rag_file):
            return {"messages": []}
        
        with open(rag_file, 'r') as f:
            rag_data = json.load(f)
        
        # Find chat_history document
        chat_doc = None
        for doc in rag_data["documents"]:
            if doc.get("document_type") == "chat_history":
                chat_doc = doc
                break
        
        if not chat_doc:
            return {"messages": []}
        
        # Parse chat entries from content
        messages = []
        content = chat_doc["content"]
        
        # Split by chat entry separators
        entries = content.split("--- Chat Entry")[1:]  # Skip the title line
        
        for entry in entries:
            lines = entry.strip().split('\n')
            if len(lines) >= 3:
                timestamp_line = lines[0].strip()
                user_line = lines[1].strip()
                assistant_line = lines[2].strip()
                
                if user_line.startswith("User:") and assistant_line.startswith("Assistant:"):
                    messages.append({
                        "timestamp": timestamp_line,
                        "question": user_line[6:],  # Remove "User: " prefix
                        "answer": assistant_line[11:],  # Remove "Assistant: " prefix
                        "log_id": log_id
                    })
        
        return {"messages": messages}
        
    except Exception as e:
        print(f"Error retrieving chat history: {e}")
        return {"messages": []}

@app.post("/api/flight-data")
async def get_flight_data(request: FlightDataRequest):
    """Get flight data around a specific timestamp"""
    try:
        rag_file = f'data/{request.log_id}_rag.json'
        if not os.path.exists(rag_file):
            raise HTTPException(status_code=404, detail=f"Log {request.log_id} not found")
        
        agent = ChatAgent(log_id=request.log_id)
        data = agent.get_flight_data(
            timestamp_ms=request.timestamp_ms,
            signals=request.signals,
            window_sec=request.window_seconds
        )
        
        return data
        
    except Exception as e:
        print(f"Error getting flight data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/flight-events/{log_id}")
async def get_flight_events(log_id: str):
    """Get all flight events for a log"""
    try:
        rag_file = f'data/{log_id}_rag.json'
        if not os.path.exists(rag_file):
            raise HTTPException(status_code=404, detail=f"Log {log_id} not found")
        
        with open(f'data/{log_id}.json', 'r') as f:
            log_data = json.load(f)
        
        events = []
        flight_summary = log_data.get('flight_summary', {})
        
        # Add mode changes
        for mode_time, mode in flight_summary.get('modes', []):
            events.append({
                "timestamp_ms": mode_time,
                "type": "mode_change",
                "description": f"Mode change to {mode}"
            })
        
        # Add flight events
        for event in flight_summary.get('events', []):
            if len(event) >= 2:
                events.append({
                    "timestamp_ms": event[0],
                    "type": "flight_event",
                    "description": event[1]
                })
        
        # Add important messages
        for msg in flight_summary.get('text_messages', []):
            if len(msg) >= 3:
                message_text = msg[2].lower()
                if any(word in message_text for word in ["error", "fail", "fault", "warning", "armed", "disarmed"]):
                    events.append({
                        "timestamp_ms": msg[0],
                        "type": "system_message",
                        "description": msg[2]
                    })
        
        events.sort(key=lambda x: x["timestamp_ms"])
        
        return {"log_id": log_id, "events": events}
        
    except Exception as e:
        print(f"Error getting flight events: {e}")
        raise HTTPException(status_code=500, detail=str(e))
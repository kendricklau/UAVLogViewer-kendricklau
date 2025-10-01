# backend/app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import json
import os
from datetime import datetime
import uuid

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
            "flight_duration_ms": data.get("lastTime", 0),
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
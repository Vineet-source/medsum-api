
from dotenv import load_dotenv
load_dotenv()

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from agents.graph import app_graph
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load AI Graph. Check your .env keys. Error: {e}")
    app_graph = None

app = FastAPI(title="Doctor's Friend Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "online", "message": "Doctor's Friend API is running"}

@app.get("/ask")
async def ask_doctor_query(question: str):
    if not app_graph:
        return {"summary": "Backend Error: AI Graph not initialized.", "sources": []}
    
    inputs = {"query": question}
    
    try:
        result = await app_graph.ainvoke(inputs)
        
        # Ensure we return keys that match your Flutter 'data['summary']'
        return {
            "summary": result.get("summary", "No summary generated."),
            "sources": result.get("verified_sources", [])
        }
    except Exception as e:
        print(f"Execution Error: {e}")
        return {"summary": f"Error during analysis: {str(e)}", "sources": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
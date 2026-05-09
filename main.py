from dotenv import load_dotenv
load_dotenv()

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 1. Load the AI Graph
try:
    from agents.graph import app_graph
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load AI Graph. Check your .env keys. Error: {e}")
    app_graph = None

# 2. Load the Cache System (THIS WAS MISSING!)
try:
    from semantic_cache import cache_system
except ImportError as e:
    print(f"Cache module not found, running without cache. Error: {e}")
    cache_system = None

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
    
    if cache_system:
        try:
            cached_result = cache_system.check_cache(question)
            if cached_result:
                return cached_result  # INSTANT RETURN (Skips LangGraph entirely)
        except Exception as cache_error:
            print(f"Cache check failed, proceeding to LangGraph. Error: {cache_error}")
    # ==========================================

    inputs = {"query": question}
    
    try:
        result = await app_graph.ainvoke(inputs)
        
        # Ensure we return keys that match your Flutter 'data['summary']'
        final_response = {
            "summary": result.get("summary", "No summary generated."),
            "sources": result.get("verified_sources", [])
        }

        if cache_system:
            try:
                cache_system.add_to_cache(question, final_response)
            except Exception as save_error:
                print(f"Warning: Could not save to cache. Error: {save_error}")
        # ==========================================

        return final_response

    except Exception as e:
        print(f"Execution Error: {e}")
        return {"summary": f"Error during analysis: {str(e)}", "sources": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
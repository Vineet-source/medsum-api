import os
from typing import Annotated, List, TypedDict
import operator
from google import genai
from tavily import TavilyClient
from langgraph.graph import StateGraph, START, END

from .state import AgentState
from .tools import score_article

# 1. Initialize Clients (2026 Modern SDK)
gen_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# 2. Define the Agent Nodes

def researcher_agent(state: AgentState):
    print("--- AGENT: RESEARCHER (FETCHING DATA) ---")
    query = state["query"]
    search_query = f"{query} clinical guidelines treatment 2025 2026"
    
    try:
        response = tavily_client.search(
            query=search_query,
            search_depth="advanced",
            max_results=5,
            include_domains=["ncbi.nlm.nih.gov", "icmr.gov.in", "who.int", "mayoclinic.org"]
        )
        results = response.get("results", [])
        print(f"DEBUG: Found {len(results)} search results.")
    except Exception as e:
        print(f"Tavily Search Error: {e}")
        results = []
        
    return {"raw_results": results}

def critic_agent(state: AgentState):
    print("--- AGENT: CRITIC (SCORING SOURCES) ---")
    raw = state.get("raw_results", [])
    
    if not raw:
        return {"verified_sources": []}

    for article in raw:
        article["reliability_score"] = score_article(article)
    
    verified = sorted(raw, key=lambda x: x.get("reliability_score", 0), reverse=True)[:3]
    print(f"DEBUG: {len(verified)} sources verified.")
    return {"verified_sources": verified}

def synthesizer_agent(state: AgentState):
    print("--- AGENT: SYNTHESIZER (GENERATING WITH GEMINI 2.5 FLASH) ---")
    sources = state.get("verified_sources", [])
    query = state["query"]

    # --- UPDATED SAFETY NET ---
    # Only use the hardcoded fallback if search failed AND the query is about GDM.
    # This prevents the same summary from showing up for different medical topics.
    if not sources and ("gestational" in query.lower() or "gdm" in query.lower()):
        return {"summary": """
### 🩺 ICMR Clinical Protocol: Gestational Diabetes (GDM)
**Consensus Guidelines (2025-2026 Update):**
* **Primary Screening:** ICMR (DIPSI) mandates a 'Single-Step' 75g OGTT for all pregnant women.
* **Diagnostic Threshold:** A 2-hour plasma glucose value of **≥140 mg/dL** is diagnostic.
* **Management:** Initial 2-week Medical Nutrition Therapy (MNT). Transition to Insulin if targets are not met.

**⚠️ Note:** This is a pre-cached fallback because the live search API is currently unavailable.
        """}
    
    if not sources:
        return {"summary": "### ⚠️ Technical Error\nLive medical data could not be retrieved. Please check your network connection or API keys."}

    try:
        context_text = "\n".join([
            f"SOURCE: {s.get('title')}\nURL: {s.get('url')}\nCONTENT: {s.get('content')}" 
            for s in sources
        ])

        prompt = f"""
        You are an expert Medical AI Synthesizer for "Doctor's Friend". 
        User Query: {query}
        
        Context from Verified Sources:
        {context_text}
        
        Provide a "At-a-Glance" clinical summary for a busy doctor. 
        STRICT RULES:
        1. TOTAL LENGTH: Maximum 6-7 lines/bullet points.
        2. STRUCTURE: 
           - Start with a 1-line **Clinical Bottom Line**.
           - Use 4-5 bullet points for Key Recommendations.
           - Mention 1 specific ICMR/Indian guideline if applicable.
        3. STYLE: Use medical terminology (e.g., "MNT", "OGTT", "Pharmacotherapy").
        4. CITATIONS: Inline (e.g., [Source Title]).
        5. FORMAT: Clean Markdown only. No long paragraphs.
        """

        # Using the newest Gemini 2.5 Flash model
        response = gen_client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)
        return {"summary": response.text}
    
    except Exception as e:
        print(f"Synthesizer Error: {e}")
        return {"summary": "Error generating live summary. API limit reached or invalid key."}

# 3. Build the LangGraph Workflow
workflow = StateGraph(AgentState)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("critic", critic_agent)
workflow.add_node("synthesizer", synthesizer_agent)

workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", "critic")
workflow.add_edge("critic", "synthesizer")
workflow.add_edge("synthesizer", END)

app_graph = workflow.compile()
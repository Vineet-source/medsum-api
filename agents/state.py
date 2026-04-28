from typing import List, TypedDict, Annotated
import operator

class AgentState(TypedDict):
    query: str
    raw_results: Annotated[List[dict], operator.add] 
    verified_sources: List[dict]
    summary: str
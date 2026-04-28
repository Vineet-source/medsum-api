import os

def score_article(article):
    """
    Logic to pick the 'better' article. 
    A higher score wins.
    """
    # Start with the AI search engine's base confidence (0-100)
    # Tavily returns 'score' as a float (e.g., 0.85). We convert to 0-100.
    score = article.get("score", 0) * 100 
    url = article.get("url", "").lower()
    
    # 1. High-Authority Global Domains
    if any(d in url for d in ["ncbi.nlm.nih.gov", "thelancet.com", "mayoclinic.org", "who.int", "cdc.gov"]):
        score += 30
    
    # 2. Local Indian Context (Crucial for your problem statement!)
    if any(d in url for d in ["icmr.gov.in", "nhp.gov.in", "aiims.edu", "gov.in"]):
        score += 35
        
    # 3. Penalize non-medical or commercial blogs/social media
    if any(d in url for d in ["medium.com", "facebook.com", "reddit.com", "quora.com"]):
        score -= 50
        
    return min(score, 100) # Cap at 100
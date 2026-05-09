import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading Embedding Model (This takes a few seconds on startup)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

CACHE_FILE = "semantic_cache.json"
SIMILARITY_THRESHOLD = 0.90  

class SemanticCache:
    def __init__(self):
        self.cache = []
        self.load_cache()

    def load_cache(self):
        """Loads previous searches from the local JSON file."""
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                self.cache = json.load(f)

    def save_cache(self):
        """Saves new searches to the local JSON file."""
        with open(CACHE_FILE, 'w') as f:
            json.dump(self.cache, f)

    def check_cache(self, new_query):
        """Checks if a mathematically similar question was asked recently."""
        if not self.cache:
            return None

        new_embedding = model.encode([new_query])[0]
        
        cached_embeddings = [np.array(item["embedding"]) for item in self.cache]

        similarities = cosine_similarity([new_embedding], cached_embeddings)[0]
        
        best_match_idx = np.argmax(similarities)
        best_score = similarities[best_match_idx]

        if best_score >= SIMILARITY_THRESHOLD:
            print(f"✅ CACHE HIT! Similarity: {best_score * 100:.2f}%")
            return self.cache[best_match_idx]["response"]

        print(f"❌ CACHE MISS. Best match was only {best_score * 100:.2f}%. Routing to LangGraph.")
        return None

    def add_to_cache(self, query, response):
        """Saves a new LangGraph response to the cache."""
        embedding = model.encode([query])[0].tolist()
        self.cache.append({
            "query": query,
            "embedding": embedding,
            "response": response
        })
        self.save_cache()

cache_system = SemanticCache()
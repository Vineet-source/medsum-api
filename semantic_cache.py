import json
import os
import math
from google import genai

# Initialize the Gemini Client for embeddings
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

CACHE_FILE = "semantic_cache.json"
SIMILARITY_THRESHOLD = 0.90  # 90% match required

def calculate_similarity(vec1, vec2):
    """Pure Python cosine similarity to avoid heavy memory libraries."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    return dot_product / (magnitude1 * magnitude2)

class SemanticCache:
    def __init__(self):
        self.cache = []
        self.load_cache()

    def load_cache(self):
        """Loads previous searches from the JSON file."""
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'r') as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = []

    def save_cache(self):
        """Saves new searches to the JSON file."""
        with open(CACHE_FILE, 'w') as f:
            json.dump(self.cache, f)

    def get_embedding(self, text):
        """Offloads the heavy math to Google's free embedding API."""
        response = client.models.embed_content(
            model='embedding-001',
            contents=text,
        )
        return response.embeddings[0].values

    def check_cache(self, new_query):
        """Checks if a mathematically similar question was asked recently."""
        if not self.cache:
            return None

        try:
            new_embedding = self.get_embedding(new_query)
            
            best_score = 0
            best_match = None

            # Compare against all cached vectors
            for item in self.cache:
                score = calculate_similarity(new_embedding, item["embedding"])
                if score > best_score:
                    best_score = score
                    best_match = item

            if best_score >= SIMILARITY_THRESHOLD:
                print(f"✅ CACHE HIT! Similarity: {best_score * 100:.2f}%")
                return best_match["response"]

            print(f"❌ CACHE MISS. Best match was only {best_score * 100:.2f}%. Routing to LangGraph.")
            return None
            
        except Exception as e:
            print(f"Embedding API Error: {e}")
            return None

    def add_to_cache(self, query, response):
        """Saves a new LangGraph response to the cache."""
        try:
            embedding = self.get_embedding(query)
            self.cache.append({
                "query": query,
                "embedding": embedding,
                "response": response
            })
            self.save_cache()
        except Exception as e:
            print(f"Failed to save to cache: {e}")

# Global instance for FastAPI
cache_system = SemanticCache()
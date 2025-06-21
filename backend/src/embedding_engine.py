import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
import json
import hashlib

@dataclass
class EmbeddingResult:
    """Result of embedding generation"""
    embedding: np.ndarray
    text: str
    embedding_type: str
    metadata: Dict[str, Any]

class CodeEmbedder:
    """Generate embeddings for code changes using CodeBERT"""
    
    def __init__(self, model_name: str = "microsoft/codebert-base"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
    
    def embed_code(self, code: str, max_length: int = 512) -> np.ndarray:
        """Generate embedding for code snippet"""
        # Tokenize and truncate
        inputs = self.tokenizer(
            code, 
            return_tensors="pt", 
            max_length=max_length, 
            truncation=True, 
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling of last hidden states
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        
        return embedding
    
    def embed_diff(self, added_lines: List[str], removed_lines: List[str]) -> np.ndarray:
        """Generate embedding for a code diff"""
        # Combine added and removed lines with special tokens
        diff_text = ""
        
        if removed_lines:
            diff_text += "[REMOVED] " + "\n".join(removed_lines) + " "
        
        if added_lines:
            diff_text += "[ADDED] " + "\n".join(added_lines)
        
        return self.embed_code(diff_text)

class TextEmbedder:
    """Generate embeddings for commit messages and text using sentence transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        return self.model.encode([text])[0]
    
    def embed_commit_message(self, message: str) -> np.ndarray:
        """Generate embedding for commit message"""
        # Clean and normalize commit message
        cleaned_message = self._clean_commit_message(message)
        return self.embed_text(cleaned_message)
    
    def _clean_commit_message(self, message: str) -> str:
        """Clean and normalize commit message"""
        # Remove common prefixes and normalize
        lines = message.split('\n')
        first_line = lines[0].strip()
        
        # Remove common prefixes
        prefixes = ['fix:', 'feat:', 'docs:', 'style:', 'refactor:', 'test:', 'chore:']
        for prefix in prefixes:
            if first_line.lower().startswith(prefix):
                first_line = first_line[len(prefix):].strip()
                break
        
        return first_line

class SemanticAnalyzer:
    """Analyze semantic drift and similarity between code changes"""
    
    def __init__(self):
        self.code_embedder = CodeEmbedder()
        self.text_embedder = TextEmbedder()
    
    def generate_hybrid_embedding(self, 
                                 code_diff: Tuple[List[str], List[str]], 
                                 commit_message: str,
                                 code_weight: float = 0.7,
                                 text_weight: float = 0.3) -> EmbeddingResult:
        """Generate hybrid embedding combining code and text"""
        added_lines, removed_lines = code_diff
        
        # Generate individual embeddings
        code_embedding = self.code_embedder.embed_diff(added_lines, removed_lines)
        text_embedding = self.text_embedder.embed_commit_message(commit_message)
        
        # Normalize embeddings to same dimension if needed
        if len(code_embedding) != len(text_embedding):
            # Pad shorter embedding with zeros
            max_dim = max(len(code_embedding), len(text_embedding))
            code_embedding = np.pad(code_embedding, (0, max_dim - len(code_embedding)))
            text_embedding = np.pad(text_embedding, (0, max_dim - len(text_embedding)))
        
        # Combine with weights
        hybrid_embedding = code_weight * code_embedding + text_weight * text_embedding
        
        return EmbeddingResult(
            embedding=hybrid_embedding,
            text=f"Code: {len(added_lines)} added, {len(removed_lines)} removed | Message: {commit_message[:100]}",
            embedding_type="hybrid",
            metadata={
                "added_lines_count": len(added_lines),
                "removed_lines_count": len(removed_lines),
                "commit_message": commit_message,
                "code_weight": code_weight,
                "text_weight": text_weight
            }
        )
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        return cosine_similarity([embedding1], [embedding2])[0][0]
    
    def calculate_drift_score(self, embeddings: List[np.ndarray]) -> List[float]:
        """Calculate semantic drift scores over time"""
        if len(embeddings) < 2:
            return []
        
        drift_scores = []
        base_embedding = embeddings[0]
        
        for i in range(1, len(embeddings)):
            similarity = self.calculate_similarity(base_embedding, embeddings[i])
            drift_score = 1 - similarity  # Higher drift = lower similarity
            drift_scores.append(drift_score)
        
        return drift_scores
    
    def calculate_change_magnitude(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate magnitude of change between two embeddings"""
        return np.linalg.norm(embedding1 - embedding2)
    
    def detect_significant_changes(self, 
                                 embeddings: List[EmbeddingResult], 
                                 threshold: float = 0.3) -> List[int]:
        """Detect indices of commits with significant semantic changes"""
        if len(embeddings) < 2:
            return []
        
        significant_changes = []
        
        for i in range(1, len(embeddings)):
            drift = 1 - self.calculate_similarity(
                embeddings[i-1].embedding, 
                embeddings[i].embedding
            )
            
            if drift > threshold:
                significant_changes.append(i)
        
        return significant_changes
    
    def analyze_change_patterns(self, embeddings: List[EmbeddingResult]) -> Dict[str, Any]:
        """Analyze patterns in semantic changes"""
        if len(embeddings) < 2:
            return {}
        
        # Calculate all pairwise similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = self.calculate_similarity(
                    embeddings[i].embedding, 
                    embeddings[j].embedding
                )
                similarities.append(sim)
        
        # Calculate drift progression
        drift_scores = []
        for i in range(1, len(embeddings)):
            drift = 1 - self.calculate_similarity(
                embeddings[0].embedding,  # Compare to original
                embeddings[i].embedding
            )
            drift_scores.append(drift)
        
        return {
            "average_similarity": np.mean(similarities),
            "similarity_std": np.std(similarities),
            "max_drift": max(drift_scores) if drift_scores else 0,
            "cumulative_drift": sum(drift_scores) if drift_scores else 0,
            "drift_trend": self._calculate_trend(drift_scores),
            "significant_changes": self.detect_significant_changes(embeddings),
            "total_changes": len(embeddings)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction (increasing, decreasing, stable)"""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"

class EmbeddingCache:
    """Cache embeddings to avoid recomputation"""
    
    def __init__(self, cache_file: str = "embedding_cache.json"):
        self.cache_file = cache_file
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load cache from file"""
        try:
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def _save_cache(self):
        """Save cache to file"""
        # Convert numpy arrays to lists for JSON serialization
        cache_to_save = {}
        for key, value in self.cache.items():
            if isinstance(value.get('embedding'), np.ndarray):
                cache_to_save[key] = {
                    **value,
                    'embedding': value['embedding'].tolist()
                }
            else:
                cache_to_save[key] = value
        
        with open(self.cache_file, 'w') as f:
            json.dump(cache_to_save, f)
    
    def _generate_key(self, text: str, embedding_type: str) -> str:
        """Generate cache key"""
        return hashlib.md5(f"{text}_{embedding_type}".encode()).hexdigest()
    
    def get_embedding(self, text: str, embedding_type: str) -> Optional[np.ndarray]:
        """Get embedding from cache"""
        key = self._generate_key(text, embedding_type)
        if key in self.cache:
            embedding = self.cache[key]['embedding']
            if isinstance(embedding, list):
                return np.array(embedding)
            return embedding
        return None
    
    def store_embedding(self, text: str, embedding_type: str, embedding: np.ndarray):
        """Store embedding in cache"""
        key = self._generate_key(text, embedding_type)
        self.cache[key] = {
            'embedding': embedding,
            'text': text[:100],  # Store first 100 chars for reference
            'embedding_type': embedding_type
        }
        self._save_cache()

def example_usage():
    """Example of how to use the embedding system"""
    analyzer = SemanticAnalyzer()
    
    # Example code changes
    changes = [
        {
            'added': ['def hello():', '    print("Hello, World!")'],
            'removed': [],
            'message': 'Add hello function'
        },
        {
            'added': ['def hello():', '    print("Hello, Universe!")'],
            'removed': ['def hello():', '    print("Hello, World!")'],
            'message': 'Update hello message'
        },
        {
            'added': ['def greet(name):', '    print(f"Hello, {name}!")'],
            'removed': ['def hello():', '    print("Hello, Universe!")'],
            'message': 'Refactor hello to accept name parameter'
        }
    ]
    
    # Generate embeddings
    embeddings = []
    for change in changes:
        embedding_result = analyzer.generate_hybrid_embedding(
            (change['added'], change['removed']),
            change['message']
        )
        embeddings.append(embedding_result)
    
    # Analyze patterns
    patterns = analyzer.analyze_change_patterns(embeddings)
    print("Change patterns:", patterns)
    
    # Calculate drift
    embedding_vectors = [e.embedding for e in embeddings]
    drift_scores = analyzer.calculate_drift_score(embedding_vectors)
    print("Drift scores:", drift_scores)

if __name__ == "__main__":
    example_usage()

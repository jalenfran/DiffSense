"""
Configuration management for DiffSense backend
Centralizes all environment variable handling
"""

import os
from typing import Any, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration manager for DiffSense"""
    
    def __init__(self):
        # API Keys
        self.claude_api_key = os.getenv('CLAUDE_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        # Model Configuration
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        self.code_model = os.getenv('CODE_MODEL', 'microsoft/codebert-base')
        
        # Thresholds
        self.semantic_similarity_threshold = float(os.getenv('SEMANTIC_SIMILARITY_THRESHOLD', '0.3'))
        self.breaking_change_threshold = float(os.getenv('BREAKING_CHANGE_THRESHOLD', '0.7'))
        
        # System Configuration
        self.max_commits_default = int(os.getenv('MAX_COMMITS_DEFAULT', '50'))
        self.temp_dir_prefix = os.getenv('TEMP_DIR_PREFIX', 'diffsense_')
        
        # Development
        self.debug = os.getenv('DEBUG', 'False').lower() == 'true'
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return status"""
        status = {
            "claude_available": bool(self.claude_api_key),
            "openai_available": bool(self.openai_api_key),
            "embedding_model": self.embedding_model,
            "code_model": self.code_model,
            "debug_mode": self.debug,
            "log_level": self.log_level
        }
        return status
    
    def get_model_device(self) -> str:
        """Determine the best device for model inference"""
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
        else:
            return "cpu"

# Global config instance
config = Config()
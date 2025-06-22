"""
Repository Storage Manager
Handles organized storage of cloned repositories with better file structure
"""

import os
import shutil
import hashlib
import tempfile
from typing import Dict, Optional
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)

class RepositoryStorageManager:
    """Manages organized storage of cloned repositories"""
    
    def __init__(self, base_storage_path: str = "./repos"):
        self.base_storage_path = os.path.abspath(base_storage_path)
        self.ensure_storage_structure()
    
    def ensure_storage_structure(self):
        """Create organized storage directory structure"""
        directories = [
            self.base_storage_path,
            os.path.join(self.base_storage_path, "active"),
            os.path.join(self.base_storage_path, "archived"),
            os.path.join(self.base_storage_path, "temp"),
            os.path.join(self.base_storage_path, "cache")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def generate_repo_id(self, repo_url: str) -> str:
        """Generate unique repository ID from URL"""
        # Parse URL to get owner/repo name
        parsed = urlparse(repo_url)
        path_parts = parsed.path.strip('/').split('/')
        
        if len(path_parts) >= 2:
            owner = path_parts[-2]
            repo_name = path_parts[-1].replace('.git', '')
            base_id = f"{owner}_{repo_name}"
        else:
            # Fallback to hash-based ID
            base_id = hashlib.md5(repo_url.encode()).hexdigest()[:8]
        
        # Add unique suffix if needed
        counter = 0
        repo_id = base_id
        while self.get_repository_path(repo_id) and os.path.exists(self.get_repository_path(repo_id)):
            counter += 1
            repo_id = f"{base_id}_{counter}"
        
        return repo_id
    
    def get_repository_path(self, repo_id: str, status: str = "active") -> str:
        """Get the full path for a repository"""
        return os.path.join(self.base_storage_path, status, repo_id)
    
    def get_temp_path(self) -> str:
        """Get a temporary path for cloning"""
        return tempfile.mkdtemp(dir=os.path.join(self.base_storage_path, "temp"))
    
    def move_repository(self, temp_path: str, repo_id: str, status: str = "active") -> str:
        """Move repository from temp to permanent storage"""
        target_path = self.get_repository_path(repo_id, status)
        
        # Remove existing if present
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
        
        # Move from temp to target
        shutil.move(temp_path, target_path)
        logger.info(f"Moved repository {repo_id} to {target_path}")
        
        return target_path
    
    def archive_repository(self, repo_id: str) -> bool:
        """Archive a repository (move to archived folder)"""
        try:
            active_path = self.get_repository_path(repo_id, "active")
            archived_path = self.get_repository_path(repo_id, "archived")
            
            if os.path.exists(active_path):
                if os.path.exists(archived_path):
                    shutil.rmtree(archived_path)
                shutil.move(active_path, archived_path)
                logger.info(f"Archived repository {repo_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error archiving repository {repo_id}: {e}")
            return False
    
    def delete_repository(self, repo_id: str) -> bool:
        """Completely delete a repository"""
        try:
            for status in ["active", "archived"]:
                repo_path = self.get_repository_path(repo_id, status)
                if os.path.exists(repo_path):
                    shutil.rmtree(repo_path)
                    logger.info(f"Deleted repository {repo_id} from {status}")
            return True
        except Exception as e:
            logger.error(f"Error deleting repository {repo_id}: {e}")
            return False
    
    def cleanup_temp_files(self, max_age_hours: int = 24):
        """Clean up old temporary files"""
        import time
        temp_dir = os.path.join(self.base_storage_path, "temp")
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        try:
            for item in os.listdir(temp_dir):
                item_path = os.path.join(temp_dir, item)
                if os.path.isdir(item_path):
                    creation_time = os.path.getctime(item_path)
                    if current_time - creation_time > max_age_seconds:
                        shutil.rmtree(item_path)
                        logger.info(f"Cleaned up old temp directory: {item}")
        except Exception as e:
            logger.error(f"Error during temp cleanup: {e}")
    
    def get_storage_stats(self) -> Dict[str, any]:
        """Get storage usage statistics"""
        stats = {
            "active_repos": 0,
            "archived_repos": 0,
            "total_size_mb": 0,
            "temp_files": 0
        }
        
        try:
            # Count active repos
            active_dir = os.path.join(self.base_storage_path, "active")
            if os.path.exists(active_dir):
                stats["active_repos"] = len([d for d in os.listdir(active_dir) 
                                           if os.path.isdir(os.path.join(active_dir, d))])
            
            # Count archived repos
            archived_dir = os.path.join(self.base_storage_path, "archived")
            if os.path.exists(archived_dir):
                stats["archived_repos"] = len([d for d in os.listdir(archived_dir) 
                                             if os.path.isdir(os.path.join(archived_dir, d))])
            
            # Calculate total size
            def get_dir_size(path):
                total = 0
                for dirpath, dirnames, filenames in os.walk(path):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        try:
                            total += os.path.getsize(filepath)
                        except:
                            pass
                return total
            
            total_bytes = get_dir_size(self.base_storage_path)
            stats["total_size_mb"] = round(total_bytes / (1024 * 1024), 2)
            
            # Count temp files
            temp_dir = os.path.join(self.base_storage_path, "temp")
            if os.path.exists(temp_dir):
                stats["temp_files"] = len([d for d in os.listdir(temp_dir) 
                                         if os.path.isdir(os.path.join(temp_dir, d))])
        
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
        
        return stats
    
    def list_repositories(self, status: str = "active") -> list:
        """List all repositories in a given status"""
        repo_dir = os.path.join(self.base_storage_path, status)
        if not os.path.exists(repo_dir):
            return []
        
        return [d for d in os.listdir(repo_dir) 
                if os.path.isdir(os.path.join(repo_dir, d))]

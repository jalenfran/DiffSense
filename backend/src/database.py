"""
Database models and operations for DiffSense
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class Repository:
    """Repository database model"""
    id: str
    url: str
    name: str
    local_path: str
    created_at: str
    updated_at: str
    total_commits: int = 0
    total_files: int = 0
    primary_language: str = ""
    status: str = "active"  # active, indexing, archived

@dataclass
class Commit:
    """Commit database model"""
    id: str
    repo_id: str
    hash: str
    message: str
    author: str
    timestamp: str
    files_changed: str  # JSON string
    additions: int
    deletions: int
    risk_score: float
    breaking_changes: str  # JSON string
    semantic_embedding: Optional[bytes] = None  # Pickled numpy array

@dataclass
class FileRecord:
    """File database model"""
    id: str
    repo_id: str
    path: str
    language: str
    size: int
    last_modified: str
    modification_frequency: int
    functions: str  # JSON string
    classes: str  # JSON string
    dependencies: str  # JSON string
    semantic_embedding: Optional[bytes] = None

class DatabaseManager:
    """Database manager for DiffSense"""
    
    def __init__(self, db_path: str = "./diffsense.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Repositories table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS repositories (
                    id TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    name TEXT NOT NULL,
                    local_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    total_commits INTEGER DEFAULT 0,
                    total_files INTEGER DEFAULT 0,
                    primary_language TEXT DEFAULT '',
                    status TEXT DEFAULT 'active'
                )
            """)
            
            # Commits table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS commits (
                    id TEXT PRIMARY KEY,
                    repo_id TEXT NOT NULL,
                    hash TEXT NOT NULL,
                    message TEXT NOT NULL,
                    author TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    files_changed TEXT NOT NULL,
                    additions INTEGER DEFAULT 0,
                    deletions INTEGER DEFAULT 0,
                    risk_score REAL DEFAULT 0.0,
                    breaking_changes TEXT DEFAULT '[]',
                    semantic_embedding BLOB,
                    FOREIGN KEY (repo_id) REFERENCES repositories (id),
                    UNIQUE (repo_id, hash)
                )
            """)
            
            # Files table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    id TEXT PRIMARY KEY,
                    repo_id TEXT NOT NULL,
                    path TEXT NOT NULL,
                    language TEXT NOT NULL,
                    size INTEGER DEFAULT 0,
                    last_modified TEXT NOT NULL,
                    modification_frequency INTEGER DEFAULT 0,
                    functions TEXT DEFAULT '[]',
                    classes TEXT DEFAULT '[]',
                    dependencies TEXT DEFAULT '[]',
                    semantic_embedding BLOB,
                    FOREIGN KEY (repo_id) REFERENCES repositories (id),
                    UNIQUE (repo_id, path)
                )
            """)
            
            # Indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_commits_repo_id ON commits (repo_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_commits_hash ON commits (hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_files_repo_id ON files (repo_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_files_path ON files (path)")
            
            conn.commit()
    
    def save_repository(self, repository: Repository) -> bool:
        """Save repository to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO repositories 
                    (id, url, name, local_path, created_at, updated_at, total_commits, 
                     total_files, primary_language, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    repository.id, repository.url, repository.name, repository.local_path,
                    repository.created_at, repository.updated_at, repository.total_commits,
                    repository.total_files, repository.primary_language, repository.status
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving repository: {e}")
            return False
    
    def get_repository(self, repo_id: str) -> Optional[Repository]:
        """Get repository from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM repositories WHERE id = ?", (repo_id,))
                row = cursor.fetchone()
                if row:
                    return Repository(*row)
                return None
        except Exception as e:
            logger.error(f"Error getting repository: {e}")
            return None
    
    def get_repository_by_url(self, url: str) -> Optional[Repository]:
        """Get repository from database by URL"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM repositories WHERE url = ?", (url,))
                row = cursor.fetchone()
                if row:
                    return Repository(*row)
                return None
        except Exception as e:
            logger.error(f"Error getting repository by URL: {e}")
            return None
    
    def list_repositories(self) -> List[Repository]:
        """List all repositories"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM repositories ORDER BY updated_at DESC")
                rows = cursor.fetchall()
                return [Repository(*row) for row in rows]
        except Exception as e:
            logger.error(f"Error listing repositories: {e}")
            return []
    
    def delete_repository(self, repo_id: str) -> bool:
        """Delete repository and all related data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM files WHERE repo_id = ?", (repo_id,))
                cursor.execute("DELETE FROM commits WHERE repo_id = ?", (repo_id,))
                cursor.execute("DELETE FROM repositories WHERE id = ?", (repo_id,))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error deleting repository: {e}")
            return False
    
    def save_commits(self, commits: List[Commit]) -> bool:
        """Save multiple commits to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                for commit in commits:
                    cursor.execute("""
                        INSERT OR REPLACE INTO commits 
                        (id, repo_id, hash, message, author, timestamp, files_changed,
                         additions, deletions, risk_score, breaking_changes, semantic_embedding)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        commit.id, commit.repo_id, commit.hash, commit.message,
                        commit.author, commit.timestamp, commit.files_changed,
                        commit.additions, commit.deletions, commit.risk_score,
                        commit.breaking_changes, commit.semantic_embedding
                    ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving commits: {e}")
            return False
    
    def get_commits(self, repo_id: str, limit: int = 100) -> List[Commit]:
        """Get commits for repository"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM commits WHERE repo_id = ? 
                    ORDER BY timestamp DESC LIMIT ?
                """, (repo_id, limit))
                rows = cursor.fetchall()
                return [Commit(*row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting commits: {e}")
            return []
    
    def save_files(self, files: List[FileRecord]) -> bool:
        """Save multiple files to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                for file_record in files:
                    cursor.execute("""
                        INSERT OR REPLACE INTO files 
                        (id, repo_id, path, language, size, last_modified,
                         modification_frequency, functions, classes, dependencies, semantic_embedding)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        file_record.id, file_record.repo_id, file_record.path,
                        file_record.language, file_record.size, file_record.last_modified,
                        file_record.modification_frequency, file_record.functions,
                        file_record.classes, file_record.dependencies, file_record.semantic_embedding
                    ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving files: {e}")
            return False
    
    def get_files(self, repo_id: str) -> List[FileRecord]:
        """Get files for repository"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM files WHERE repo_id = ?", (repo_id,))
                rows = cursor.fetchall()
                return [FileRecord(*row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting files: {e}")
            return []
    
    def search_commits(self, repo_id: str, query: str, limit: int = 10) -> List[Commit]:
        """Search commits by message or author"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM commits WHERE repo_id = ? 
                    AND (message LIKE ? OR author LIKE ?)
                    ORDER BY timestamp DESC LIMIT ?
                """, (repo_id, f"%{query}%", f"%{query}%", limit))
                rows = cursor.fetchall()
                return [Commit(*row) for row in rows]
        except Exception as e:
            logger.error(f"Error searching commits: {e}")
            return []
    
    def search_files(self, repo_id: str, query: str, limit: int = 10) -> List[FileRecord]:
        """Search files by path or content"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM files WHERE repo_id = ? 
                    AND (path LIKE ? OR functions LIKE ? OR classes LIKE ?)
                    LIMIT ?
                """, (repo_id, f"%{query}%", f"%{query}%", f"%{query}%", limit))
                rows = cursor.fetchall()
                return [FileRecord(*row) for row in rows]
        except Exception as e:
            logger.error(f"Error searching files: {e}")
            return []
    
    def add_repository(self, url: str, name: str, storage_path: str, last_analyzed: datetime) -> bool:
        """Add a new repository to the database"""
        try:
            # Generate repository ID from URL
            import hashlib
            repo_id = hashlib.sha256(url.encode()).hexdigest()[:16]
            
            repository = Repository(
                id=repo_id,
                url=url,
                name=name,
                local_path=storage_path,
                created_at=datetime.now().isoformat(),
                updated_at=last_analyzed.isoformat(),
                total_commits=0,
                total_files=0,
                primary_language="",
                status="active"
            )
            
            return self.save_repository(repository)
        except Exception as e:
            logger.error(f"Error adding repository: {e}")
            return False

    def get_commit(self, repo_id: str, commit_hash: str) -> Optional[Commit]:
        """Get a specific commit from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM commits WHERE repo_id = ? AND hash = ?", 
                    (repo_id, commit_hash)
                )
                row = cursor.fetchone()
                if row:
                    return Commit(*row)
                return None
        except Exception as e:
            logger.error(f"Error getting commit: {e}")
            return None

    def add_commit(self, repository_id: str, commit_hash: str, message: str, author: str, 
                   timestamp: datetime, files_changed: int) -> bool:
        """Add a single commit to the database"""
        try:
            # Generate commit ID
            import hashlib
            commit_id = hashlib.sha256(f"{repository_id}_{commit_hash}".encode()).hexdigest()[:16]
            
            commit = Commit(
                id=commit_id,
                repo_id=repository_id,
                hash=commit_hash,
                message=message,
                author=author,
                timestamp=timestamp.isoformat(),
                files_changed=str(files_changed),
                additions=0,
                deletions=0,
                risk_score=0.0,
                breaking_changes="[]",
                semantic_embedding=None
            )
            
            return self.save_commits([commit])
        except Exception as e:
            logger.error(f"Error adding commit: {e}")
            return False

    def add_file_record(self, commit_id: str, file_path: str, lines_added: int, lines_deleted: int) -> bool:
        """Add a file record for a commit"""
        try:
            # Generate file record ID
            import hashlib
            file_id = hashlib.sha256(f"{commit_id}_{file_path}".encode()).hexdigest()[:16]
            
            # Get repository ID from commit
            commit = None
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT repo_id FROM commits WHERE id = ?", (commit_id,))
                row = cursor.fetchone()
                if row:
                    repo_id = row[0]
                else:
                    logger.error(f"Commit {commit_id} not found")
                    return False
            
            file_record = FileRecord(
                id=file_id,
                repo_id=repo_id,
                path=file_path,
                language="",
                size=0,
                last_modified=datetime.now().isoformat(),
                modification_frequency=1,
                functions="[]",
                classes="[]",
                dependencies="[]",
                semantic_embedding=None
            )
            
            return self.save_files([file_record])
        except Exception as e:
            logger.error(f"Error adding file record: {e}")
            return False

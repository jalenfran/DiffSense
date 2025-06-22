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
    user_id: str  # GitHub user ID who owns this repository
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

@dataclass
class User:
    """User database model"""
    id: str  # GitHub user ID
    github_username: str
    github_avatar: str
    github_email: Optional[str]
    display_name: str
    access_token: str  # GitHub access token (encrypted)
    created_at: str
    last_login: str
    is_active: bool = True

@dataclass
class Chat:
    """Chat conversation database model"""
    id: str
    user_id: str
    repo_id: Optional[str]  # Can be None for general chats
    title: str
    created_at: str
    updated_at: str
    is_archived: bool = False
    message_count: int = 0

@dataclass
class ChatMessage:
    """Chat message database model"""
    id: str
    chat_id: str
    user_id: str
    message: str
    response: str
    query_type: str  # file, commit, author, security, diff, general
    context_used: str  # JSON string of sources/context
    confidence: float
    timestamp: str
    claude_enhanced: bool = False

@dataclass
class CommitAnalysis:
    """Commit analysis results database model"""
    id: str
    repo_id: str
    commit_hash: str
    overall_risk_score: float
    breaking_changes_count: int
    complexity_score: float
    semantic_drift_score: float
    files_changed: str  # JSON string
    breaking_changes: str  # JSON string
    analyzed_at: str
    analysis_version: str = "1.0"  # For future compatibility

@dataclass
class RepositoryDashboard:
    """Repository dashboard cache model"""
    id: str
    repo_id: str
    dashboard_data: str  # JSON string of complete dashboard
    last_updated: str
    commits_analyzed: int
    cache_version: str = "1.0"

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
                    user_id TEXT NOT NULL,
                    total_commits INTEGER DEFAULT 0,
                    total_files INTEGER DEFAULT 0,
                    primary_language TEXT DEFAULT '',
                    status TEXT DEFAULT 'active',
                    FOREIGN KEY (user_id) REFERENCES users (id)
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
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    github_username TEXT NOT NULL,
                    github_avatar TEXT NOT NULL,
                    github_email TEXT,
                    display_name TEXT NOT NULL,
                    access_token TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_login TEXT NOT NULL,
                    is_active INTEGER DEFAULT 1
                )
            """)
            
            # Chats table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chats (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    repo_id TEXT,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    is_archived INTEGER DEFAULT 0,
                    message_count INTEGER DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users (id),
                    FOREIGN KEY (repo_id) REFERENCES repositories (id)
                )
            """)
            
            # Chat messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    message TEXT NOT NULL,
                    response TEXT NOT NULL,
                    query_type TEXT NOT NULL,
                    context_used TEXT NOT NULL,
                    confidence REAL DEFAULT 0.0,
                    timestamp TEXT NOT NULL,
                    claude_enhanced INTEGER DEFAULT 0,
                    FOREIGN KEY (chat_id) REFERENCES chats (id),
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            # Commit analysis table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS commit_analysis (
                    id TEXT PRIMARY KEY,
                    repo_id TEXT NOT NULL,
                    commit_hash TEXT NOT NULL,
                    overall_risk_score REAL NOT NULL,
                    breaking_changes_count INTEGER DEFAULT 0,
                    complexity_score REAL DEFAULT 0.0,
                    semantic_drift_score REAL DEFAULT 0.0,
                    files_changed TEXT NOT NULL,
                    breaking_changes TEXT NOT NULL,
                    analyzed_at TEXT NOT NULL,
                    analysis_version TEXT DEFAULT '1.0',
                    FOREIGN KEY (repo_id) REFERENCES repositories (id),
                    UNIQUE (repo_id, commit_hash)
                )
            """)
            
            # Repository dashboard cache table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS repository_dashboards (
                    id TEXT PRIMARY KEY,
                    repo_id TEXT NOT NULL,
                    dashboard_data TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    commits_analyzed INTEGER DEFAULT 0,
                    cache_version TEXT DEFAULT '1.0',
                    FOREIGN KEY (repo_id) REFERENCES repositories (id),
                    UNIQUE (repo_id)
                )
            """)
            
            # Indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_commits_repo_id ON commits (repo_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_commits_hash ON commits (hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_files_repo_id ON files (repo_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_files_path ON files (path)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_github_username ON users (github_username)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chats_user_id ON chats (user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_chat_id ON chat_messages (chat_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_commit_analysis_repo_id ON commit_analysis (repo_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_commit_analysis_hash ON commit_analysis (commit_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_repository_dashboards_repo_id ON repository_dashboards (repo_id)")
            
            conn.commit()
    
    def save_repository(self, repository: Repository) -> bool:
        """Save repository to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO repositories 
                    (id, url, name, local_path, created_at, updated_at, user_id, total_commits, 
                     total_files, primary_language, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    repository.id, repository.url, repository.name, repository.local_path,
                    repository.created_at, repository.updated_at, repository.user_id, repository.total_commits,
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
    
    def add_repository(self, url: str, name: str, storage_path: str, user_id: str, last_analyzed: datetime) -> bool:
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
                user_id=user_id,
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

    # ===== USER MANAGEMENT =====
    
    def save_user(self, user: User) -> bool:
        """Save or update user in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO users 
                    (id, github_username, github_avatar, github_email, display_name, 
                     access_token, created_at, last_login, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user.id, user.github_username, user.github_avatar, user.github_email,
                    user.display_name, user.access_token, user.created_at, user.last_login,
                    1 if user.is_active else 0
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving user: {e}")
            return False
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
                row = cursor.fetchone()
                if row:
                    # Convert SQLite boolean back to Python boolean
                    row = list(row)
                    row[-1] = bool(row[-1])  # is_active field
                    return User(*row)
                return None
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            return None
    
    def get_user_by_github_username(self, username: str) -> Optional[User]:
        """Get user by GitHub username"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM users WHERE github_username = ?", (username,))
                row = cursor.fetchone()
                if row:
                    row = list(row)
                    row[-1] = bool(row[-1])  # is_active field
                    return User(*row)
                return None
        except Exception as e:
            logger.error(f"Error getting user by username: {e}")
            return None
    
    def update_user_last_login(self, user_id: str) -> bool:
        """Update user's last login timestamp"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE users SET last_login = ? WHERE id = ?
                """, (datetime.now().isoformat(), user_id))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error updating user last login: {e}")
            return False
    
    # ===== CHAT MANAGEMENT =====
    
    def create_chat(self, user_id: str, repo_id: Optional[str] = None, title: str = "New Chat") -> Optional[str]:
        """Create a new chat conversation"""
        try:
            import uuid
            chat_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO chats (id, user_id, repo_id, title, created_at, updated_at, is_archived, message_count)
                    VALUES (?, ?, ?, ?, ?, ?, 0, 0)
                """, (chat_id, user_id, repo_id, title, now, now))
                conn.commit()
                return chat_id
        except Exception as e:
            logger.error(f"Error creating chat: {e}")
            return None
    
    def get_user_chats(self, user_id: str, include_archived: bool = False) -> List[Chat]:
        """Get all chats for a user"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if include_archived:
                    cursor.execute("""
                        SELECT * FROM chats WHERE user_id = ? 
                        ORDER BY updated_at DESC
                    """, (user_id,))
                else:
                    cursor.execute("""
                        SELECT * FROM chats WHERE user_id = ? AND is_archived = 0 
                        ORDER BY updated_at DESC
                    """, (user_id,))
                
                rows = cursor.fetchall()
                chats = []
                for row in rows:
                    row = list(row)
                    row[-2] = bool(row[-2])  # is_archived field
                    chats.append(Chat(*row))
                return chats
        except Exception as e:
            logger.error(f"Error getting user chats: {e}")
            return []
    
    def get_chat(self, chat_id: str) -> Optional[Chat]:
        """Get a specific chat"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM chats WHERE id = ?", (chat_id,))
                row = cursor.fetchone()
                if row:
                    row = list(row)
                    row[-2] = bool(row[-2])  # is_archived field
                    return Chat(*row)
                return None
        except Exception as e:
            logger.error(f"Error getting chat: {e}")
            return None
    
    def update_chat_title(self, chat_id: str, title: str) -> bool:
        """Update chat title"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE chats SET title = ?, updated_at = ? WHERE id = ?
                """, (title, datetime.now().isoformat(), chat_id))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error updating chat title: {e}")
            return False
    
    def archive_chat(self, chat_id: str) -> bool:
        """Archive a chat"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE chats SET is_archived = 1, updated_at = ? WHERE id = ?
                """, (datetime.now().isoformat(), chat_id))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error archiving chat: {e}")
            return False
    
    def delete_chat(self, chat_id: str) -> bool:
        """Delete a chat and all its messages"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM chat_messages WHERE chat_id = ?", (chat_id,))
                cursor.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error deleting chat: {e}")
            return False
    
    def save_chat(self, chat: Chat) -> bool:
        """Save a chat to the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO chats 
                    (id, user_id, repo_id, title, created_at, updated_at, is_archived, message_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                """, (
                    chat.id, chat.user_id, chat.repo_id, chat.title,
                    chat.created_at, chat.updated_at, 1 if chat.is_archived else 0
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving chat: {e}")
            return False
    
    def update_chat_timestamp(self, chat_id: str) -> bool:
        """Update chat's updated_at timestamp"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE chats SET updated_at = ? WHERE id = ?
                """, (datetime.now().isoformat(), chat_id))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error updating chat timestamp: {e}")
            return False
    
    # ===== CHAT MESSAGE MANAGEMENT =====
    
    def save_chat_message(self, message: ChatMessage) -> bool:
        """Save a chat message"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO chat_messages 
                    (id, chat_id, user_id, message, response, query_type, context_used, 
                     confidence, timestamp, claude_enhanced)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    message.id, message.chat_id, message.user_id, message.message,
                    message.response, message.query_type, message.context_used,
                    message.confidence, message.timestamp, 1 if message.claude_enhanced else 0
                ))
                
                # Update chat message count and updated_at
                cursor.execute("""
                    UPDATE chats SET message_count = message_count + 1, updated_at = ? 
                    WHERE id = ?
                """, (message.timestamp, message.chat_id))
                
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving chat message: {e}")
            return False
    
    def get_chat_messages(self, chat_id: str, limit: int = 50) -> List[ChatMessage]:
        """Get messages for a chat"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM chat_messages WHERE chat_id = ? 
                    ORDER BY timestamp DESC LIMIT ?
                """, (chat_id, limit))
                
                rows = cursor.fetchall()
                messages = []
                for row in rows:
                    row = list(row)
                    row[-1] = bool(row[-1])  # claude_enhanced field
                    messages.append(ChatMessage(*row))
                return list(reversed(messages))  # Return in chronological order
        except Exception as e:
            logger.error(f"Error getting chat messages: {e}")
            return []
    
    # ===== USER-REPOSITORY ASSOCIATION =====
    
    def link_repository_to_user(self, repo_id: str, user_id: str) -> bool:
        """Link a repository to a user (add user_id column if needed)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if user_id column exists in repositories table
                cursor.execute("PRAGMA table_info(repositories)")
                columns = [column[1] for column in cursor.fetchall()]
                
                if 'user_id' not in columns:
                    # Add user_id column to repositories table
                    cursor.execute("ALTER TABLE repositories ADD COLUMN user_id TEXT")
                
                # Update repository with user_id
                cursor.execute("""
                    UPDATE repositories SET user_id = ? WHERE id = ?
                """, (user_id, repo_id))
                
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error linking repository to user: {e}")
            return False
    
    def get_user_repositories(self, user_id: str) -> List[Repository]:
        """Get all repositories for a user"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if user_id column exists
                cursor.execute("PRAGMA table_info(repositories)")
                columns = [column[1] for column in cursor.fetchall()]
                
                if 'user_id' in columns:
                    cursor.execute("""
                        SELECT * FROM repositories WHERE user_id = ? 
                        ORDER BY updated_at DESC
                    """, (user_id,))
                else:
                    # Fallback to all repositories if no user association exists yet
                    cursor.execute("SELECT * FROM repositories ORDER BY updated_at DESC")
                
                rows = cursor.fetchall()
                repositories = []
                for row in rows:
                    # Handle both old format (10 fields) and new format (11 fields with user_id)
                    if len(row) == 11:
                        # New format with user_id - all fields present
                        repositories.append(Repository(*row))
                    else:
                        # Old format without user_id - add user_id as None for backwards compatibility
                        repo_data = list(row)
                        repo_data.insert(6, user_id)  # Insert user_id after updated_at (index 5)
                        repositories.append(Repository(*repo_data))
                return repositories
        except Exception as e:
            logger.error(f"Error getting user repositories: {e}")
            return []
    
    def user_has_repository(self, user_id: str, repo_url: str) -> bool:
        """Check if a user already has a repository with the given URL"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if user_id column exists
                cursor.execute("PRAGMA table_info(repositories)")
                columns = [column[1] for column in cursor.fetchall()]
                
                if 'user_id' in columns:
                    cursor.execute("""
                        SELECT COUNT(*) FROM repositories 
                        WHERE user_id = ? AND url = ?
                    """, (user_id, repo_url))
                    count = cursor.fetchone()[0]
                    return count > 0
                else:
                    # Fallback to check by URL only if no user association exists
                    cursor.execute("SELECT COUNT(*) FROM repositories WHERE url = ?", (repo_url,))
                    count = cursor.fetchone()[0]
                    return count > 0
        except Exception as e:
            logger.error(f"Error checking if user has repository: {e}")
            return False
    
    def get_user_repository_by_url(self, user_id: str, repo_url: str) -> Optional[Repository]:
        """Get a specific repository for a user by URL"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if user_id column exists
                cursor.execute("PRAGMA table_info(repositories)")
                columns = [column[1] for column in cursor.fetchall()]
                
                if 'user_id' in columns:
                    cursor.execute("""
                        SELECT * FROM repositories 
                        WHERE user_id = ? AND url = ?
                    """, (user_id, repo_url))
                else:
                    # Fallback to check by URL only if no user association exists
                    cursor.execute("SELECT * FROM repositories WHERE url = ?", (repo_url,))
                
                row = cursor.fetchone()
                if row:
                    if len(row) == 11:
                        # New format with user_id
                        return Repository(*row)
                    else:
                        # Old format without user_id - add user_id
                        repo_data = list(row)
                        repo_data.insert(6, user_id)
                        return Repository(*repo_data)
                return None
        except Exception as e:
            logger.error(f"Error getting user repository by URL: {e}")
            return None
    
    # ===== COMMIT ANALYSIS METHODS =====
    
    def save_commit_analysis(self, analysis: CommitAnalysis) -> bool:
        """Save commit analysis to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO commit_analysis 
                    (id, repo_id, commit_hash, overall_risk_score, breaking_changes_count,
                     complexity_score, semantic_drift_score, files_changed, breaking_changes,
                     analyzed_at, analysis_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    analysis.id, analysis.repo_id, analysis.commit_hash,
                    analysis.overall_risk_score, analysis.breaking_changes_count,
                    analysis.complexity_score, analysis.semantic_drift_score,
                    analysis.files_changed, analysis.breaking_changes,
                    analysis.analyzed_at, analysis.analysis_version
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving commit analysis: {e}")
            return False
    
    def get_commit_analysis(self, repo_id: str, commit_hash: str) -> Optional[CommitAnalysis]:
        """Get commit analysis from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM commit_analysis 
                    WHERE repo_id = ? AND commit_hash = ?
                """, (repo_id, commit_hash))
                row = cursor.fetchone()
                if row:
                    return CommitAnalysis(*row)
                return None
        except Exception as e:
            logger.error(f"Error getting commit analysis: {e}")
            return None
    
    def get_repository_commit_analyses(self, repo_id: str, limit: int = 50) -> List[CommitAnalysis]:
        """Get all commit analyses for a repository"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM commit_analysis 
                    WHERE repo_id = ?
                    ORDER BY analyzed_at DESC
                    LIMIT ?
                """, (repo_id, limit))
                rows = cursor.fetchall()
                return [CommitAnalysis(*row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting repository commit analyses: {e}")
            return []
    
    def count_analyzed_commits(self, repo_id: str) -> int:
        """Count analyzed commits for a repository"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*) FROM commit_analysis WHERE repo_id = ?
                """, (repo_id,))
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Error counting analyzed commits: {e}")
            return 0
    
    # ===== DASHBOARD CACHE METHODS =====
    
    def save_dashboard_cache(self, dashboard: RepositoryDashboard) -> bool:
        """Save dashboard cache to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO repository_dashboards 
                    (id, repo_id, dashboard_data, last_updated, commits_analyzed, cache_version)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    dashboard.id, dashboard.repo_id, dashboard.dashboard_data,
                    dashboard.last_updated, dashboard.commits_analyzed, dashboard.cache_version
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving dashboard cache: {e}")
            return False
    
    def get_dashboard_cache(self, repo_id: str) -> Optional[RepositoryDashboard]:
        """Get dashboard cache from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM repository_dashboards WHERE repo_id = ?
                """, (repo_id,))
                row = cursor.fetchone()
                if row:
                    return RepositoryDashboard(*row)
                return None
        except Exception as e:
            logger.error(f"Error getting dashboard cache: {e}")
            return None
    
    def is_dashboard_cache_valid(self, repo_id: str, max_age_hours: int = 24) -> bool:
        """Check if dashboard cache is still valid"""
        try:
            dashboard = self.get_dashboard_cache(repo_id)
            if not dashboard:
                return False
            
            from datetime import datetime, timedelta
            last_updated = datetime.fromisoformat(dashboard.last_updated)
            max_age = timedelta(hours=max_age_hours)
            
            return datetime.now() - last_updated < max_age
        except Exception as e:
            logger.error(f"Error checking dashboard cache validity: {e}")
            return False

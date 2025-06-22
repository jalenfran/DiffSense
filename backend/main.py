from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any, Optional, Union
import tempfile
import shutil
import os
import sqlite3
import subprocess
import secrets
from datetime import datetime, timedelta
import json
import traceback
import logging
import numpy as np
import uuid
from dotenv import load_dotenv
from dataclasses import asdict

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def _parse_context_used(context_used_str: str) -> List[Dict[str, Any]]:
    """Parse context_used string into list of dictionaries for ChatMessageResponse"""
    if not context_used_str:
        return []
    
    try:
        # Try to parse as JSON first
        parsed = json.loads(context_used_str)
        if isinstance(parsed, list):
            # Convert list items to dictionaries if they're strings
            result = []
            for item in parsed:
                if isinstance(item, str):
                    result.append({"content": item})
                elif isinstance(item, dict):
                    result.append(item)
                else:
                    result.append({"content": str(item)})
            return result
        else:
            return [{"content": str(parsed)}]
    except json.JSONDecodeError:
        # If not valid JSON, treat as plain string
        return [{"content": context_used_str}]

from src.config import config
from src.git_analyzer import GitAnalyzer
# Legacy imports removed - using enhanced systems only
from src.claude_analyzer import ClaudeAnalyzer, SmartContext
from src.embedding_engine import SemanticAnalyzer, TextEmbedder, CodeEmbedder
from src.database import DatabaseManager, User, Chat, ChatMessage, Commit, Repository, CommitAnalysis, RepositoryDashboard, FileContent, FileHistory, CommitFileCache
from src.storage_manager import RepositoryStorageManager
from src.github_service import GitHubService
# Enhanced intelligent systems
from src.code_analyzer import CodeAnalyzer
from src.rag_system import RAGSystem
from src.claude_analyzer import ClaudeAnalyzer
from src.suggestions_engine import SuggestionsEngine

# Advanced Breaking Change Detection imports
from src.breaking_change_detector import (
    BreakingChangeDetector, 
    BreakingChange, 
    ChangeType, 
    ImpactSeverity,
    ChangeIntent
)

app = FastAPI(title="DiffSense API", description="Semantic drift detection for git repositories")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Core request models
class AnalyzeCommitRequest(BaseModel):
    commit_hash: str
    include_claude_analysis: bool = True

class AnalyzeCommitRangeRequest(BaseModel):
    start_commit: str
    end_commit: str
    max_commits: int = 100
    include_claude_analysis: bool = False

class QueryRepositoryRequest(BaseModel):
    query: str
    max_results: int = 10

class QueryRequest(BaseModel):
    query: str
    max_results: int = 10

# Removed legacy models - no longer needed

class CommitAnalysisRequest(BaseModel):
    commit_hash: str

# Authentication models
class GitHubOAuthRequest(BaseModel):
    code: str
    state: str

class UserResponse(BaseModel):
    id: str
    github_username: str
    github_avatar: str
    github_email: Optional[str]
    display_name: str
    created_at: str
    last_login: str

class GitHubRepositoryResponse(BaseModel):
    id: str
    name: str
    full_name: str
    clone_url: str
    private: bool
    description: Optional[str]
    language: Optional[str]
    default_branch: str

# Chat models
class CreateChatRequest(BaseModel):
    repo_id: Optional[str] = None
    title: str = "New Chat"

class ChatResponse(BaseModel):
    id: str
    user_id: str
    repo_id: Optional[str]
    title: str
    created_at: str
    updated_at: str
    is_archived: bool
    message_count: int

class ChatMessageRequest(BaseModel):
    chat_id: str
    message: str

class ChatMessageResponse(BaseModel):
    id: str
    chat_id: str
    user_id: str
    message: str
    response: str
    query_type: str
    context_used: List[Dict[str, Any]]
    confidence: float
    timestamp: str
    claude_enhanced: bool

class UpdateChatTitleRequest(BaseModel):
    title: str

class CloneRepositoryRequest(BaseModel):
    repo_url: HttpUrl

# Response models
class BreakingChangeResponse(BaseModel):
    change_type: str
    risk_level: str
    confidence: float
    file_path: str
    line_number: Optional[int]
    description: str
    before_code: Optional[str]
    after_code: Optional[str]
    impact_analysis: Dict[str, Any]
    mitigation_suggestions: List[str]
    related_files: List[str]

class CommitAnalysisResponse(BaseModel):
    commit_hash: str
    commit_message: str
    timestamp: str
    author: str
    overall_risk_score: float
    breaking_changes: List[BreakingChangeResponse]
    files_changed: List[str]
    lines_added: int
    lines_removed: int
    complexity_score: float
    semantic_drift_score: float
    claude_analysis: Optional[Dict[str, Any]] = None

class RepositoryQueryResponse(BaseModel):
    query: str
    response: str
    confidence: float
    sources: List[Dict[str, Any]]
    context_used: List[str]
    suggestions: List[str]
    claude_enhanced: bool = False

class DriftEventResponse(BaseModel):
    commit_hash: str
    timestamp: str
    drift_score: float
    change_magnitude: float
    file_path: str
    description: str
    commit_message: str
    added_lines: int
    removed_lines: int

class FeatureHistoryResponse(BaseModel):
    feature_id: str
    file_path: str
    function_name: Optional[str]
    overall_drift: float
    total_commits: int
    drift_events: List[DriftEventResponse]
    change_summary: Dict[str, Any]
    timeline: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]

class RepositoryStatsResponse(BaseModel):
    total_commits: int
    contributors: int
    date_range: Dict[str, Optional[str]]
    active_branches: int
    file_count: int

# Global storage for active repositories (in production, use Redis or database)
from src.database import DatabaseManager
from src.storage_manager import RepositoryStorageManager

# Initialize AI systems
semantic_analyzer = SemanticAnalyzer()
claude_analyzer = ClaudeAnalyzer()
rag_system = RAGSystem(git_analyzer=None, claude_analyzer=claude_analyzer)
breaking_change_detector = BreakingChangeDetector(git_analyzer=None, claude_analyzer=claude_analyzer)
github_service = GitHubService()

# Initialize database and storage
db_manager = DatabaseManager()
storage_manager = RepositoryStorageManager()

# Legacy in-memory storage for backwards compatibility
active_repos: Dict[str, GitAnalyzer] = {}
temp_dirs: Dict[str, str] = {}

# Session management
user_sessions: Dict[str, str] = {}  # session_id -> user_id mapping

# Authentication utilities
def generate_session_id() -> str:
    """Generate a secure session ID"""
    return secrets.token_urlsafe(32)

async def get_current_user(authorization: str = Header(None)) -> Optional[str]:
    """Get current user from authorization header"""
    if not authorization:
        return None
    
    if authorization.startswith("Bearer "):
        session_id = authorization[7:]  # Remove "Bearer " prefix
        return user_sessions.get(session_id)
    
    return None

async def require_auth(current_user: Optional[str] = Depends(get_current_user)) -> str:
    """Require authentication for endpoints"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return current_user

def check_repository_access(repo_id: str, current_user: str) -> bool:
    """Check if the current user has access to the specified repository"""
    try:
        # Get the repository from database
        repository = db_manager.get_repository(repo_id)
        if not repository:
            return False
        
        # Check if user owns this repository
        if repository.user_id == current_user:
            return True
        
        # For now, we'll use a simple ownership model
        # In the future, this could be extended to support team access, public repos, etc.
        return False
        
    except Exception as e:
        logger.error(f"Error checking repository access: {e}")
        return False

async def require_repository_access(repo_id: str, current_user: str = Depends(require_auth)) -> str:
    """Require authentication and repository access for endpoints"""
    if not check_repository_access(repo_id, current_user):
        raise HTTPException(
            status_code=403, 
            detail="Access denied. You don't have permission to access this repository."
        )
    return current_user

# Startup event handler to load existing repositories
@app.on_event("startup")
async def startup_event():
    """Load existing repositories from database into active_repos on startup"""
    try:
        logger.info("Loading existing repositories from database...")
        
        # Get all repositories from database
        repositories = db_manager.list_repositories()
        
        for repo in repositories:
            try:
                # Check if the repository path still exists
                if os.path.exists(repo.local_path):
                    # Initialize GitAnalyzer for existing repository
                    git_analyzer = GitAnalyzer(repo.local_path)
                    
                    # Add to active repos
                    active_repos[repo.id] = git_analyzer
                    
                    logger.info(f"Loaded repository: {repo.name} ({repo.id})")
                else:
                    logger.warning(f"Repository path does not exist, skipping: {repo.local_path}")
                    
            except Exception as e:
                logger.error(f"Failed to load repository {repo.name} ({repo.id}): {e}")
                continue
        
        logger.info(f"Successfully loaded {len(active_repos)} repositories into active_repos")
        
    except Exception as e:
        logger.error(f"Error during startup repository loading: {e}")
        # Don't fail startup if repository loading fails
        pass

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "DiffSense API is running", 
        "timestamp": datetime.now().isoformat(),
        "config_status": config.validate_config()
    }

@app.post("/api/clone-repository")
async def clone_repository(
    request: CloneRepositoryRequest, 
    background_tasks: BackgroundTasks,
    current_user: str = Depends(require_auth)
):
    """Clone a repository and return basic stats"""
    try:
        print(f"Received request: {request}")
        repo_url_str = str(request.repo_url)
        print(f"Repository URL: {repo_url_str}")
        
        # Normalize URL for comparison
        normalized_url = normalize_repository_url(repo_url_str)
        
        # Check if user already has this repository
        user_repositories = db_manager.get_user_repositories(current_user)
        for repo in user_repositories:
            if normalize_repository_url(repo.url) == normalized_url:
                raise HTTPException(
                    status_code=409, 
                    detail=f"Repository already exists for this user: {repo.name} (ID: {repo.id})"
                )
        
        # Require authenticated user
        authenticated_url = repo_url_str
        user = db_manager.get_user(current_user)
        
        # If user is authenticated, try to use OAuth token for private repos
        if user and github_service.available:
            try:
                access_token = github_service.decrypt_token(user.access_token)
                
                if github_service.validate_repository_access(access_token, repo_url_str):
                    authenticated_url = github_service.create_authenticated_clone_url(access_token, repo_url_str)
                    print("Using authenticated URL for private repository")
                else:
                    print("Repository access validation failed, trying public access")
            except Exception as e:
                print(f"Authentication failed, trying public access: {e}")
        
        # Generate repository ID and setup storage
        repo_id = storage_manager.generate_repo_id(repo_url_str)
        repo_path = storage_manager.get_repository_path(repo_id, "active")
        
        # Check if repository already exists in database
        existing_repo = db_manager.get_repository_by_url(repo_url_str)
        if existing_repo and os.path.exists(repo_path):
            print(f"Repository already exists: {repo_id}")
            
            # Link to user if authenticated
            if user:
                db_manager.link_repository_to_user(repo_id, user.id)
            
            # Load existing repository
            git_analyzer = GitAnalyzer(repo_path)
            active_repos[repo_id] = git_analyzer
            
            # Get repository stats
            stats = git_analyzer.analyze_repository_stats()
            print(f"Got repository stats for existing repo: {stats}")
            
            return {
                "repo_id": repo_id,
                "status": "loaded_existing",
                "stats": RepositoryStatsResponse(**stats),
                "message": f"Repository already exists and loaded successfully"
            }
        
        # Clone repository to organized storage
        print(f"Cloning to: {repo_path}")
        git_analyzer = GitAnalyzer.clone_repository(authenticated_url, repo_path)
        print(f"Successfully created GitAnalyzer")
        
        # Link to user if authenticated
        if user:
            db_manager.link_repository_to_user(repo_id, user.id)
        
        # Store repository in database
        if not existing_repo:
            # Save repository metadata
            from src.database import Repository
            repository = Repository(
                id=repo_id,
                url=repo_url_str,
                name=repo_id,
                local_path=repo_path,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                user_id=current_user,  # Link repository to the authenticated user
                total_commits=0,
                total_files=0,
                primary_language="",
                status="active"
            )
            db_manager.save_repository(repository)
        
        # Store in global state for API access
        active_repos[repo_id] = git_analyzer
        print(f"Stored as repo_id: {repo_id}")
        
        # Get repository stats
        stats = git_analyzer.analyze_repository_stats()
        print(f"Got repository stats: {stats}")
        
        # Index repository in RAG system for intelligent querying
        background_tasks.add_task(index_repository_for_rag, repo_id, repo_url_str, git_analyzer)
        
        # Store commits in database for future analysis
        background_tasks.add_task(store_repository_commits, repo_id, git_analyzer)
        
        return {
            "repo_id": repo_id,
            "status": "cloned",
            "stats": RepositoryStatsResponse(**stats),
            "message": f"Successfully cloned repository from {repo_url_str}"
        }
        
    except Exception as e:
        print(f"Error cloning repository: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        
        # Cleanup on failure
        try:
            repo_id = storage_manager.generate_repo_id(repo_url_str)
            storage_manager.cleanup_repository(repo_id)
        except:
            pass
        
        raise HTTPException(status_code=400, detail=f"Failed to clone repository: {str(e)}")

# ===== CHAT SYSTEM ENDPOINTS =====

@app.post("/api/chats")
async def create_chat(request: CreateChatRequest, current_user: str = Depends(require_auth)):
    """Create a new chat conversation"""
    try:
        chat_id = f"chat_{secrets.token_urlsafe(16)}"
        now = datetime.now().isoformat()
        
        chat = Chat(
            id=chat_id,
            user_id=current_user,
            repo_id=request.repo_id,
            title=request.title,
            created_at=now,
            updated_at=now,
            is_archived=False
        )
        
        if not db_manager.save_chat(chat):
            raise HTTPException(status_code=500, detail="Failed to create chat")
        
        return ChatResponse(
            id=chat.id,
            user_id=chat.user_id,
            repo_id=chat.repo_id,
            title=chat.title,
            created_at=chat.created_at,
            updated_at=chat.updated_at,
            is_archived=chat.is_archived,
            message_count=0
        )
        
    except Exception as e:
        logger.error(f"Error creating chat: {e}")
        raise HTTPException(status_code=500, detail="Failed to create chat")

@app.get("/api/chats")
async def get_user_chats(
    current_user: str = Depends(require_auth),
    include_archived: bool = Query(False, description="Include archived chats")
):
    """Get user's chat conversations"""
    try:
        chats = db_manager.get_user_chats(current_user, include_archived)
        
        return [
            ChatResponse(
                id=chat.id,
                user_id=chat.user_id,
                repo_id=chat.repo_id,
                title=chat.title,
                created_at=chat.created_at,
                updated_at=chat.updated_at,
                is_archived=chat.is_archived,
                message_count=len(db_manager.get_chat_messages(chat.id))
            )
            for chat in chats
        ]
        
    except Exception as e:
        logger.error(f"Error getting chats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get chats")

@app.get("/api/chats/{chat_id}")
async def get_chat(chat_id: str, current_user: str = Depends(require_auth)):
    """Get chat details and messages"""
    try:
        chat = db_manager.get_chat(chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        if chat.user_id != current_user:
            raise HTTPException(status_code=403, detail="Access denied")
        
        messages = db_manager.get_chat_messages(chat_id)
        
        return {
            "chat": ChatResponse(
                id=chat.id,
                user_id=chat.user_id,
                repo_id=chat.repo_id,
                title=chat.title,
                created_at=chat.created_at,
                updated_at=chat.updated_at,
                is_archived=chat.is_archived,
                message_count=len(messages)
            ),
            "messages": [
                ChatMessageResponse(
                    id=msg.id,
                    chat_id=msg.chat_id,
                    user_id=msg.user_id,
                    message=msg.message,
                    response=msg.response,
                    query_type=msg.query_type,
                    context_used=_parse_context_used(msg.context_used),
                    confidence=msg.confidence,
                    timestamp=msg.timestamp,
                    claude_enhanced=msg.claude_enhanced
                )
                for msg in messages
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chat: {e}")
        raise HTTPException(status_code=500, detail="Failed to get chat")

@app.post("/api/chats/{chat_id}/messages")
async def send_chat_message(
    chat_id: str, 
    request: ChatMessageRequest, 
    current_user: str = Depends(require_auth)
):
    """Send a message to a chat and get AI response"""
    try:
        # Verify chat exists and user has access
        chat = db_manager.get_chat(chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        if chat.user_id != current_user:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Process query with RAG system
        if chat.repo_id and chat.repo_id in active_repos:
            # Repository-specific query - create RAG system with specific git_analyzer
            git_analyzer = active_repos[chat.repo_id]
            repo_rag_system = RAGSystem(git_analyzer, claude_analyzer=claude_analyzer)
            rag_result = repo_rag_system.enhanced_query(chat.repo_id, request.message, max_results=10)
            query_type = "repository"
            claude_enhanced = True  # RAG system provides enhanced analysis
        else:
            # General query (fallback response)
            rag_result = RepositoryQueryResponse(
                query=request.message,
                response="I'm ready to help analyze repositories once you've cloned one.",
                confidence=1.0,
                sources=[],
                context_used=[],
                suggestions=["Clone a repository first using the /api/clone-repository endpoint"],
                claude_enhanced=False
            )
            query_type = "general"
            claude_enhanced = False
        
        # Save message and response
        message_id = f"msg_{secrets.token_urlsafe(16)}"
        now = datetime.now().isoformat()
        
        # Ensure context_used is a list of dictionaries for ChatMessageResponse
        context_used = getattr(rag_result, 'context_used', [])
        if isinstance(context_used, str):
            # If it's a string, wrap it in a list or try to parse it
            try:
                context_used = json.loads(context_used) if context_used.startswith('[') else [{"content": context_used}]
            except:
                context_used = [{"content": context_used}] if context_used else []
        elif isinstance(context_used, list):
            # Convert list of strings to list of dictionaries
            formatted_context = []
            for item in context_used:
                if isinstance(item, str):
                    formatted_context.append({"content": item})
                elif isinstance(item, dict):
                    formatted_context.append(item)
                else:
                    formatted_context.append({"content": str(item)})
            context_used = formatted_context
        else:
            context_used = []
        
        chat_message = ChatMessage(
            id=message_id,
            chat_id=chat_id,
            user_id=current_user,
            message=request.message,
            response=rag_result.response,
            query_type=query_type,
            context_used=json.dumps(context_used),
            confidence=getattr(rag_result, 'confidence', 1.0),
            timestamp=now,
            claude_enhanced=claude_enhanced
        )
        
        if not db_manager.save_chat_message(chat_message):
            raise HTTPException(status_code=500, detail="Failed to save message")
        
        # Update chat timestamp
        db_manager.update_chat_timestamp(chat_id)
        
        return ChatMessageResponse(
            id=chat_message.id,
            chat_id=chat_message.chat_id,
            user_id=chat_message.user_id,
            message=chat_message.message,
            response=chat_message.response,
            query_type=chat_message.query_type,
            context_used=context_used,  # Use the already parsed context_used
            confidence=chat_message.confidence,
            timestamp=chat_message.timestamp,
            claude_enhanced=chat_message.claude_enhanced
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        raise HTTPException(status_code=500, detail="Failed to send message")

@app.put("/api/chats/{chat_id}")
async def update_chat(
    chat_id: str, 
    request: UpdateChatTitleRequest, 
    current_user: str = Depends(require_auth)
):
    """Update chat title"""
    try:
        chat = db_manager.get_chat(chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        if chat.user_id != current_user:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not db_manager.update_chat_title(chat_id, request.title):
            raise HTTPException(status_code=500, detail="Failed to update chat")
        
        return {"message": "Chat title updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating chat: {e}")
        raise HTTPException(status_code=500, detail="Failed to update chat")

@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str, current_user: str = Depends(require_auth)):
    """Delete a chat conversation"""
    try:
        chat = db_manager.get_chat(chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        if chat.user_id != current_user:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not db_manager.delete_chat(chat_id):
            raise HTTPException(status_code=500, detail="Failed to delete chat")
        
        return {"message": "Chat deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting chat: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete chat")

@app.post("/api/chats/{chat_id}/archive")
async def archive_chat(chat_id: str, current_user: str = Depends(require_auth)):
    """Archive a chat conversation"""
    try:
        chat = db_manager.get_chat(chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        if chat.user_id != current_user:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not db_manager.archive_chat(chat_id):
            raise HTTPException(status_code=500, detail="Failed to archive chat")
        
        return {"message": "Chat archived successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error archiving chat: {e}")
        raise HTTPException(status_code=500, detail="Failed to archive chat")

# ===== Legacy endpoints removed - all functionality moved to enhanced systems =====

@app.get("/api/repository/{repo_id}/files")
async def list_repository_files(
    repo_id: str,
    current_user: str = Depends(require_repository_access)
):
    """List files in the repository for selection"""
    try:
        if repo_id not in active_repos:
            raise HTTPException(status_code=404, detail="Repository not found.")
        
        git_analyzer = active_repos[repo_id]
        
        # Get all files from the latest commit
        repo = git_analyzer.repo
        tree = repo.head.commit.tree
        
        files = []
        for item in tree.traverse():
            if item.type == 'blob':  # It's a file
                files.append({
                    'path': item.path,
                    'size': item.size,
                    'type': 'file'
                })
        
        # Filter for code files
        code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs', '.rb', '.php'}
        code_files = [f for f in files if any(f['path'].endswith(ext) for ext in code_extensions)]
        
        return {
            'total_files': len(files),
            'code_files': len(code_files),
            'files': code_files[:100]  # Limit to first 100 for demo
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")

@app.get("/api/repository/{repo_id}/stats")
async def get_repository_stats(
    repo_id: str,
    current_user: str = Depends(require_repository_access)
):
    """Get detailed repository statistics"""
    try:
        if repo_id not in active_repos:
            raise HTTPException(status_code=404, detail="Repository not found.")
        
        git_analyzer = active_repos[repo_id]
        stats = git_analyzer.analyze_repository_stats()
        
        return RepositoryStatsResponse(**stats)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@app.delete("/api/repository/{repo_id}")
async def cleanup_repository(
    repo_id: str,
    current_user: str = Depends(require_repository_access)
):
    """Clean up repository resources completely - filesystem and database"""
    try:
        # Get repository details before deletion
        repository = db_manager.get_repository(repo_id)
        if not repository:
            raise HTTPException(status_code=404, detail="Repository not found")
        
        # Clean up in-memory resources
        if repo_id in active_repos:
            del active_repos[repo_id]
        
        if repo_id in temp_dirs:
            temp_dir = temp_dirs[repo_id]
            shutil.rmtree(temp_dir, ignore_errors=True)
            del temp_dirs[repo_id]
        
        # Clean up filesystem - remove repository directory
        repo_local_path = repository.local_path
        if repo_local_path and os.path.exists(repo_local_path):
            logger.info(f"Removing repository directory: {repo_local_path}")
            shutil.rmtree(repo_local_path, ignore_errors=True)
        
        # Also check common storage paths
        storage_paths_to_check = [
            f"./repos/active/{repo_id}",
            f"./repos/cache/{repo_id}",
            f"./repos/temp/{repo_id}",
            f"./knowledge_base/{repo_id}"
        ]
        
        for path in storage_paths_to_check:
            if os.path.exists(path):
                logger.info(f"Removing storage path: {path}")
                shutil.rmtree(path, ignore_errors=True)
        
        # Clean up database - remove all related data
        success = await _cleanup_repository_from_database(repo_id)
        
        if not success:
            logger.warning(f"Some database cleanup may have failed for repo {repo_id}")
        
        return {
            "message": f"Repository {repo_id} completely cleaned up",
            "filesystem_cleaned": True,
            "database_cleaned": success
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cleanup failed for repository {repo_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

async def _cleanup_repository_from_database(repo_id: str) -> bool:
    """Comprehensive database cleanup for a repository"""
    try:
        import sqlite3
        
        with sqlite3.connect(db_manager.db_path) as conn:
            cursor = conn.cursor()
            
            # Delete from all tables that reference this repository
            cleanup_queries = [
                "DELETE FROM file_history WHERE repo_id = ?",
                "DELETE FROM file_content WHERE repo_id = ?", 
                "DELETE FROM commit_file_cache WHERE repo_id = ?",
                "DELETE FROM repository_dashboards WHERE repo_id = ?",
                "DELETE FROM commit_analysis WHERE repo_id = ?",
                "DELETE FROM chat_messages WHERE chat_id IN (SELECT id FROM chats WHERE repo_id = ?)",
                "DELETE FROM chats WHERE repo_id = ?",
                "DELETE FROM files WHERE repo_id = ?",
                "DELETE FROM commits WHERE repo_id = ?",
                "DELETE FROM repositories WHERE id = ?"
            ]
            
            # Execute cleanup queries
            for query in cleanup_queries:
                cursor.execute(query, (repo_id,))
                logger.info(f"Executed cleanup query: {query} for repo {repo_id}")
            
            conn.commit()
            logger.info(f"Database cleanup completed for repository {repo_id}")
            return True
            
    except Exception as e:
        logger.error(f"Database cleanup failed for repository {repo_id}: {e}")
        return False

@app.get("/api/repository/{repo_id}/commits")
async def get_repository_commits(
    repo_id: str, 
    limit: int = Query(50, description="Maximum number of commits to return"),
    current_user: str = Depends(require_repository_access)
):
    """Get list of commits for a repository"""
    try:
        if repo_id not in active_repos:
            raise HTTPException(status_code=404, detail="Repository not found.")
        
        git_analyzer = active_repos[repo_id]
        repo = git_analyzer.repo
        
        # Get recent commits
        commits = []
        for commit in repo.iter_commits(max_count=limit):
            commits.append({
                "hash": commit.hexsha,
                "short_hash": commit.hexsha[:8],
                "message": commit.message.strip(),
                "author": commit.author.name,
                "timestamp": datetime.fromtimestamp(commit.committed_date).isoformat(),
                "files_changed": len(commit.stats.files),
                "insertions": commit.stats.total['insertions'],
                "deletions": commit.stats.total['deletions']
            })
        
        return {
            "repo_id": repo_id,
            "total_commits": len(commits),
            "commits": commits
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get commits: {str(e)}")

# ===== AUTHENTICATION ENDPOINTS =====

@app.get("/api/auth/github")
async def github_oauth_init():
    """Initiate GitHub OAuth flow"""
    if not github_service.available:
        raise HTTPException(status_code=501, detail="GitHub OAuth not configured")
    
    oauth_url, state = github_service.get_oauth_url()
    
    return {
        "oauth_url": oauth_url,
        "state": state
    }

@app.get("/api/auth/github/callback")
async def github_oauth_callback(
    code: str = Query(...),
    state: str = Query(...),
    referer: str = Header(None)
):
    """Handle GitHub OAuth callback and redirect to frontend"""
    if not github_service.available:
        raise HTTPException(status_code=501, detail="GitHub OAuth not configured")
    
    try:
        # Exchange code for access token
        access_token = github_service.exchange_code_for_token(code, state)
        if not access_token:
            raise HTTPException(status_code=400, detail="Failed to exchange OAuth code")
        
        # Get user information
        github_user = github_service.get_user_info(access_token)
        if not github_user:
            raise HTTPException(status_code=400, detail="Failed to get user information")
        
        # Encrypt and store the access token
        encrypted_token = github_service.encrypt_token(access_token)
        
        # Create or update user in database
        now = datetime.now().isoformat()
        user = User(
            id=github_user.id,
            github_username=github_user.username,
            github_avatar=github_user.avatar_url,
            github_email=github_user.email,
            display_name=github_user.name,
            access_token=encrypted_token,
            created_at=now,
            last_login=now,
            is_active=True
        )
        
        # Check if user exists
        existing_user = db_manager.get_user(github_user.id)
        if existing_user:
            # Update existing user
            user.created_at = existing_user.created_at
            db_manager.update_user_last_login(github_user.id)
        
        # Save user
        if not db_manager.save_user(user):
            raise HTTPException(status_code=500, detail="Failed to save user")
        
        # Create session
        session_id = generate_session_id()
        user_sessions[session_id] = github_user.id
        
        # Determine redirect URL from Referer header
        if referer:
            # Extract the origin from the referer
            from urllib.parse import urlparse
            parsed = urlparse(referer)
            frontend_origin = f"{parsed.scheme}://{parsed.netloc}"
        else:
            # Fallback to localhost
            frontend_origin = "http://localhost:3000"
        
        # Redirect to frontend with session ID
        redirect_url = f"{frontend_origin}/auth/callback?session_id={session_id}&success=true"
        return RedirectResponse(url=redirect_url, status_code=302)
        
    except Exception as e:
        # Determine redirect URL for error case
        if referer:
            from urllib.parse import urlparse
            parsed = urlparse(referer)
            frontend_origin = f"{parsed.scheme}://{parsed.netloc}"
        else:
            frontend_origin = "http://localhost:3000"
        
        error_url = f"{frontend_origin}/auth/callback?error={str(e)}&success=false"
        return RedirectResponse(url=error_url, status_code=302)

@app.get("/api/auth/user")
async def get_current_user_info(current_user: str = Depends(require_auth)):
    """Get current user information"""
    user = db_manager.get_user(current_user)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return UserResponse(
        id=user.id,
        github_username=user.github_username,
        github_avatar=user.github_avatar,
        github_email=user.github_email,
        display_name=user.display_name,
        created_at=user.created_at,
        last_login=user.last_login
    )

@app.post("/api/auth/logout")
async def logout(authorization: str = Header(None)):
    """Logout and invalidate session"""
    if authorization and authorization.startswith("Bearer "):
        session_id = authorization[7:]
        if session_id in user_sessions:
            del user_sessions[session_id]
    
    return {"message": "Logged out successfully"}

@app.get("/api/auth/github/repositories")
async def get_github_repositories(current_user: str = Depends(require_auth)):
    """Get user's GitHub repositories (including private ones)"""
    if not github_service.available:
        raise HTTPException(status_code=501, detail="GitHub service not available")
    
    user = db_manager.get_user(current_user)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    try:
        # Decrypt access token
        access_token = github_service.decrypt_token(user.access_token)
        
        # Get repositories
        repositories = github_service.get_user_repositories(access_token)
        
        return [
            GitHubRepositoryResponse(
                id=repo.id,
                name=repo.name,
                full_name=repo.full_name,
                clone_url=repo.clone_url,
                private=repo.private,
                description=repo.description,
                language=repo.language,
                default_branch=repo.default_branch
            )
            for repo in repositories
        ]
        
    except Exception as e:
        logger.error(f"Error getting GitHub repositories: {e}")
        raise HTTPException(status_code=500, detail="Failed to get repositories")

@app.get("/api/user/repositories")
async def get_user_repositories(current_user: str = Depends(require_auth)):
    """Get repositories associated with the current user"""
    try:
        repositories = db_manager.get_user_repositories(current_user)
        
        return [
            {
                "id": repo.id,
                "url": repo.url,
                "name": repo.name,
                "local_path": repo.local_path,
                "created_at": repo.created_at,
                "updated_at": repo.updated_at,
                "total_commits": repo.total_commits,
                "total_files": repo.total_files,
                "primary_language": repo.primary_language,
                "status": repo.status
            }
            for repo in repositories
        ]
        
    except Exception as e:
        logger.error(f"Error getting user repositories: {e}")
        raise HTTPException(status_code=500, detail="Failed to get repositories")

@app.post("/api/analyze-commit/{repo_id}")
async def analyze_commit_breaking_changes(repo_id: str, request: AnalyzeCommitRequest):
    """Analyze a specific commit for breaking changes"""
    try:
        if repo_id not in active_repos:
            raise HTTPException(status_code=404, detail="Repository not found. Please clone first.")
        
        git_analyzer = active_repos[repo_id]
        
        # Analyze commit for breaking changes
        analysis = breaking_change_detector.analyze_commit(git_analyzer, request.commit_hash)
        
        # Enhance with Claude analysis if requested
        claude_analysis = None
        if request.include_claude_analysis and claude_analyzer.available:
            breaking_changes_data = [
                {
                    "change_type": bc.change_type.value,
                    "risk_level": bc.risk_level.value,
                    "confidence": bc.confidence,
                    "file_path": bc.file_path,
                    "description": bc.description,
                    "before_code": bc.before_code,
                    "after_code": bc.after_code,
                }
                for bc in analysis.breaking_changes
            ]
            
            commit_context = {
                "commit_hash": analysis.commit_hash,
                "message": analysis.commit_message,
                "author": analysis.author,
                "files_changed": analysis.files_changed
            }
            
            claude_response = claude_analyzer.analyze_commit_breaking_changes(analysis, "")
            claude_analysis = {
                "content": claude_response.get("claude_analysis", {}).get("summary", ""),
                "confidence": claude_response.get("confidence", 0.0),
                "suggestions": claude_response.get("recommendations", []),
                "metadata": claude_response.get("model_used", "")
            }
        
        # Convert to response format
        breaking_changes_response = [
            BreakingChangeResponse(
                change_type=bc.change_type.value,
                risk_level=bc.risk_level.value,
                confidence=float(bc.confidence),
                file_path=bc.file_path,
                line_number=bc.line_number,
                description=bc.description,
                before_code=bc.before_code,
                after_code=bc.after_code,
                impact_analysis=convert_numpy_types(bc.impact_analysis),
                mitigation_suggestions=bc.mitigation_suggestions,
                related_files=bc.related_files
            )
            for bc in analysis.breaking_changes
        ]
        
        return CommitAnalysisResponse(
            commit_hash=analysis.commit_hash,
            commit_message=analysis.commit_message,
            timestamp=analysis.timestamp,
            author=analysis.author,
            overall_risk_score=float(analysis.overall_risk_score),
            breaking_changes=breaking_changes_response,
            files_changed=analysis.files_changed,
            lines_added=analysis.lines_added,
            lines_removed=analysis.lines_removed,
            complexity_score=float(analysis.complexity_score),
            semantic_drift_score=float(analysis.semantic_drift_score),
            claude_analysis=claude_analysis
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/analyze-commit-range/{repo_id}")
async def analyze_commit_range(repo_id: str, request: AnalyzeCommitRangeRequest):
    """Analyze a range of commits for breaking changes"""
    try:
        if repo_id not in active_repos:
            raise HTTPException(status_code=404, detail="Repository not found. Please clone first.")
        
        git_analyzer = active_repos[repo_id]
        
        # Analyze commit range
        analyses = breaking_change_detector.analyze_commit_range(
            git_analyzer, request.start_commit, request.end_commit
        )
        
        # Limit results if requested
        if request.max_commits > 0:
            analyses = analyses[:request.max_commits]
        
        # Convert to response format
        results = []
        for analysis in analyses:
            # Enhance with Claude analysis if requested
            claude_analysis = None
            if request.include_claude_analysis and claude_analyzer.available and analysis.breaking_changes:
                breaking_changes_data = [
                    {
                        "change_type": bc.change_type.value,
                        "risk_level": bc.risk_level.value,
                        "confidence": bc.confidence,
                        "file_path": bc.file_path,
                        "description": bc.description,
                    }
                    for bc in analysis.breaking_changes
                ]
                
                commit_context = {
                    "commit_hash": analysis.commit_hash,
                    "message": analysis.commit_message,
                    "author": analysis.author,
                    "files_changed": analysis.files_changed
                }
                
                claude_response = claude_analyzer.analyze_commit_breaking_changes(analysis, "")
                claude_analysis = {
                    "content": claude_response.get("claude_analysis", {}).get("summary", ""),
                    "confidence": claude_response.get("confidence", 0.0),
                    "suggestions": claude_response.get("recommendations", [])
                }
            
            breaking_changes_response = [
                BreakingChangeResponse(
                    change_type=bc.change_type.value,
                    risk_level=bc.risk_level.value,
                    confidence=float(bc.confidence),
                    file_path=bc.file_path,
                    line_number=bc.line_number,
                    description=bc.description,
                    before_code=bc.before_code,
                    after_code=bc.after_code,
                    impact_analysis=convert_numpy_types(bc.impact_analysis),
                    mitigation_suggestions=bc.mitigation_suggestions,
                    related_files=bc.related_files
                )
                for bc in analysis.breaking_changes
            ]
            
            results.append(CommitAnalysisResponse(
                commit_hash=analysis.commit_hash,
                commit_message=analysis.commit_message,
                timestamp=analysis.timestamp,
                author=analysis.author,
                overall_risk_score=float(analysis.overall_risk_score),
                breaking_changes=breaking_changes_response,
                files_changed=analysis.files_changed,
                lines_added=analysis.lines_added,
                lines_removed=analysis.lines_removed,
                complexity_score=float(analysis.complexity_score),
                semantic_drift_score=float(analysis.semantic_drift_score),
                claude_analysis=claude_analysis
            ))
        
        return {
            "total_commits_analyzed": len(results),
            "high_risk_commits": len([r for r in results if r.overall_risk_score > 0.7]),
            "total_breaking_changes": sum(len(r.breaking_changes) for r in results),
            "commits": results
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/query-repository/{repo_id}")
async def query_repository(repo_id: str, request: QueryRepositoryRequest):
    """Enhanced intelligent query using Smart RAG + Multi-Expert Claude Analysis"""
    
    try:
        if repo_id not in active_repos:
            raise HTTPException(status_code=404, detail="Repository not found. Please clone first.")
        
        git_analyzer = active_repos[repo_id]
        
        # Step 1: Smart Code Analysis - Deep understanding of repository structure
        from src.code_analyzer import CodeAnalyzer
        code_analyzer = CodeAnalyzer()
        
        # Analyze repository structure and get relevant files
        relevant_files = code_analyzer.find_relevant_files(git_analyzer, request.query)
        file_analyses = []
        
        for file_path in relevant_files[:10]:  # Analyze top 10 most relevant files
            try:
                analysis = code_analyzer.analyze_file_deep(git_analyzer, file_path)
                file_analyses.append(analysis)
            except Exception as e:
                logger.warning(f"Failed to analyze file {file_path}: {e}")
                continue
        
        # Step 2: Enhanced RAG with intelligent context gathering
        from src.rag_system import RAGSystem
        enhanced_rag = RAGSystem(semantic_analyzer, claude_analyzer)
        
        # Get intelligent query analysis
        query_analysis = enhanced_rag.analyze_query_intent(request.query)
        
        # Gather multi-layered context
        context = enhanced_rag.gather_intelligent_context(
            git_analyzer, request.query, query_analysis, file_analyses
        )
        
        # Step 3: Multi-Expert Claude Analysis
        from src.claude_analyzer import ClaudeAnalyzer
        advanced_claude = ClaudeAnalyzer()
        
        expert_analyses = {}
        specialized_data = context.get('specialized_analysis', {})
        
        if advanced_claude.available:
            # Determine which experts to consult based on query
            relevant_domains = enhanced_rag.determine_expert_domains(request.query, query_analysis)
            
            # Multi-expert consultation
            expert_analyses = advanced_claude.multi_expert_consultation(
                request.query, context, file_analyses, relevant_domains
            )
            
            # Synthesize expert opinions
            expert_synthesis = advanced_claude.synthesize_expert_opinions(expert_analyses)
        else:
            expert_synthesis = {
                'consensus_recommendations': [],
                'unified_risk_assessment': {},
                'cross_domain_insights': []
            }
        
        # Step 4: Smart Suggestions Generation
        from src.suggestions_engine import SuggestionsEngine
        suggestions_engine = SuggestionsEngine()
        
        smart_suggestions = suggestions_engine.generate_comprehensive_suggestions(
            request.query, context, file_analyses, expert_analyses
        )
        
        # Step 5: Build enhanced response
        # Primary response - best expert or enhanced RAG
        if expert_analyses:
            # Use the highest confidence expert analysis as primary response
            best_expert = max(expert_analyses.values(), key=lambda x: x.confidence)
            primary_response = best_expert.analysis
            confidence = best_expert.confidence
        else:
            # Fallback to enhanced RAG
            primary_response = enhanced_rag.generate_intelligent_response(context, request.query)
            confidence = context.get('confidence', 0.7)
        
        # Combine all sources
        sources = []
        
        # Add file sources
        for analysis in file_analyses:
            sources.append({
                'type': 'file',
                'path': analysis.file_path,
                'relevance': analysis.relevance_score,
                'complexity': analysis.complexity_score,
                'language': analysis.language
            })
        
        # Add commit sources if available
        for commit in context.get('commits', [])[:5]:
            sources.append({
                'type': 'commit',
                'hash': commit.get('hash', ''),
                'message': commit.get('message', ''),
                'relevance': commit.get('relevance_score', 0.5)
            })
        
        # Enhanced context used
        context_used = []
        
        # Add expert insights
        for domain, analysis in expert_analyses.items():
            context_used.append({
                'type': 'expert_analysis',
                'domain': domain,
                'expert_level': analysis.expert_level,
                'confidence': analysis.confidence,
                'key_insights': analysis.analysis[:200] + "..." if len(analysis.analysis) > 200 else analysis.analysis
            })
        
        # Add code analysis context
        if file_analyses:
            context_used.append({
                'type': 'code_analysis',
                'files_analyzed': len(file_analyses),
                'patterns_detected': sum(len(fa.architecture_patterns) for fa in file_analyses),
                'security_issues': sum(len(fa.security_patterns) for fa in file_analyses)
            })
        
        # Build comprehensive suggestions combining all sources
        all_suggestions = []
        
        # Smart suggestions (contextual)
        all_suggestions.extend(smart_suggestions.get('contextual_suggestions', []))
        
        # Expert recommendations
        for domain, analysis in expert_analyses.items():
            for rec in analysis.actionable_recommendations[:2]:  # Top 2 per expert
                all_suggestions.append(f"[{domain.title()} Expert] {rec['text']}")
        
        # Enhanced suggestions from synthesis
        for consensus in expert_synthesis.get('consensus_recommendations', [])[:3]:
            domains_str = ", ".join(consensus['supporting_domains'])
            all_suggestions.append(f"[Multi-Expert Consensus: {domains_str}] {consensus['primary_recommendation']}")
        
        # Intelligent follow-up suggestions
        all_suggestions.extend(smart_suggestions.get('follow_up_suggestions', []))
        
        # Limit suggestions to most relevant
        final_suggestions = all_suggestions[:8]
        
        return RepositoryQueryResponse(
            query=request.query,
            response=primary_response,
            confidence=confidence,
            sources=sources,
            context_used=[item.get('content', str(item)) if isinstance(item, dict) else str(item) for item in context_used],
            suggestions=final_suggestions,
            claude_enhanced=advanced_claude.available
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Enhanced query failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/api/repository/{repo_id}/summary")
async def get_repository_summary(
    repo_id: str,
    current_user: str = Depends(require_repository_access)
):
    """Get comprehensive repository summary with risk analysis"""
    try:
        if repo_id not in active_repos:
            raise HTTPException(status_code=404, detail="Repository not found.")
        
        # Get summary from RAG system
        summary = rag_system.get_repository_summary(repo_id)
        
        # Enhance with Claude risk assessment if available
        if claude_analyzer.available:
            recent_changes = summary.get("recent_activity", [])
            claude_response = claude_analyzer.analyze_repository_risk(summary)
            
            summary["claude_risk_assessment"] = {
                "content": claude_response.get("risk_assessment", {}).get("summary", ""),
                "confidence": claude_response.get("overall_risk_level", "unknown"),
                "suggestions": claude_response.get("recommendations", []),
                "metadata": claude_response.get("model_used", "")
            }
        
        return summary
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get summary: {str(e)}")

@app.get("/api/repository/{repo_id}/commits/search")
async def search_commits(
    repo_id: str, 
    query: str = Query(...), 
    max_results: int = Query(10),
    current_user: str = Depends(require_repository_access)
):
    """Search commits using semantic search"""
    try:
        if repo_id not in active_repos:
            raise HTTPException(status_code=404, detail="Repository not found.")
        
        results = rag_system.search_commits(repo_id, query, max_results)
        
        return {
            "query": query,
            "total_results": len(results),
            "commits": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/api/repository/{repo_id}/files/search")
async def search_files(
    repo_id: str, 
    query: str = Query(...), 
    max_results: int = Query(10),
    current_user: str = Depends(require_repository_access)
):
    """Search files using semantic search"""
    try:
        if repo_id not in active_repos:
            raise HTTPException(status_code=404, detail="Repository not found.")
        
        results = rag_system.search_files(repo_id, query, max_results)
        
        return {
            "query": query,
            "total_results": len(results),
            "files": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/api/repository/{repo_id}/risk-dashboard")
async def get_risk_dashboard(
    repo_id: str, 
    force_refresh: bool = Query(False, description="Force refresh of dashboard"),
    current_user: str = Depends(require_repository_access)
):
    """Get comprehensive risk dashboard for repository with intelligent caching"""
    try:
        if repo_id not in active_repos:
            raise HTTPException(status_code=404, detail="Repository not found.")
        
        # Check if we have a valid cached dashboard
        if not force_refresh and db_manager.is_dashboard_cache_valid(repo_id, max_age_hours=6):
            logger.info(f"Using cached dashboard for {repo_id}")
            cached_dashboard = db_manager.get_dashboard_cache(repo_id)
            if cached_dashboard:
                return json.loads(cached_dashboard.dashboard_data)
        
        logger.info(f"Generating fresh dashboard for {repo_id}")
        git_analyzer = active_repos[repo_id]
        repo = git_analyzer.repo
        
        # Get recent commits (limited to speed up analysis)
        recent_commits = list(repo.iter_commits(max_count=100))
        
        # Check which commits we've already analyzed
        analyzed_commits = {analysis.commit_hash: analysis 
                          for analysis in db_manager.get_repository_commit_analyses(repo_id, limit=200)}
        
        # Analyze only new commits using advanced breaking change detector
        new_analyses = []
        for commit in recent_commits[:50]:  # Limit to 50 most recent for dashboard
            if commit.hexsha not in analyzed_commits:
                try:
                    logger.info(f"🔍 Advanced analysis for commit: {commit.hexsha[:8]}")
                    # Use the advanced breaking change detector
                    breaking_changes = breaking_change_detector.analyze_commit_for_breaking_changes(commit.hexsha, git_analyzer)
                    
                    # Calculate enhanced risk metrics
                    critical_changes = [bc for bc in breaking_changes if bc.severity.value == "critical"]
                    high_changes = [bc for bc in breaking_changes if bc.severity.value == "high"]
                    accidental_changes = [bc for bc in breaking_changes if bc.intent.value == "accidental"]
                    
                    # Enhanced risk score calculation
                    risk_score = 0.0
                    if breaking_changes:
                        severity_weights = {"critical": 1.0, "high": 0.8, "medium": 0.5, "low": 0.2, "enhancement": 0.1}
                        intent_weights = {"accidental": 1.5, "unclear": 1.2, "intentional": 1.0}
                        
                        total_weight = 0
                        for bc in breaking_changes:
                            severity_weight = severity_weights.get(bc.severity.value, 0.5)
                            intent_weight = intent_weights.get(bc.intent.value, 1.0)
                            migration_complexity_weight = {"very_complex": 1.5, "complex": 1.3, "moderate": 1.1, "easy": 0.9, "trivial": 0.7}.get(bc.migration_complexity, 1.0)
                            
                            change_risk = severity_weight * intent_weight * migration_complexity_weight * bc.confidence_score
                            total_weight += change_risk
                        
                        risk_score = min(total_weight / len(breaking_changes), 1.0)
                    
                    # Get basic analysis for backward compatibility
                    analysis = breaking_change_detector.analyze_commit(git_analyzer, commit.hexsha)
                    
                    # Enhanced commit analysis with advanced detector data
                    commit_analysis = CommitAnalysis(
                        id=f"{repo_id}_{commit.hexsha}",
                        repo_id=repo_id,
                        commit_hash=commit.hexsha,
                        overall_risk_score=float(max(risk_score, analysis.overall_risk_score)),  # Use higher of the two calculations
                        breaking_changes_count=len(breaking_changes),
                        complexity_score=float(analysis.complexity_score),
                        semantic_drift_score=float(analysis.semantic_drift_score),
                        files_changed=json.dumps(analysis.files_changed),
                        breaking_changes=json.dumps([{
                            "change_type": bc.change_type.value,
                            "severity": bc.severity.value,
                            "intent": bc.intent.value,
                            "confidence_score": bc.confidence_score,
                            "file_path": bc.file_path,
                            "description": bc.description,
                            "migration_complexity": bc.migration_complexity,
                            "affected_users_estimate": bc.affected_users_estimate,
                            "detection_methods": bc.detection_methods,
                            "old_signature": bc.old_signature,
                            "new_signature": bc.new_signature
                        } for bc in breaking_changes]),
                        analyzed_at=datetime.now().isoformat()
                    )
                    
                    if db_manager.save_commit_analysis(commit_analysis):
                        analyzed_commits[commit.hexsha] = commit_analysis
                        new_analyses.append(commit_analysis)
                    
                except Exception as e:
                    logger.error(f"Error analyzing commit {commit.hexsha}: {e}")
                    continue
        
        # Generate enhanced dashboard from all analyzed commits
        risk_data = {
            "total_commits_analyzed": len(analyzed_commits),
            "high_risk_commits": 0,
            "critical_risk_commits": 0,
            "accidental_breaking_changes": 0,
            "intentional_breaking_changes": 0,
            "breaking_changes_by_type": {},
            "breaking_changes_by_severity": {},
            "breaking_changes_by_intent": {},
            "migration_complexity_breakdown": {},
            "risk_trend": [],
            "most_risky_files": {},
            "recent_high_risk_commits": [],
            "advanced_insights": {
                "most_dangerous_change_types": {},
                "files_with_most_breaking_changes": {},
                "average_confidence_score": 0.0,
                "detection_method_effectiveness": {}
            },
            "performance_stats": {
                "new_commits_analyzed": len(new_analyses),
                "cached_analyses_used": len(analyzed_commits) - len(new_analyses),
                "cache_hit_rate": round((len(analyzed_commits) - len(new_analyses)) / max(len(analyzed_commits), 1) * 100, 1)
            }
        }
        
        # Process cached analyses for enhanced dashboard
        total_confidence = 0
        confidence_count = 0
        
        for analysis in analyzed_commits.values():
            # Basic risk categorization
            if analysis.overall_risk_score > 0.7:
                risk_data["high_risk_commits"] += 1
                risk_data["recent_high_risk_commits"].append({
                    "commit_hash": analysis.commit_hash[:8],
                    "risk_score": float(analysis.overall_risk_score),
                    "breaking_changes_count": analysis.breaking_changes_count
                })
            
            if analysis.overall_risk_score > 0.9:
                risk_data["critical_risk_commits"] += 1
            
            # Enhanced breaking change analysis from stored data
            try:
                breaking_changes = json.loads(analysis.breaking_changes)
                for bc in breaking_changes:
                    # Count by type
                    change_type = bc.get("change_type", "unknown")
                    risk_data["breaking_changes_by_type"][change_type] = \
                        risk_data["breaking_changes_by_type"].get(change_type, 0) + 1
                    
                    # Count by severity
                    severity = bc.get("severity", "unknown")
                    risk_data["breaking_changes_by_severity"][severity] = \
                        risk_data["breaking_changes_by_severity"].get(severity, 0) + 1
                    
                    # Count by intent
                    intent = bc.get("intent", "unknown")
                    risk_data["breaking_changes_by_intent"][intent] = \
                        risk_data["breaking_changes_by_intent"].get(intent, 0) + 1
                    
                    if intent == "accidental":
                        risk_data["accidental_breaking_changes"] += 1
                    elif intent == "intentional":
                        risk_data["intentional_breaking_changes"] += 1
                    
                    # Migration complexity
                    complexity = bc.get("migration_complexity", "unknown")
                    risk_data["migration_complexity_breakdown"][complexity] = \
                        risk_data["migration_complexity_breakdown"].get(complexity, 0) + 1
                    
                    # Advanced insights
                    if severity in ["critical", "high"]:
                        risk_data["advanced_insights"]["most_dangerous_change_types"][change_type] = \
                            risk_data["advanced_insights"]["most_dangerous_change_types"].get(change_type, 0) + 1
                    
                    # Track confidence scores
                    confidence = bc.get("confidence_score", 0)
                    if confidence > 0:
                        total_confidence += confidence
                        confidence_count += 1
                    
                    # Track detection methods
                    detection_methods = bc.get("detection_methods", [])
                    for method in detection_methods:
                        risk_data["advanced_insights"]["detection_method_effectiveness"][method] = \
                            risk_data["advanced_insights"]["detection_method_effectiveness"].get(method, 0) + 1
                    
                    # Track files with breaking changes
                    file_path = bc.get("file_path", "unknown")
                    risk_data["advanced_insights"]["files_with_most_breaking_changes"][file_path] = \
                        risk_data["advanced_insights"]["files_with_most_breaking_changes"].get(file_path, 0) + 1
                        
            except Exception as e:
                logger.debug(f"Error processing breaking changes for {analysis.commit_hash}: {e}")
                pass
            
            # Track risky files
            try:
                files_changed = json.loads(analysis.files_changed)
                for file_path in files_changed:
                    risk_data["most_risky_files"][file_path] = \
                        risk_data["most_risky_files"].get(file_path, 0) + analysis.overall_risk_score
            except:
                pass
            
            # Add to risk trend
            risk_data["risk_trend"].append({
                "commit_hash": analysis.commit_hash[:8],
                "timestamp": analysis.analyzed_at,
                "risk_score": float(analysis.overall_risk_score),
                "breaking_changes": analysis.breaking_changes_count
            })
        
        # Calculate average confidence
        if confidence_count > 0:
            risk_data["advanced_insights"]["average_confidence_score"] = total_confidence / confidence_count
        
        # Sort and limit results
        risk_data["most_risky_files"] = dict(
            sorted([(k, float(v)) for k, v in risk_data["most_risky_files"].items()], 
                   key=lambda x: x[1], reverse=True)[:10]
        )
        
        risk_data["recent_high_risk_commits"] = sorted(
            risk_data["recent_high_risk_commits"],
            key=lambda x: x["risk_score"], reverse=True
        )[:10]
        
        # Sort risk trend by timestamp (most recent first)
        risk_data["risk_trend"] = sorted(
            risk_data["risk_trend"],
            key=lambda x: x["timestamp"], reverse=True
        )[:30]
        
        # Calculate overall repository risk score
        if risk_data["total_commits_analyzed"] > 0:
            risk_data["overall_risk_score"] = float(risk_data["high_risk_commits"] / risk_data["total_commits_analyzed"])
            
            # Enhanced percentage breakdowns for breaking changes
            total_breaking_changes = sum(risk_data["breaking_changes_by_severity"].values())
            if total_breaking_changes > 0:
                risk_data["breaking_changes_percentages"] = {
                    "by_severity": {
                        severity: {
                            "count": count,
                            "percentage": round((count / total_breaking_changes) * 100, 1)
                        }
                        for severity, count in risk_data["breaking_changes_by_severity"].items()
                    },
                    "by_intent": {
                        intent: {
                            "count": count,
                            "percentage": round((count / total_breaking_changes) * 100, 1)
                        }
                        for intent, count in risk_data["breaking_changes_by_intent"].items()
                    },
                    "by_type": {
                        change_type: {
                            "count": count,
                            "percentage": round((count / total_breaking_changes) * 100, 1)
                        }
                        for change_type, count in risk_data["breaking_changes_by_type"].items()
                    },
                    "by_migration_complexity": {
                        complexity: {
                            "count": count,
                            "percentage": round((count / total_breaking_changes) * 100, 1)
                        }
                        for complexity, count in risk_data["migration_complexity_breakdown"].items()
                    }
                }
            
            # Commit risk percentages
            risk_data["commit_risk_percentages"] = {
                "high_risk": {
                    "count": risk_data["high_risk_commits"],
                    "percentage": round((risk_data["high_risk_commits"] / risk_data["total_commits_analyzed"]) * 100, 1)
                },
                "critical_risk": {
                    "count": risk_data["critical_risk_commits"],
                    "percentage": round((risk_data["critical_risk_commits"] / risk_data["total_commits_analyzed"]) * 100, 1)
                },
                "safe_commits": {
                    "count": risk_data["total_commits_analyzed"] - risk_data["high_risk_commits"],
                    "percentage": round(((risk_data["total_commits_analyzed"] - risk_data["high_risk_commits"]) / risk_data["total_commits_analyzed"]) * 100, 1)
                }
            }
            
            # Intent-based percentages for overall repository
            total_intent_changes = risk_data["accidental_breaking_changes"] + risk_data["intentional_breaking_changes"]
            if total_intent_changes > 0:
                risk_data["intent_risk_percentages"] = {
                    "accidental_percentage": round((risk_data["accidental_breaking_changes"] / total_intent_changes) * 100, 1),
                    "intentional_percentage": round((risk_data["intentional_breaking_changes"] / total_intent_changes) * 100, 1),
                    "accidental_risk_score": round((risk_data["accidental_breaking_changes"] / risk_data["total_commits_analyzed"]) * 100, 1),
                    "intentional_risk_score": round((risk_data["intentional_breaking_changes"] / risk_data["total_commits_analyzed"]) * 100, 1)
                }
        else:
            risk_data["overall_risk_score"] = 0.0
        
        # Convert numpy types
        risk_data = convert_numpy_types(risk_data)
        
        # Cache the dashboard
        dashboard_cache = RepositoryDashboard(
            id=f"dashboard_{repo_id}",
            repo_id=repo_id,
            dashboard_data=json.dumps(risk_data),
            last_updated=datetime.now().isoformat(),
            commits_analyzed=risk_data["total_commits_analyzed"]
        )
        db_manager.save_dashboard_cache(dashboard_cache)
        
        logger.info(f"Dashboard generated for {repo_id}: {len(new_analyses)} new analyses, {risk_data['performance_stats']['cache_hit_rate']}% cache hit rate")
        
        return risk_data
        
    except Exception as e:
        logger.error(f"Failed to generate risk dashboard: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate risk dashboard: {str(e)}")

@app.get("/api/repository/{repo_id}/commit/{commit_hash}/analysis")
async def get_commit_analysis(
    repo_id: str, 
    commit_hash: str,
    current_user: str = Depends(require_repository_access)
):
    """Get detailed analysis of a specific commit with advanced breaking change detection and RAG context"""
    try:
        if repo_id not in active_repos:
            raise HTTPException(status_code=404, detail="Repository not found.")
        
        git_analyzer = active_repos[repo_id]
        
        # Get advanced breaking change analysis with AI for individual commit analysis
        logger.info(f"🔍 Advanced breaking change analysis for commit {commit_hash[:8]}")
        breaking_changes = breaking_change_detector.analyze_commit_for_breaking_changes(commit_hash, git_analyzer, enable_ai_analysis=True)
        
        # Get basic commit analysis for backward compatibility
        commit_analysis = breaking_change_detector.analyze_commit(git_analyzer, commit_hash)
        
        # Enhanced risk calculation using advanced detector
        enhanced_risk_score = 0.0
        if breaking_changes:
            severity_weights = {"critical": 1.0, "high": 0.8, "medium": 0.5, "low": 0.2, "enhancement": 0.1}
            intent_weights = {"accidental": 1.5, "unclear": 1.2, "intentional": 1.0}
            
            total_weight = 0
            for bc in breaking_changes:
                severity_weight = severity_weights.get(bc.severity.value, 0.5)
                intent_weight = intent_weights.get(bc.intent.value, 1.0)
                migration_complexity_weight = {"very_complex": 1.5, "complex": 1.3, "moderate": 1.1, "easy": 0.9, "trivial": 0.7}.get(bc.migration_complexity, 1.0)
                
                change_risk = severity_weight * intent_weight * migration_complexity_weight * bc.confidence_score
                total_weight += change_risk
            
            enhanced_risk_score = min(total_weight / len(breaking_changes), 1.0)
        
        # Use higher of the two risk calculations
        final_risk_score = max(enhanced_risk_score, commit_analysis.overall_risk_score)
        
        # Get RAG context for this commit enhanced with breaking change information
        rag_query = f"analyze commit {commit_hash} changes risk breaking"
        if breaking_changes:
            critical_changes = [bc for bc in breaking_changes if bc.severity.value == "critical"]
            accidental_changes = [bc for bc in breaking_changes if bc.intent.value == "accidental"]
            rag_query += f" [CONTEXT: {len(breaking_changes)} breaking changes found, {len(critical_changes)} critical, {len(accidental_changes)} accidental]"
        
        rag_result = rag_system.query(repo_id, rag_query, max_results=8)
        
        # Enhanced analysis with Claude if available
        enhanced_analysis = None
        if claude_analyzer.available:
            try:
                commit_data = {
                    "commit_hash": commit_analysis.commit_hash,
                    "message": commit_analysis.commit_message,
                    "files_changed": commit_analysis.files_changed,
                    "breaking_changes": [asdict(bc) for bc in commit_analysis.breaking_changes],
                    "advanced_breaking_changes": [{
                        "change_type": bc.change_type.value,
                        "severity": bc.severity.value,
                        "intent": bc.intent.value,
                        "confidence_score": bc.confidence_score,
                        "description": bc.description,
                        "migration_complexity": bc.migration_complexity,
                        "affected_users_estimate": bc.affected_users_estimate,
                        "detection_methods": bc.detection_methods
                    } for bc in breaking_changes],
                    "risk_score": final_risk_score,
                    "enhanced_risk_score": enhanced_risk_score
                }
                enhanced_analysis = claude_analyzer.analyze_commit_breaking_changes(
                    commit_data, rag_result.context_used
                )
            except Exception as e:
                logger.warning(f"Claude analysis failed: {str(e)}")
        
        # Advanced breaking change insights
        breaking_change_insights = {
            "total_breaking_changes": len(breaking_changes),
            "by_severity": {},
            "by_intent": {},
            "by_type": {},
            "migration_complexity": {},
            "high_confidence_changes": [],
            "files_affected": set(),
            "detection_methods_used": set()
        }
        
        for bc in breaking_changes:
            # Count by categories
            breaking_change_insights["by_severity"][bc.severity.value] = \
                breaking_change_insights["by_severity"].get(bc.severity.value, 0) + 1
            breaking_change_insights["by_intent"][bc.intent.value] = \
                breaking_change_insights["by_intent"].get(bc.intent.value, 0) + 1
            breaking_change_insights["by_type"][bc.change_type.value] = \
                breaking_change_insights["by_type"].get(bc.change_type.value, 0) + 1
            breaking_change_insights["migration_complexity"][bc.migration_complexity] = \
                breaking_change_insights["migration_complexity"].get(bc.migration_complexity, 0) + 1
            
            # High confidence changes
            if bc.confidence_score > 0.8:
                breaking_change_insights["high_confidence_changes"].append({
                    "description": bc.description,
                    "confidence": bc.confidence_score,
                    "severity": bc.severity.value,
                    "file": bc.file_path
                })
            
            # Track affected files and detection methods
            breaking_change_insights["files_affected"].add(bc.file_path)
            breaking_change_insights["detection_methods_used"].update(bc.detection_methods)
        
        # Convert sets to lists for JSON serialization
        breaking_change_insights["files_affected"] = list(breaking_change_insights["files_affected"])
        breaking_change_insights["detection_methods_used"] = list(breaking_change_insights["detection_methods_used"])
        
        result = {
            "commit_analysis": asdict(commit_analysis),
            "advanced_breaking_changes": [{
                "id": bc.id,
                "change_type": bc.change_type.value,
                "severity": bc.severity.value,
                "intent": bc.intent.value,
                "confidence_score": bc.confidence_score,
                "affected_component": bc.affected_component,
                "file_path": bc.file_path,
                "line_number": bc.line_number,
                "old_signature": bc.old_signature,
                "new_signature": bc.new_signature,
                "description": bc.description,
                "technical_details": bc.technical_details,
                "migration_complexity": bc.migration_complexity,
                "affected_users_estimate": bc.affected_users_estimate,
                "suggested_migration": bc.suggested_migration,
                "detection_methods": bc.detection_methods,
                "expert_recommendations": bc.expert_recommendations
            } for bc in breaking_changes],
            "breaking_change_insights": breaking_change_insights,
            "rag_insights": {
                "response": rag_result.response,
                "confidence": rag_result.confidence,
                "sources": rag_result.sources,
                "suggestions": rag_result.suggestions
            },
            "enhanced_analysis": enhanced_analysis,
            "risk_assessment": {
                "original_risk_score": commit_analysis.overall_risk_score,
                "enhanced_risk_score": enhanced_risk_score,
                "final_risk_score": final_risk_score,
                "is_high_risk": final_risk_score > 0.7,
                "is_critical_risk": final_risk_score > 0.9,
                "risk_level": "Critical" if final_risk_score > 0.9 
                           else "High" if final_risk_score > 0.7 
                           else "Medium" if final_risk_score > 0.3 
                           else "Low",
                "total_breaking_changes": len(breaking_changes),
                "critical_breaking_changes": len([bc for bc in breaking_changes if bc.severity.value == "critical"]),
                "accidental_breaking_changes": len([bc for bc in breaking_changes if bc.intent.value == "accidental"]),
                "files_at_risk": len(commit_analysis.files_changed),
                
                # Enhanced percentage analysis
                "risk_percentage": round(final_risk_score * 100, 1),
                "breaking_change_percentages": {
                    "by_severity": {
                        severity: {
                            "count": len([bc for bc in breaking_changes if bc.severity.value == severity]),
                            "percentage": round((len([bc for bc in breaking_changes if bc.severity.value == severity]) / max(len(breaking_changes), 1)) * 100, 1)
                        }
                        for severity in ["critical", "high", "medium", "low", "enhancement"]
                    } if breaking_changes else {},
                    "by_intent": {
                        intent: {
                            "count": len([bc for bc in breaking_changes if bc.intent.value == intent]),
                            "percentage": round((len([bc for bc in breaking_changes if bc.intent.value == intent]) / max(len(breaking_changes), 1)) * 100, 1)
                        }
                        for intent in ["accidental", "intentional", "unclear"]
                    } if breaking_changes else {},
                    "by_migration_complexity": {
                        complexity: {
                            "count": len([bc for bc in breaking_changes if bc.migration_complexity == complexity]),
                            "percentage": round((len([bc for bc in breaking_changes if bc.migration_complexity == complexity]) / max(len(breaking_changes), 1)) * 100, 1)
                        }
                        for complexity in ["very_complex", "complex", "moderate", "easy", "trivial"]
                    } if breaking_changes else {}
                },
                "confidence_metrics": {
                    "average_confidence": round(sum(bc.confidence_score for bc in breaking_changes) / max(len(breaking_changes), 1), 2) if breaking_changes else 0.0,
                    "high_confidence_changes": len([bc for bc in breaking_changes if bc.confidence_score > 0.8]),
                    "low_confidence_changes": len([bc for bc in breaking_changes if bc.confidence_score < 0.5]),
                    "confidence_distribution": {
                        "high_confidence_percentage": round((len([bc for bc in breaking_changes if bc.confidence_score > 0.8]) / max(len(breaking_changes), 1)) * 100, 1),
                        "medium_confidence_percentage": round((len([bc for bc in breaking_changes if 0.5 <= bc.confidence_score <= 0.8]) / max(len(breaking_changes), 1)) * 100, 1),
                        "low_confidence_percentage": round((len([bc for bc in breaking_changes if bc.confidence_score < 0.5]) / max(len(breaking_changes), 1)) * 100, 1)
                    } if breaking_changes else {}
                },
                "impact_analysis": {
                    "files_affected_percentage": round((len(commit_analysis.files_changed) / max(len(git_analyzer.repo.head.commit.tree.traverse()), 1)) * 100, 2),
                    "estimated_affected_users": sum(getattr(bc, 'affected_users_estimate', 0) for bc in breaking_changes),
                    "risk_factors": [
                        f"{round(final_risk_score * 100, 1)}% overall risk score",
                        f"{len([bc for bc in breaking_changes if bc.severity.value in ['critical', 'high']])} high/critical severity changes",
                        f"{len([bc for bc in breaking_changes if bc.intent.value == 'accidental'])} accidental breaking changes",
                        f"{len([bc for bc in breaking_changes if bc.migration_complexity in ['very_complex', 'complex']])} complex migrations required"
                    ] if breaking_changes else ["No breaking changes detected"]
                }
            }
        }
        
        return convert_numpy_types(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Commit analysis failed: {str(e)}")

@app.get("/api/repository/{repo_id}/risk-summary")
async def get_repository_risk_summary(
    repo_id: str,
    current_user: str = Depends(require_repository_access)
):
    """Get a quick risk percentage summary for the repository"""
    try:
        if repo_id not in active_repos:
            raise HTTPException(status_code=404, detail="Repository not found.")
        
        # Get cached dashboard or force a minimal calculation
        cached_dashboard = db_manager.get_dashboard_cache(repo_id)
        if cached_dashboard:
            dashboard_data = json.loads(cached_dashboard.dashboard_data)
            
            # Extract key percentages
            risk_summary = {
                "overall_risk_percentage": round(dashboard_data.get("overall_risk_score", 0) * 100, 1),
                "high_risk_commits_percentage": dashboard_data.get("commit_risk_percentages", {}).get("high_risk", {}).get("percentage", 0),
                "critical_risk_commits_percentage": dashboard_data.get("commit_risk_percentages", {}).get("critical_risk", {}).get("percentage", 0),
                "safe_commits_percentage": dashboard_data.get("commit_risk_percentages", {}).get("safe_commits", {}).get("percentage", 0),
                "accidental_breaking_changes_percentage": dashboard_data.get("intent_risk_percentages", {}).get("accidental_percentage", 0),
                "breaking_changes_by_severity": dashboard_data.get("breaking_changes_percentages", {}).get("by_severity", {}),
                "total_commits_analyzed": dashboard_data.get("total_commits_analyzed", 0),
                "cache_status": "cached",
                "last_updated": cached_dashboard.last_updated
            }
        else:
            # Quick calculation without full dashboard generation
            git_analyzer = active_repos[repo_id]
            recent_commits = list(git_analyzer.repo.iter_commits(max_count=50))
            analyzed_commits = db_manager.get_repository_commit_analyses(repo_id, limit=100)
            
            high_risk_count = len([a for a in analyzed_commits if a.overall_risk_score > 0.7])
            critical_risk_count = len([a for a in analyzed_commits if a.overall_risk_score > 0.9])
            total_analyzed = len(analyzed_commits)
            
            risk_summary = {
                "overall_risk_percentage": round((high_risk_count / max(total_analyzed, 1)) * 100, 1),
                "high_risk_commits_percentage": round((high_risk_count / max(total_analyzed, 1)) * 100, 1),
                "critical_risk_commits_percentage": round((critical_risk_count / max(total_analyzed, 1)) * 100, 1),
                "safe_commits_percentage": round(((total_analyzed - high_risk_count) / max(total_analyzed, 1)) * 100, 1),
                "total_commits_analyzed": total_analyzed,
                "total_commits_in_repo": len(recent_commits),
                "cache_status": "live_calculation",
                "last_updated": datetime.now().isoformat(),
                "recommendation": "Run full risk dashboard analysis for detailed percentages" if total_analyzed < 10 else None
            }
        
        return risk_summary
        
    except Exception as e:
        logger.error(f"Failed to get risk summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get risk summary: {str(e)}")

@app.get("/api/repository/{repo_id}/file/{file_path:path}/insights")
async def get_file_insights(
    repo_id: str, 
    file_path: str,
    current_user: str = Depends(require_repository_access)
):
    """Get comprehensive insights about a specific file with advanced breaking change analysis"""
    try:
        if repo_id not in active_repos:
            raise HTTPException(status_code=404, detail="Repository not found.")
        
        git_analyzer = active_repos[repo_id]
        
        # Get file history and analysis
        repo = git_analyzer.repo
        file_commits = list(repo.iter_commits(paths=file_path, max_count=50))
        
        # Analyze file risk patterns with advanced breaking change detector
        risk_history = []
        advanced_breaking_changes_history = []
        total_risk = 0.0
        enhanced_total_risk = 0.0
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "enhancement": 0}
        intent_counts = {"accidental": 0, "intentional": 0, "unclear": 0}
        
        for commit in file_commits[:20]:
            try:
                # Basic analysis
                analysis = breaking_change_detector.analyze_commit(git_analyzer, commit.hexsha)
                
                # Advanced breaking change analysis
                advanced_breaking_changes = []
                enhanced_risk_score = 0.0
                try:
                    all_advanced_changes = breaking_change_detector.analyze_commit_for_breaking_changes(commit.hexsha, git_analyzer)
                    # Filter for changes affecting this specific file
                    advanced_breaking_changes = [bc for bc in all_advanced_changes if bc.file_path == file_path]
                    
                    if advanced_breaking_changes:
                        severity_weights = {"critical": 1.0, "high": 0.8, "medium": 0.5, "low": 0.2, "enhancement": 0.1}
                        intent_weights = {"accidental": 1.5, "unclear": 1.2, "intentional": 1.0}
                        
                        total_weight = 0
                        for bc in advanced_breaking_changes:
                            severity_weight = severity_weights.get(bc.severity.value, 0.5)
                            intent_weight = intent_weights.get(bc.intent.value, 1.0)
                            migration_complexity_weight = {"very_complex": 1.5, "complex": 1.3, "moderate": 1.1, "easy": 0.9, "trivial": 0.7}.get(bc.migration_complexity, 1.0)
                            
                            change_risk = severity_weight * intent_weight * migration_complexity_weight * bc.confidence_score
                            total_weight += change_risk
                            
                            # Count by categories
                            severity_counts[bc.severity.value] += 1
                            intent_counts[bc.intent.value] += 1
                        
                        enhanced_risk_score = min(total_weight / len(advanced_breaking_changes), 1.0)
                        
                        # Store detailed breaking change info
                        for bc in advanced_breaking_changes:
                            advanced_breaking_changes_history.append({
                                "commit_hash": commit.hexsha[:8],
                                "timestamp": commit.committed_datetime.isoformat(),
                                "change_type": bc.change_type.value,
                                "severity": bc.severity.value,
                                "intent": bc.intent.value,
                                "confidence": bc.confidence_score,
                                "description": bc.description,
                                "migration_complexity": bc.migration_complexity,
                                "old_signature": bc.old_signature,
                                "new_signature": bc.new_signature
                            })
                            
                except Exception as e:
                    logger.debug(f"Advanced analysis failed for commit {commit.hexsha}: {e}")
                
                if file_path in analysis.files_changed:
                    final_risk_score = max(enhanced_risk_score, analysis.overall_risk_score)
                    
                    risk_history.append({
                        "commit_hash": commit.hexsha[:8],
                        "timestamp": commit.committed_datetime.isoformat(),
                        "message": commit.message[:100],
                        "risk_score": float(analysis.overall_risk_score),
                        "enhanced_risk_score": float(enhanced_risk_score),
                        "final_risk_score": float(final_risk_score),
                        "breaking_changes": len([bc for bc in analysis.breaking_changes 
                                               if file_path in bc.file_path]),
                        "advanced_breaking_changes": len(advanced_breaking_changes)
                    })
                    total_risk += analysis.overall_risk_score
                    enhanced_total_risk += final_risk_score
                    
            except Exception as e:
                continue
        
        # Enhanced RAG query with breaking change context
        breaking_change_context = ""
        if advanced_breaking_changes_history:
            critical_count = severity_counts["critical"]
            accidental_count = intent_counts["accidental"]
            breaking_change_context = f" [CONTEXT: {len(advanced_breaking_changes_history)} breaking changes in file history, {critical_count} critical, {accidental_count} accidental]"
        
        rag_query = f"file {file_path} changes history modifications risk{breaking_change_context}"
        rag_result = rag_system.query(repo_id, rag_query, max_results=10)
        
        # Calculate enhanced file risk metrics
        avg_risk = total_risk / len(risk_history) if risk_history else 0.0
        avg_enhanced_risk = enhanced_total_risk / len(risk_history) if risk_history else 0.0
        modification_frequency = len(file_commits)
        recent_activity = len([c for c in file_commits if (datetime.now() - c.committed_datetime.replace(tzinfo=None)).days <= 30])
        
        # Get file content analysis
        try:
            file_content = repo.head.commit.tree[file_path].data_stream.read().decode('utf-8', errors='ignore')
            file_size = len(file_content)
            lines_count = len(file_content.split('\n'))
        except:
            file_content = None
            file_size = 0
            lines_count = 0
        
        result = {
            "file_path": file_path,
            "rag_insights": {
                "response": rag_result.response,
                "confidence": rag_result.confidence,
                "sources": rag_result.sources,
                "suggestions": rag_result.suggestions
            },
            "risk_metrics": {
                "average_risk_score": float(avg_risk),
                "average_enhanced_risk_score": float(avg_enhanced_risk),
                "risk_level": "Critical" if avg_enhanced_risk > 0.9 
                           else "High" if avg_enhanced_risk > 0.7 
                           else "Medium" if avg_enhanced_risk > 0.3 
                           else "Low",
                "modification_frequency": modification_frequency,
                "recent_activity_score": recent_activity,
                "is_high_activity": modification_frequency > 10,
                "breaking_changes_count": len(advanced_breaking_changes_history)
            },
            "breaking_change_analysis": {
                "total_breaking_changes": len(advanced_breaking_changes_history),
                "by_severity": severity_counts,
                "by_intent": intent_counts,
                "recent_breaking_changes": advanced_breaking_changes_history[:10],  # Most recent 10
                "most_severe_changes": sorted(
                    advanced_breaking_changes_history,
                    key=lambda x: {"critical": 4, "high": 3, "medium": 2, "low": 1, "enhancement": 0}.get(x["severity"], 0),
                    reverse=True
                )[:5]
            },
            "history": {
                "total_commits": len(file_commits),
                "recent_commits": len([c for c in file_commits if (datetime.now() - c.committed_datetime.replace(tzinfo=None)).days <= 30]),
                "risk_history": risk_history[:10],
                "contributors": list(set(c.author.name for c in file_commits))[:5]
            },
            "file_info": {
                "size_bytes": file_size,
                "lines_count": lines_count,
                "last_modified": file_commits[0].committed_datetime.isoformat() if file_commits else None,
                "last_author": file_commits[0].author.name if file_commits else None
            }
        }
        
        return convert_numpy_types(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File insights failed: {str(e)}")

@app.post("/api/repository/{repo_id}/query/enhanced")
async def enhanced_rag_query(
    repo_id: str, 
    request: QueryRequest,
    current_user: str = Depends(require_repository_access)
):
    """Enhanced RAG query with Claude integration and advanced breaking change context"""
    try:
        if repo_id not in active_repos:
            raise HTTPException(status_code=404, detail="Repository not found.")
        
        git_analyzer = active_repos[repo_id]
        
        # Check if query is related to breaking changes or risk
        breaking_change_keywords = ['breaking', 'break', 'risk', 'danger', 'critical', 'severe', 'migration', 'compatibility', 'change']
        is_breaking_change_query = any(keyword in request.query.lower() for keyword in breaking_change_keywords)
        
        # Get comprehensive RAG response
        rag_result = rag_system.query(repo_id, request.query, max_results=10)
        
        # If query relates to breaking changes, enhance with advanced analysis
        breaking_change_context = None
        if is_breaking_change_query:
            try:
                # Get recent commits for breaking change analysis
                repo = git_analyzer.repo
                recent_commits = list(repo.iter_commits(max_count=20))
                
                # Analyze recent commits for breaking changes
                breaking_changes_found = []
                for commit in recent_commits[:10]:  # Limit to prevent timeout
                    try:
                        commit_breaking_changes = breaking_change_detector.analyze_commit_for_breaking_changes(commit.hexsha, git_analyzer)
                        if commit_breaking_changes:
                            breaking_changes_found.extend([{
                                "commit_hash": commit.hexsha[:8],
                                "change_type": bc.change_type.value,
                                "severity": bc.severity.value,
                                "intent": bc.intent.value,
                                "confidence": bc.confidence_score,
                                "description": bc.description,
                                "file_path": bc.file_path,
                                "migration_complexity": bc.migration_complexity,
                                "affected_users": bc.affected_users_estimate
                            } for bc in commit_breaking_changes])
                    except Exception as e:
                        logger.debug(f"Error analyzing commit {commit.hexsha} for breaking changes: {e}")
                        continue
                
                # Summarize breaking change context
                if breaking_changes_found:
                    severity_counts = {}
                    intent_counts = {}
                    change_type_counts = {}
                    
                    for bc in breaking_changes_found:
                        severity_counts[bc["severity"]] = severity_counts.get(bc["severity"], 0) + 1
                        intent_counts[bc["intent"]] = intent_counts.get(bc["intent"], 0) + 1
                        change_type_counts[bc["change_type"]] = change_type_counts.get(bc["change_type"], 0) + 1
                    
                    breaking_change_context = {
                        "recent_breaking_changes": breaking_changes_found[:15],  # Limit for response size
                        "summary": {
                            "total_found": len(breaking_changes_found),
                            "by_severity": severity_counts,
                            "by_intent": intent_counts,
                            "by_type": change_type_counts,
                            "critical_count": severity_counts.get("critical", 0),
                            "accidental_count": intent_counts.get("accidental", 0)
                        }
                    }
                    
                    # Enhance the RAG query with breaking change context
                    enhanced_query = f"{request.query} [CONTEXT: Found {len(breaking_changes_found)} recent breaking changes including {severity_counts.get('critical', 0)} critical and {intent_counts.get('accidental', 0)} accidental changes]"
                    rag_result = rag_system.query(repo_id, enhanced_query, max_results=12)
                    
            except Exception as e:
                logger.warning(f"Error enhancing query with breaking change context: {e}")
        
        # Enhanced response with Claude if available
        enhanced_response = None
        if claude_analyzer.available and rag_result.confidence > 0.3:
            try:
                # Include breaking change context in Claude analysis
                claude_context = rag_result.sources[:8]
                if breaking_change_context:
                    claude_context.append({
                        "type": "breaking_change_analysis",
                        "content": f"Advanced breaking change analysis found {breaking_change_context['summary']['total_found']} changes with {breaking_change_context['summary']['critical_count']} critical issues"
                    })
                
                enhanced_response = claude_analyzer.enhance_rag_response(
                    request.query, 
                    rag_result.response, 
                    claude_context
                )
            except Exception as e:
                logger.warning(f"Claude enhancement failed: {str(e)}")
        
        # Get repository summary for additional context
        repo_summary = rag_system.get_repository_summary(repo_id)
        
        # Search for related commits and files
        related_commits = rag_system.search_commits(repo_id, request.query, max_results=5)
        related_files = rag_system.search_files(repo_id, request.query, max_results=5)
        
        result = {
            "query": request.query,
            "primary_response": {
                "response": rag_result.response,
                "confidence": rag_result.confidence,
                "sources": rag_result.sources,
                "context_used": rag_result.context_used,
                "suggestions": rag_result.suggestions
            },
            "enhanced_response": enhanced_response,
            "breaking_change_context": breaking_change_context,  # New field with advanced analysis
            "related_content": {
                "commits": related_commits,
                "files": related_files
            },
            "repository_context": {
                "total_commits": repo_summary["statistics"]["total_commits_indexed"],
                "high_risk_commits": repo_summary["statistics"]["high_risk_commits"],
                "average_risk": repo_summary["statistics"]["average_risk_score"],
                "primary_languages": repo_summary["repository"]["primary_languages"]
            },
            "analysis_metadata": {
                "query_processed_at": datetime.now().isoformat(),
                "context_sources": len(rag_result.sources),
                "claude_enhanced": enhanced_response is not None,
                "breaking_change_enhanced": breaking_change_context is not None,
                "confidence_level": "High" if rag_result.confidence > 0.8 
                                  else "Medium" if rag_result.confidence > 0.5 
                                  else "Low"
            }
        }
        
        return convert_numpy_types(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhanced query failed: {str(e)}")

@app.get("/api/repository/{repo_id}/insights/semantic-drift")
async def get_semantic_drift_analysis(
    repo_id: str, 
    days: int = Query(30, description="Number of days to analyze"),
    current_user: str = Depends(require_repository_access)
):
    """Analyze semantic drift patterns in the repository"""
    try:
        if repo_id not in active_repos:
            raise HTTPException(status_code=404, detail="Repository not found.")
        
        git_analyzer = active_repos[repo_id]
        repo = git_analyzer.repo
        
        # Get commits from specified time period
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_commits = [
            commit for commit in repo.iter_commits(max_count=200)
            if commit.committed_datetime.replace(tzinfo=None) >= cutoff_date
        ]
        
        # Analyze semantic patterns with enhanced breaking change analysis
        semantic_patterns = {
            "commit_message_themes": {},
            "file_change_patterns": {},
            "risk_evolution": [],
            "breaking_change_trends": {},
            "advanced_breaking_change_trends": {},
            "developer_impact": {},
            "severity_trends": {},
            "intent_patterns": {},
            "migration_complexity_trends": {}
        }
        
        for commit in recent_commits:
            try:
                # Get basic analysis
                analysis = breaking_change_detector.analyze_commit(git_analyzer, commit.hexsha)
                
                # Get advanced breaking change analysis
                advanced_breaking_changes = []
                try:
                    advanced_breaking_changes = breaking_change_detector.analyze_commit_for_breaking_changes(commit.hexsha, git_analyzer)
                except Exception as e:
                    logger.debug(f"Advanced analysis failed for commit {commit.hexsha}: {e}")
                
                # Calculate enhanced risk score
                enhanced_risk_score = 0.0
                if advanced_breaking_changes:
                    severity_weights = {"critical": 1.0, "high": 0.8, "medium": 0.5, "low": 0.2, "enhancement": 0.1}
                    intent_weights = {"accidental": 1.5, "unclear": 1.2, "intentional": 1.0}
                    
                    total_weight = 0
                    for bc in advanced_breaking_changes:
                        severity_weight = severity_weights.get(bc.severity.value, 0.5)
                        intent_weight = intent_weights.get(bc.intent.value, 1.0)
                        migration_complexity_weight = {"very_complex": 1.5, "complex": 1.3, "moderate": 1.1, "easy": 0.9, "trivial": 0.7}.get(bc.migration_complexity, 1.0)
                        
                        change_risk = severity_weight * intent_weight * migration_complexity_weight * bc.confidence_score
                        total_weight += change_risk
                    
                    enhanced_risk_score = min(total_weight / len(advanced_breaking_changes), 1.0)
                
                # Use higher of the two risk calculations
                final_risk_score = max(enhanced_risk_score, analysis.overall_risk_score)
                
                # Track enhanced risk evolution
                semantic_patterns["risk_evolution"].append({
                    "date": commit.committed_datetime.isoformat(),
                    "timestamp": commit.committed_datetime.isoformat(),
                    "risk_score": float(final_risk_score),
                    "original_risk_score": float(analysis.overall_risk_score),
                    "enhanced_risk_score": float(enhanced_risk_score),
                    "breaking_changes": len(analysis.breaking_changes),
                    "advanced_breaking_changes": len(advanced_breaking_changes)
                })
                
                # Analyze commit message themes
                message_words = analysis.commit_message.lower().split()
                for word in message_words:
                    if len(word) > 3:  # Filter short words
                        semantic_patterns["commit_message_themes"][word] = \
                            semantic_patterns["commit_message_themes"].get(word, 0) + 1
                
                # Track file change patterns
                for file_path in analysis.files_changed:
                    file_ext = os.path.splitext(file_path)[1]
                    semantic_patterns["file_change_patterns"][file_ext] = \
                        semantic_patterns["file_change_patterns"].get(file_ext, 0) + 1
                
                # Track basic breaking change trends
                for bc in analysis.breaking_changes:
                    change_type = bc.change_type.value
                    semantic_patterns["breaking_change_trends"][change_type] = \
                        semantic_patterns["breaking_change_trends"].get(change_type, 0) + 1
                
                # Track advanced breaking change trends
                for bc in advanced_breaking_changes:
                    # By change type
                    change_type = bc.change_type.value
                    semantic_patterns["advanced_breaking_change_trends"][change_type] = \
                        semantic_patterns["advanced_breaking_change_trends"].get(change_type, 0) + 1
                    
                    # By severity
                    severity = bc.severity.value
                    semantic_patterns["severity_trends"][severity] = \
                        semantic_patterns["severity_trends"].get(severity, 0) + 1
                    
                    # By intent
                    intent = bc.intent.value
                    semantic_patterns["intent_patterns"][intent] = \
                        semantic_patterns["intent_patterns"].get(intent, 0) + 1
                    
                    # By migration complexity
                    complexity = bc.migration_complexity
                    semantic_patterns["migration_complexity_trends"][complexity] = \
                        semantic_patterns["migration_complexity_trends"].get(complexity, 0) + 1
                
                # Enhanced developer impact analysis
                author = commit.author.name
                if author not in semantic_patterns["developer_impact"]:
                    semantic_patterns["developer_impact"][author] = {
                        "commits": 0,
                        "total_risk": 0.0,
                        "enhanced_risk": 0.0,
                        "breaking_changes": 0,
                        "advanced_breaking_changes": 0,
                        "critical_changes": 0,
                        "accidental_changes": 0
                    }
                
                semantic_patterns["developer_impact"][author]["commits"] += 1
                semantic_patterns["developer_impact"][author]["total_risk"] += analysis.overall_risk_score
                semantic_patterns["developer_impact"][author]["enhanced_risk"] += final_risk_score
                semantic_patterns["developer_impact"][author]["breaking_changes"] += len(analysis.breaking_changes)
                semantic_patterns["developer_impact"][author]["advanced_breaking_changes"] += len(advanced_breaking_changes)
                semantic_patterns["developer_impact"][author]["critical_changes"] += len([bc for bc in advanced_breaking_changes if bc.severity.value == "critical"])
                semantic_patterns["developer_impact"][author]["accidental_changes"] += len([bc for bc in advanced_breaking_changes if bc.intent.value == "accidental"])
                
            except Exception as e:
                continue
        
        # Get top themes and patterns
        top_themes = sorted(semantic_patterns["commit_message_themes"].items(), 
                           key=lambda x: x[1], reverse=True)[:10]
        
        top_file_patterns = sorted(semantic_patterns["file_change_patterns"].items(), 
                                  key=lambda x: x[1], reverse=True)[:10]
        
        # Calculate enhanced developer risk scores
        for author, stats in semantic_patterns["developer_impact"].items():
            if stats["commits"] > 0:
                stats["avg_risk"] = float(stats["total_risk"] / stats["commits"])
                stats["avg_enhanced_risk"] = float(stats["enhanced_risk"] / stats["commits"])
                stats["critical_change_rate"] = float(stats["critical_changes"] / stats["commits"])
                stats["accidental_change_rate"] = float(stats["accidental_changes"] / stats["commits"])
            else:
                stats["avg_risk"] = 0.0
                stats["avg_enhanced_risk"] = 0.0
                stats["critical_change_rate"] = 0.0
                stats["accidental_change_rate"] = 0.0
        
        top_risk_developers = sorted(
            [(author, stats) for author, stats in semantic_patterns["developer_impact"].items()],
            key=lambda x: x[1]["avg_enhanced_risk"], reverse=True  # Use enhanced risk for sorting
        )[:5]
        
        # Enhanced RAG query about drift patterns with advanced context
        total_advanced_breaking_changes = sum(r["advanced_breaking_changes"] for r in semantic_patterns["risk_evolution"])
        critical_severity_count = semantic_patterns["severity_trends"].get("critical", 0)
        accidental_count = semantic_patterns["intent_patterns"].get("accidental", 0)
        
        drift_query = f"semantic drift patterns changes {days} days breaking changes trends [CONTEXT: {total_advanced_breaking_changes} advanced breaking changes detected, {critical_severity_count} critical, {accidental_count} accidental]"
        rag_result = rag_system.query(repo_id, drift_query, max_results=8)
        
        result = {
            "analysis_period": {
                "days": days,
                "commits_analyzed": len(recent_commits),
                "start_date": cutoff_date.isoformat(),
                "end_date": datetime.now().isoformat()
            },
            "semantic_drift": {
                "top_commit_themes": top_themes,
                "file_change_patterns": top_file_patterns,
                "breaking_change_trends": semantic_patterns["breaking_change_trends"],
                "advanced_breaking_change_trends": semantic_patterns["advanced_breaking_change_trends"],
                "severity_trends": semantic_patterns["severity_trends"],
                "intent_patterns": semantic_patterns["intent_patterns"],
                "migration_complexity_trends": semantic_patterns["migration_complexity_trends"],
                "risk_trend_summary": {
                    "avg_risk": float(np.mean([r["risk_score"] for r in semantic_patterns["risk_evolution"]])) if semantic_patterns["risk_evolution"] else 0,
                    "avg_enhanced_risk": float(np.mean([r["enhanced_risk_score"] for r in semantic_patterns["risk_evolution"]])) if semantic_patterns["risk_evolution"] else 0,
                    "max_risk": float(max([r["risk_score"] for r in semantic_patterns["risk_evolution"]])) if semantic_patterns["risk_evolution"] else 0,
                    "total_breaking_changes": sum(r["breaking_changes"] for r in semantic_patterns["risk_evolution"]),
                    "total_advanced_breaking_changes": total_advanced_breaking_changes,
                    "critical_changes": critical_severity_count,
                    "accidental_changes": accidental_count
                }
            },
            "developer_impact": {
                "top_risk_contributors": [
                    {
                        "author": author,
                        "avg_risk_score": stats["avg_risk"],
                        "avg_enhanced_risk_score": stats["avg_enhanced_risk"],
                        "total_commits": stats["commits"],
                        "breaking_changes": stats["breaking_changes"],
                        "advanced_breaking_changes": stats["advanced_breaking_changes"],
                        "critical_changes": stats["critical_changes"],
                        "accidental_changes": stats["accidental_changes"],
                        "critical_change_rate": stats["critical_change_rate"],
                        "accidental_change_rate": stats["accidental_change_rate"]
                    }
                    for author, stats in top_risk_developers
                ]
            },
            "rag_insights": {
                "response": rag_result.response,
                "confidence": rag_result.confidence,
                "suggestions": rag_result.suggestions
            },
            "risk_evolution": semantic_patterns["risk_evolution"][-20:]  # Last 20 data points
        }
        
        return convert_numpy_types(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Semantic drift analysis failed: {str(e)}")

# ===== FILE AND COMMIT CONTENT ENDPOINTS =====

# Caching helper functions
def detect_file_language(file_path: str) -> str:
    """Detect file language from file extension"""
    ext = os.path.splitext(file_path)[1].lower()
    language_map = {
        '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
        '.jsx': 'javascript', '.tsx': 'typescript', '.vue': 'vue',
        '.java': 'java', '.cpp': 'cpp', '.c': 'c', '.h': 'c',
        '.rs': 'rust', '.go': 'go', '.rb': 'ruby', '.php': 'php',
        '.cs': 'csharp', '.swift': 'swift', '.kt': 'kotlin',
        '.md': 'markdown', '.txt': 'text', '.json': 'json',
        '.xml': 'xml', '.html': 'html', '.css': 'css', '.scss': 'scss',
        '.sql': 'sql', '.sh': 'bash', '.bat': 'batch', '.yml': 'yaml', '.yaml': 'yaml'
    }
    return language_map.get(ext, 'text')

def get_or_cache_file_content(repo_id: str, file_path: str, commit_hash: str, git_analyzer: GitAnalyzer) -> Optional[FileContent]:
    """Get file content from cache or fetch and cache it"""
    
    # Try to get from cache first
    cached_content = db_manager.get_cached_file_content(repo_id, file_path, commit_hash)
    if cached_content:
        logger.debug(f"Retrieved file content from cache: {file_path}@{commit_hash[:8]}")
        return cached_content
    
    # Not cached, fetch from git
    try:
        repo = git_analyzer.repo
        commit = repo.commit(commit_hash)
        file_content = commit.tree[file_path].data_stream.read().decode('utf-8', errors='ignore')
        file_size = len(file_content)
        lines_count = len(file_content.split('\n'))
        language = detect_file_language(file_path)
        
        # Create cache entry
        cached_file = FileContent(
            id=f"{repo_id}_{file_path}_{commit_hash}",
            repo_id=repo_id,
            file_path=file_path,
            commit_hash=commit_hash,
            content=file_content,
            size_bytes=file_size,
            lines_count=lines_count,
            language=language,
            cached_at=datetime.now().isoformat(),
            expires_at=(datetime.now() + timedelta(days=7)).isoformat()  # Cache for 7 days
        )
        
        # Save to cache
        if db_manager.cache_file_content(cached_file):
            logger.debug(f"Cached file content: {file_path}@{commit_hash[:8]}")
        
        return cached_file
        
    except Exception as e:
        logger.error(f"Error fetching file content: {e}")
        return None

def get_or_cache_commit_files(repo_id: str, commit_hash: str, git_analyzer: GitAnalyzer, include_diff_stats: bool = False) -> Optional[CommitFileCache]:
    """Get commit files data from cache or fetch and cache it"""
    
    # Try to get from cache first
    cached_data = db_manager.get_cached_commit_files(repo_id, commit_hash)
    if cached_data and not include_diff_stats:
        logger.debug(f"Retrieved commit files from cache: {commit_hash[:8]}")
        return cached_data
    
    # Not cached or need diff stats, fetch from git
    try:
        repo = git_analyzer.repo
        commit = repo.commit(commit_hash)
        
        files_changed = []
        total_insertions = 0
        total_deletions = 0
        
        if not commit.parents:
            # First commit - all files are new
            for item in commit.tree.traverse():
                if item.type == 'blob':
                    file_info = {
                        "file_path": item.path,
                        "change_type": "A",  # Added
                        "size": item.size,
                        "language": detect_file_language(item.path)
                    }
                    
                    if include_diff_stats:
                        try:
                            content = item.data_stream.read().decode('utf-8', errors='ignore')
                            lines_count = len(content.split('\n'))
                            file_info.update({
                                "insertions": lines_count,
                                "deletions": 0,
                                "changes": lines_count
                            })
                            total_insertions += lines_count
                        except:
                            file_info.update({"insertions": 0, "deletions": 0, "changes": 0})
                    
                    files_changed.append(file_info)
        else:
            # Get diff from parent
            parent_commit = commit.parents[0]
            diffs = parent_commit.diff(commit)
            
            for diff_item in diffs:
                file_path = diff_item.a_path or diff_item.b_path
                change_type_map = {'A': 'A', 'D': 'D', 'M': 'M', 'R': 'R', 'C': 'C', 'T': 'T'}
                
                file_info = {
                    "file_path": file_path,
                    "change_type": change_type_map.get(diff_item.change_type, 'M'),
                    "size": diff_item.b_blob.size if diff_item.b_blob else 0,
                    "language": detect_file_language(file_path)
                }
                
                if diff_item.change_type == 'R' and diff_item.a_path != diff_item.b_path:
                    file_info["old_path"] = diff_item.a_path
                    file_info["new_path"] = diff_item.b_path
                
                if include_diff_stats and diff_item.diff:
                    try:
                        diff_text = diff_item.diff.decode('utf-8', errors='ignore')
                        insertions = diff_text.count('\n+') - diff_text.count('\n+++')
                        deletions = diff_text.count('\n-') - diff_text.count('\n---')
                        
                        file_info.update({
                            "insertions": max(0, insertions),
                            "deletions": max(0, deletions),
                            "changes": max(0, insertions) + max(0, deletions)
                        })
                        total_insertions += max(0, insertions)
                        total_deletions += max(0, deletions)
                    except:
                        file_info.update({"insertions": 0, "deletions": 0, "changes": 0})
                
                files_changed.append(file_info)
        
        # Create summary stats
        summary_stats = {
            "total_files": len(files_changed),
            "total_insertions": total_insertions,
            "total_deletions": total_deletions,
            "total_changes": total_insertions + total_deletions,
            "files_by_type": {
                "added": len([f for f in files_changed if f["change_type"] == "A"]),
                "modified": len([f for f in files_changed if f["change_type"] == "M"]),
                "deleted": len([f for f in files_changed if f["change_type"] == "D"]),
                "renamed": len([f for f in files_changed if f["change_type"] == "R"])
            }
        }
        
        # Create cache entry
        cached_commit = CommitFileCache(
            id=f"{repo_id}_{commit_hash}",
            repo_id=repo_id,
            commit_hash=commit_hash,
            files_data=json.dumps(files_changed),
            summary_stats=json.dumps(summary_stats),
            cached_at=datetime.now().isoformat()
        )
        
        # Save to cache (only if not including diff stats to avoid large cache entries)
        if not include_diff_stats and db_manager.cache_commit_files(cached_commit):
            logger.debug(f"Cached commit files: {commit_hash[:8]}")
        
        return cached_commit
        
    except Exception as e:
        logger.error(f"Error fetching commit files: {e}")
        return None

@app.get("/api/repository/{repo_id}/file/{file_path:path}")
async def get_file_content(
    repo_id: str, 
    file_path: str,
    commit_hash: str = Query(None, description="Specific commit hash, defaults to HEAD"),
    show_diff: bool = Query(False, description="Show git diff instead of full content"),
    current_user: str = Depends(require_repository_access)
):
    """Get file content at a specific commit or show diff"""
    try:
        if repo_id not in active_repos:
            raise HTTPException(status_code=404, detail="Repository not found")
        
        git_analyzer = active_repos[repo_id]
        repo = git_analyzer.repo
        
        # Use HEAD if no commit specified
        if not commit_hash:
            commit_hash = repo.head.commit.hexsha
        
        # Get the commit
        try:
            commit = repo.commit(commit_hash)
        except:
            raise HTTPException(status_code=404, detail="Commit not found")
        
        if show_diff:
            # Show git diff for this file
            if not commit.parents:
                # First commit - show entire file as added
                try:
                    # Try to get from cache first
                    cached_content = get_or_cache_file_content(repo_id, file_path, commit_hash, git_analyzer)
                    if cached_content:
                        file_content = cached_content.content
                    else:
                        file_content = commit.tree[file_path].data_stream.read().decode('utf-8', errors='ignore')
                    
                    diff_content = f"--- /dev/null\n+++ b/{file_path}\n"
                    for i, line in enumerate(file_content.split('\n'), 1):
                        diff_content += f"+{line}\n"
                    
                    return {
                        "repo_id": repo_id,
                        "file_path": file_path,
                        "commit_hash": commit_hash,
                        "type": "diff",
                        "content": diff_content,
                        "message": "First commit - entire file added",
                        "cached": cached_content is not None
                    }
                except:
                    raise HTTPException(status_code=404, detail="File not found in commit")
            else:
                # Get diff between parent and current commit
                parent_commit = commit.parents[0]
                try:
                    diffs = parent_commit.diff(commit, paths=[file_path], create_patch=True)
                    if not diffs:
                        # Check if file exists in both commits
                        base_has_file = file_path in [item.path for item in parent_commit.tree.traverse() if item.type == 'blob']
                        head_has_file = file_path in [item.path for item in commit.tree.traverse() if item.type == 'blob']
                        
                        if not base_has_file and not head_has_file:
                            raise HTTPException(status_code=404, detail="File not found in either commit")
                        elif not base_has_file:
                            # File was added
                            file_content = commit.tree[file_path].data_stream.read().decode('utf-8', errors='ignore')
                            diff_content = f"--- /dev/null\n+++ b/{file_path}\n"
                            for line in file_content.split('\n'):
                                diff_content += f"+{line}\n"
                            
                            return {
                                "repo_id": repo_id,
                                "base_commit": parent_commit.hexsha,
                                "head_commit": commit.hexsha,
                                "file_path": file_path,
                                "change_type": "added",
                                "diff": diff_content
                            }
                        elif not head_has_file:
                            # File was deleted
                            file_content = parent_commit.tree[file_path].data_stream.read().decode('utf-8', errors='ignore')
                            diff_content = f"--- a/{file_path}\n+++ /dev/null\n"
                            for line in file_content.split('\n'):
                                diff_content += f"-{line}\n"
                            
                            return {
                                "repo_id": repo_id,
                                "base_commit": parent_commit.hexsha,
                                "head_commit": commit.hexsha,
                                "file_path": file_path,
                                "change_type": "deleted",
                                "diff": diff_content
                            }
                        else:
                            return {
                                "repo_id": repo_id,
                                "base_commit": parent_commit.hexsha,
                                "head_commit": commit.hexsha,
                                "file_path": file_path,
                                "change_type": "unchanged",
                                "diff": "No changes detected",
                                "message": "File exists in both commits but no differences found"
                            }
                
                    diff_item = diffs[0]
                    change_type_map = {
                        'A': 'added',
                        'D': 'deleted',
                        'M': 'modified',
                        'R': 'renamed',
                        'C': 'copied'
                    }
                    
                    # Calculate diff stats
                    diff_stats = {"insertions": 0, "deletions": 0}
                    if diff_item.diff:
                        diff_text = diff_item.diff.decode('utf-8', errors='ignore')
                        insertions = diff_text.count('\n+') - diff_text.count('\n+++')
                        deletions = diff_text.count('\n-') - diff_text.count('\n---')
                        diff_stats = {
                            "insertions": max(0, insertions),
                            "deletions": max(0, deletions),
                            "changes": max(0, insertions) + max(0, deletions)
                        }
                    
                    return {
                        "repo_id": repo_id,
                        "base_commit": parent_commit.hexsha,
                        "head_commit": commit.hexsha,
                        "file_path": file_path,
                        "change_type": change_type_map.get(diff_item.change_type, 'modified'),
                        "diff": str(diff_item),
                        "diff_stats": diff_stats,
                        "commit_info": {
                            "base": {
                                "hash": parent_commit.hexsha,
                                "message": parent_commit.message.strip(),
                                "author": parent_commit.author.name,
                                "date": parent_commit.committed_datetime.isoformat()
                            },
                            "head": {
                                "hash": commit.hexsha,
                                "message": commit.message.strip(),
                                "author": commit.author.name,
                                "date": commit.committed_datetime.isoformat()
                            }
                        }
                    }
                except Exception as e:
                    raise HTTPException(status_code=404, detail=f"Error generating diff: {str(e)}")
        else:
            # Show full file content using cache
            cached_content = get_or_cache_file_content(repo_id, file_path, commit_hash, git_analyzer)
            if not cached_content:
                raise HTTPException(status_code=404, detail="File not found in commit")
            
            return {
                "repo_id": repo_id,
                "file_path": file_path,
                "commit_hash": commit_hash,
                "type": "content",
                "content": cached_content.content,
                "language": cached_content.language,
                "size_bytes": cached_content.size_bytes,
                "lines": cached_content.lines_count,
                "cached_at": cached_content.cached_at,
                "from_cache": True,
                "commit_info": {
                    "message": commit.message.strip(),
                    "author": commit.author.name,
                    "date": commit.committed_datetime.isoformat(),
                    "hash": commit.hexsha
                }
            }
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting file content: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get file content: {str(e)}")

@app.get("/api/repository/{repo_id}/commit/{commit_hash}/files")
async def get_commit_files(
    repo_id: str,
    commit_hash: str,
    include_diff_stats: bool = Query(False, description="Include diff statistics for each file"),
    show_all_files: bool = Query(True, description="Show all files in commit, not just changed files"),
    current_user: str = Depends(require_repository_access)
):
    """Get list of all files in a specific commit or just the files that were changed"""
    try:
        if repo_id not in active_repos:
            raise HTTPException(status_code=404, detail="Repository not found")
        
        git_analyzer = active_repos[repo_id]
        repo = git_analyzer.repo
        
        # Get commit info
        try:
            commit = repo.commit(commit_hash)
        except:
            raise HTTPException(status_code=404, detail="Commit not found")
        
        if show_all_files:
            # Get all files in the commit tree
            all_files = []
            for item in commit.tree.traverse():
                if item.type == 'blob':  # Only include files, not directories
                    file_info = {
                        "file_path": item.path,
                        "change_type": "existing",
                        "size": item.size,
                        "language": detect_file_language(item.path),
                        "mode": oct(item.mode)
                    }
                    
                    # If diff stats are requested, calculate them for changed files
                    if include_diff_stats and commit.parents:
                        try:
                            parent_commit = commit.parents[0]
                            diffs = parent_commit.diff(commit, paths=[item.path])
                            if diffs:
                                diff_item = diffs[0]
                                change_type_map = {'A': 'added', 'D': 'deleted', 'M': 'modified', 'R': 'renamed', 'C': 'copied'}
                                file_info["change_type"] = change_type_map.get(diff_item.change_type, 'existing')
                                
                                if diff_item.diff:
                                    diff_text = diff_item.diff.decode('utf-8', errors='ignore')
                                    insertions = diff_text.count('\n+') - diff_text.count('\n+++')
                                    deletions = diff_text.count('\n-') - diff_text.count('\n---')
                                    
                                    file_info["diff_stats"] = {
                                        "insertions": max(0, insertions),
                                        "deletions": max(0, deletions),
                                        "changes": max(0, insertions) + max(0, deletions)
                                    }
                        except Exception as e:
                            logger.debug(f"Error getting diff stats for {item.path}: {e}")
                    
                    all_files.append(file_info)
            
            # Sort files by path for consistent ordering
            all_files.sort(key=lambda f: f["file_path"])
            
            # Calculate summary stats
            summary_stats = {
                "total_files": len(all_files),
                "files_by_language": {},
                "total_size": sum(f["size"] for f in all_files)
            }
            
            # Group by language
            for file_info in all_files:
                lang = file_info["language"]
                if lang not in summary_stats["files_by_language"]:
                    summary_stats["files_by_language"][lang] = 0
                summary_stats["files_by_language"][lang] += 1
            
            if include_diff_stats:
                summary_stats["files_by_change_type"] = {
                    "added": len([f for f in all_files if f["change_type"] == "added"]),
                    "modified": len([f for f in all_files if f["change_type"] == "modified"]),
                    "deleted": len([f for f in all_files if f["change_type"] == "deleted"]),
                    "renamed": len([f for f in all_files if f["change_type"] == "renamed"]),
                    "existing": len([f for f in all_files if f["change_type"] == "existing"])
                }
            
            return {
                "repo_id": repo_id,
                "commit_hash": commit_hash,
                "commit_info": {
                    "message": commit.message.strip(),
                    "author": commit.author.name,
                    "email": commit.author.email,
                    "date": commit.committed_datetime.isoformat(),
                    "hash": commit.hexsha
                },
                "files": all_files,
                "summary": summary_stats,
                "type": "all_files",
                "from_cache": False
            }
        else:
            # Use existing caching function to get only changed files
            cached_commit = get_or_cache_commit_files(repo_id, commit_hash, git_analyzer, include_diff_stats)
            if not cached_commit:
                raise HTTPException(status_code=404, detail="Commit not found or error processing commit")
            
            # Parse cached data
            files_changed = json.loads(cached_commit.files_data)
            summary_stats = json.loads(cached_commit.summary_stats)
            
            return {
                "repo_id": repo_id,
                "commit_hash": commit_hash,
                "commit_info": {
                    "message": commit.message.strip(),
                    "author": commit.author.name,
                    "email": commit.author.email,
                    "date": commit.committed_datetime.isoformat(),
                    "hash": commit.hexsha
                },
                "files": files_changed,
                "summary": summary_stats,
                "type": "changed_files",
                "cached_at": cached_commit.cached_at,
                "from_cache": True
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting commit files: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get commit files: {str(e)}")

@app.get("/api/repository/{repo_id}/file-history/{file_path:path}")
async def get_file_history(
    repo_id: str,
    file_path: str,
    limit: int = Query(50, description="Maximum number of commits to return"),
    show_diffs: bool = Query(False, description="Include diff content for each commit"),
    current_user: str = Depends(require_repository_access)
):
    """Get commit history for a specific file"""
    try:
        if repo_id not in active_repos:
            raise HTTPException(status_code=404, detail="Repository not found")
        
        git_analyzer = active_repos[repo_id]
        repo = git_analyzer.repo
        
        # Get commits that touched this file
        file_commits = list(repo.iter_commits(paths=file_path, max_count=limit))
        
        if not file_commits:
            raise HTTPException(status_code=404, detail="File not found or no commit history")
        
        history = []
        
        for commit in file_commits:
            commit_info = {
                "commit_hash": commit.hexsha,
                "short_hash": commit.hexsha[:8],
                "message": commit.message.strip(),
                "author": commit.author.name,
                "email": commit.author.email,
                "date": commit.committed_datetime.isoformat(),
                "change_type": "M"  # Default to modified
            }
            
            # Determine change type and get diff if requested
            if commit.parents:
                parent_commit = commit.parents[0]
                try:
                    diffs = parent_commit.diff(commit, paths=[file_path])
                    if diffs:
                        diff_item = diffs[0]
                        change_type_map = {
                            'A': 'added',
                            'D': 'deleted',
                            'M': 'modified',
                            'R': 'renamed',
                            'C': 'copied'
                        }
                        commit_info["change_type"] = change_type_map.get(diff_item.change_type, 'modified')
                        
                        if show_diffs:
                            try:
                                diff_content = str(diff_item)
                                commit_info["diff"] = diff_content
                                
                                # Calculate diff stats
                                if diff_item.diff:
                                    diff_text = diff_item.diff.decode('utf-8', errors='ignore')
                                    insertions = diff_text.count('\n+') - diff_text.count('\n+++')
                                    deletions = diff_text.count('\n-') - diff_text.count('\n---')
                                    
                                    commit_info["diff_stats"] = {
                                        "insertions": max(0, insertions),
                                        "deletions": max(0, deletions),
                                        "changes": max(0, insertions) + max(0, deletions)
                                    }
                            except Exception as e:
                                logger.debug(f"Error getting diff for {commit.hexsha}: {e}")
                except Exception as e:
                    logger.debug(f"Error processing commit {commit.hexsha}: {e}")
            else:
                # First commit
                commit_info["change_type"] = "added"
                if show_diffs:
                    try:
                        file_content = commit.tree[file_path].data_stream.read().decode('utf-8', errors='ignore')
                        lines = file_content.split('\n')
                        diff_content = f"--- /dev/null\n+++ b/{file_path}\n"
                        for line in lines:
                            diff_content += f"+{line}\n"
                        commit_info["diff"] = diff_content
                        commit_info["diff_stats"] = {
                            "insertions": len(lines),
                            "deletions": 0,
                            "changes": len(lines)
                        }
                    except Exception as e:
                        logger.debug(f"Error getting initial content for {commit.hexsha}: {e}")
            
            history.append(commit_info)
        
        return {
            "repo_id": repo_id,
            "file_path": file_path,
            "total_commits": len(history),
            "history": history,
            "file_info": {
                "first_commit": history[-1]["commit_hash"] if history else None,
                "last_modified": history[0]["commit_hash"] if history else None,
                "total_authors": len(set(c["author"] for c in history))
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting file history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get file history: {str(e)}")

@app.get("/api/repository/{repo_id}/compare/{base_commit}...{head_commit}")
async def compare_commits(
    repo_id: str,
    base_commit: str,
    head_commit: str,
    file_path: str = Query(None, description="Specific file to compare (optional)"),
    context_lines: int = Query(3, description="Number of context lines in diff"),
    current_user: str = Depends(require_repository_access)
):
    """Compare two commits or compare a specific file between commits"""
    try:
        if repo_id not in active_repos:
            raise HTTPException(status_code=404, detail="Repository not found")
        
        git_analyzer = active_repos[repo_id]
        repo = git_analyzer.repo
        
        # Get commits
        try:
            base = repo.commit(base_commit)
            head = repo.commit(head_commit)
        except:
            raise HTTPException(status_code=404, detail="One or both commits not found")
        
        if file_path:
            # Compare specific file
            try:
                diffs = base.diff(head, paths=[file_path], create_patch=True)
                if not diffs:
                    # Check if file exists in both commits
                    base_has_file = file_path in [item.path for item in base.tree.traverse() if item.type == 'blob']
                    head_has_file = file_path in [item.path for item in head.tree.traverse() if item.type == 'blob']
                    
                    if not base_has_file and not head_has_file:
                        raise HTTPException(status_code=404, detail="File not found in either commit")
                    elif not base_has_file:
                        # File was added
                        file_content = head.tree[file_path].data_stream.read().decode('utf-8', errors='ignore')
                        diff_content = f"--- /dev/null\n+++ b/{file_path}\n"
                        for line in file_content.split('\n'):
                            diff_content += f"+{line}\n"
                        
                        return {
                            "repo_id": repo_id,
                            "base_commit": base_commit.hexsha,
                            "head_commit": head_commit.hexsha,
                            "file_path": file_path,
                            "change_type": "added",
                            "diff": diff_content
                        }
                    elif not head_has_file:
                        # File was deleted
                        file_content = base.tree[file_path].data_stream.read().decode('utf-8', errors='ignore')
                        diff_content = f"--- a/{file_path}\n+++ /dev/null\n"
                        for line in file_content.split('\n'):
                            diff_content += f"-{line}\n"
                        
                        return {
                            "repo_id": repo_id,
                            "base_commit": base_commit.hexsha,
                            "head_commit": head_commit.hexsha,
                            "file_path": file_path,
                            "change_type": "deleted",
                            "diff": diff_content
                        }
                    else:
                        return {
                            "repo_id": repo_id,
                            "base_commit": base_commit.hexsha,
                            "head_commit": head_commit.hexsha,
                            "file_path": file_path,
                            "change_type": "unchanged",
                            "diff": "No changes detected",
                            "message": "File exists in both commits but no differences found"
                        }
                
                diff_item = diffs[0]
                change_type_map = {
                    'A': 'added',
                    'D': 'deleted',
                    'M': 'modified',
                    'R': 'renamed',
                    'C': 'copied'
                }
                
                # Calculate diff stats
                diff_stats = {"insertions": 0, "deletions": 0}
                if diff_item.diff:
                    diff_text = diff_item.diff.decode('utf-8', errors='ignore')
                    insertions = diff_text.count('\n+') - diff_text.count('\n+++')
                    deletions = diff_text.count('\n-') - diff_text.count('\n---')
                    diff_stats = {
                        "insertions": max(0, insertions),
                        "deletions": max(0, deletions),
                        "changes": max(0, insertions) + max(0, deletions)
                    }
                
                return {
                    "repo_id": repo_id,
                    "base_commit": base_commit.hexsha,
                    "head_commit": head_commit.hexsha,
                    "file_path": file_path,
                    "change_type": change_type_map.get(diff_item.change_type, 'modified'),
                    "diff": str(diff_item),
                    "diff_stats": diff_stats,
                    "commit_info": {
                        "base": {
                            "hash": base.hexsha,
                            "message": base.message.strip(),
                            "author": base.author.name,
                            "date": base.committed_datetime.isoformat()
                        },
                        "head": {
                            "hash": head.hexsha,
                            "message": head.message.strip(),
                            "author": head.author.name,
                            "date": head.committed_datetime.isoformat()
                        }
                    }
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error comparing file: {str(e)}")
        else:
            # Compare all files between commits
            diffs = base.diff(head)
            
            files_changed = []
            total_insertions = 0
            total_deletions = 0
            
            for diff_item in diffs:
                file_path = diff_item.a_path or diff_item.b_path
                change_type_map = {
                    'A': 'added',
                    'D': 'deleted',
                    'M': 'modified',
                    'R': 'renamed',
                    'C': 'copied'
                }
                
                file_info = {
                    "file_path": file_path,
                    "change_type": change_type_map.get(diff_item.change_type, 'modified')
                }
                
                # Calculate diff stats
                if diff_item.diff:
                    try:
                        diff_text = diff_item.diff.decode('utf-8', errors='ignore')
                        insertions = diff_text.count('\n+') - diff_text.count('\n+++')
                        deletions = diff_text.count('\n-') - diff_text.count('\n---')
                        
                        file_info["diff_stats"] = {
                            "insertions": max(0, insertions),
                            "deletions": max(0, deletions),
                            "changes": max(0, insertions) + max(0, deletions)
                        }
                        
                        total_insertions += max(0, insertions)
                        total_deletions += max(0, deletions)
                    except:
                        file_info["diff_stats"] = {"insertions": 0, "deletions": 0, "changes": 0}
                else:
                    file_info["diff_stats"] = {"insertions": 0, "deletions": 0, "changes": 0}
                
                files_changed.append(file_info)
            
            return {
                "repo_id": repo_id,
                "base_commit": base_commit,
                "head_commit": head_commit,
                "files_changed": files_changed,
                "summary": {
                    "total_files": len(files_changed),
                    "total_insertions": total_insertions,
                    "total_deletions": total_deletions,
                    "total_changes": total_insertions + total_deletions,
                    "files_by_type": {
                        "added": len([f for f in files_changed if f["change_type"] == "added"]),
                        "modified": len([f for f in files_changed if f["change_type"] == "modified"]),
                        "deleted": len([f for f in files_changed if f["change_type"] == "deleted"]),
                        "renamed": len([f for f in files_changed if f["change_type"] == "renamed"])
                    }
                },
                "commit_info": {
                    "base": {
                        "hash": base.hexsha,
                        "message": base.message.strip(),
                        "author": base.author.name,
                        "date": base.committed_datetime.isoformat()
                    },
                    "head": {
                        "hash": head.hexsha,
                        "message": head.message.strip(),
                        "author": head.author.name,
                        "date": head.committed_datetime.isoformat()
                    }
                }
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing commits: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to compare commits: {str(e)}")

# ===== BACKGROUND TASK FUNCTIONS =====

async def index_repository_for_rag(repo_id: str, repo_url: str, git_analyzer: GitAnalyzer):
    """Background task to index repository in RAG system"""
    try:
        logger.info(f"Starting RAG indexing for repository {repo_id}")
        rag_system.index_repository(git_analyzer, repo_id, repo_url)
        logger.info(f"Successfully indexed repository {repo_id} in RAG system")
    except Exception as e:
        logger.error(f"Failed to index repository {repo_id} in RAG system: {e}")

async def store_repository_commits(repo_id: str, git_analyzer: GitAnalyzer):
    """Background task to store repository commits in database"""
    try:
        logger.info(f"Starting commit storage for repository {repo_id}")
        
        # Get repository stats
        stats = git_analyzer.analyze_repository_stats()
        
        # Update repository with stats
        repo = db_manager.get_repository(repo_id)
        if repo:
            repo.total_commits = stats.get("commits", 0)
            repo.total_files = stats.get("files", 0)
            repo.primary_language = list(stats.get("languages", {}).keys())[0] if stats.get("languages") else ""
            repo.updated_at = datetime.now().isoformat()
            db_manager.save_repository(repo)
        
        # Store commits
        commits = git_analyzer.get_commits(limit=500)  # Store up to 500 recent commits
        commit_records = []
        
        for commit in commits:
            try:
                commit_record = Commit(
                    id=f"{repo_id}_{commit.hash}",
                    repo_id=repo_id,
                    hash=commit.hash,
                    message=commit.message,
                    author=commit.author,
                    timestamp=commit.timestamp.isoformat() if hasattr(commit.timestamp, 'isoformat') else str(commit.timestamp),
                    files_changed=json.dumps(commit.files_changed) if hasattr(commit, 'files_changed') else "[]",
                    additions=getattr(commit, 'additions', 0),
                    deletions=getattr(commit, 'deletions', 0),
                    risk_score=0.5,  # Default risk score, will be calculated later
                    breaking_changes="[]"
                )
                commit_records.append(commit_record)
            except Exception as e:
                logger.debug(f"Error processing commit {commit.hash}: {e}")
                continue
        
        # Save commits in batches
        batch_size = 50
        for i in range(0, len(commit_records), batch_size):
            batch = commit_records[i:i + batch_size]
            if not db_manager.save_commits(batch):
                logger.warning(f"Failed to save commit batch {i//batch_size + 1}")
        
        logger.info(f"Successfully stored {len(commit_records)} commits for repository {repo_id}")
        
    except Exception as e:
        logger.error(f"Failed to store commits for repository {repo_id}: {e}")

def normalize_repository_url(repo_url: str) -> str:
    """Normalize repository URL to handle different formats"""
    url = repo_url.strip().lower()
    
    # Remove trailing slash
    if url.endswith('/'):
        url = url[:-1]
    
    # Convert http to https
    if url.startswith('http://'):
        url = url.replace('http://', 'https://')
    
    # Add .git suffix if missing for GitHub URLs
    if 'github.com' in url and not url.endswith('.git'):
        url += '.git'
    
    # Remove any auth tokens from URL for comparison
    if '@' in url:
        # Remove tokens like https://token@github.com/...
        parts = url.split('@')
        if len(parts) == 2 and '://' in parts[0]:
            protocol = parts[0].split('://')[0]
            url = f"{protocol}://{parts[1]}"
    
    return url

def serialize_breaking_change(bc) -> Dict[str, Any]:
    """Serialize a BreakingChange object to a JSON-compatible dictionary"""
    data = asdict(bc)
    
    # Convert enums to their string values
    data['change_type'] = bc.change_type.value
    data['severity'] = bc.severity.value  
    data['intent'] = bc.intent.value
    
    # Convert datetime to ISO string
    if hasattr(bc, 'detected_at') and bc.detected_at:
        data['detected_at'] = bc.detected_at.isoformat()
    
    return data

# ===== ADVANCED BREAKING CHANGE DETECTION ENDPOINT =====

@app.post("/api/repository/{repo_id}/analyze-breaking-changes")
async def analyze_breaking_changes(
    repo_id: str,
    request: dict,
    current_user: str = Depends(require_repository_access)
):
    """
    Analyze commits for breaking changes using the advanced breaking change detector.
    
    Request body can contain:
    - commit_hashes: List of specific commits to analyze
    - since_commit: Analyze all commits since this commit hash
    - days_back: Analyze commits from the last N days (default: 7)
    - include_context: Include detailed context and file analysis (default: true)
    - ai_analysis: Use AI-powered analysis for deeper insights (default: true)
    """
    try:
        if repo_id not in active_repos:
            raise HTTPException(status_code=404, detail="Repository not found")
        
        git_analyzer = active_repos[repo_id]
        
        # Initialize the advanced breaking change detector
        from src.breaking_change_detector import BreakingChangeDetector
        detector = BreakingChangeDetector(git_analyzer, claude_analyzer)
        
        # Parse request parameters
        commit_hashes = request.get('commit_hashes', [])
        since_commit = request.get('since_commit')
        days_back = request.get('days_back', 7)
        include_context = request.get('include_context', True)
        ai_analysis = request.get('ai_analysis', True)
        
        # Get commits to analyze
        repo = git_analyzer.repo
        commits_to_analyze = []
        
        if commit_hashes:
            # Analyze specific commits
            for commit_hash in commit_hashes:
                try:
                    commit = repo.commit(commit_hash)
                    commits_to_analyze.append(commit)
                except Exception as e:
                    logger.warning(f"Could not find commit {commit_hash}: {e}")
        
        elif since_commit:
            # Analyze commits since a specific commit
            try:
                since = repo.commit(since_commit)
                commits_to_analyze = list(repo.iter_commits(
                    rev=f"{since_commit}..HEAD",
                    max_count=100  # Limit to prevent excessive analysis
                ))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid since_commit: {e}")
        
        else:
            # Analyze commits from the last N days
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            commits_to_analyze = []
            for commit in repo.iter_commits(max_count=200):
                if commit.committed_datetime.replace(tzinfo=None) >= cutoff_date:
                    commits_to_analyze.append(commit)
                else:
                    break
        
        if not commits_to_analyze:
            return {
                "repo_id": repo_id,
                "analysis_type": "breaking_changes",
                "commits_analyzed": 0,
                "breaking_changes": [],
                "summary": {
                    "total_changes": 0,
                    "critical_changes": 0,
                    "high_impact_changes": 0,
                    "accidental_changes": 0
                },
                "recommendations": {
                    "immediate_actions": ["No commits found to analyze"],
                    "migration_strategy": ["No migration needed"],
                    "communication_plan": ["No communication needed"],
                    "testing_recommendations": ["Standard testing sufficient"]
                }
            }
        
        # Analyze each commit for breaking changes
        all_breaking_changes = []
        commit_analyses = []
        
        logger.info(f"Analyzing {len(commits_to_analyze)} commits for breaking changes")
        
        for commit in commits_to_analyze:
            try:
                # Get commit diff
                if commit.parents:
                    parent = commit.parents[0]
                    diff_index = parent.diff(commit)
                else:
                    # First commit - analyze all files as additions
                    diff_index = commit.diff(None)
                
                # Analyze the commit using the enhanced breaking change detector
                breaking_changes = detector.analyze_commit_for_breaking_changes(commit.hexsha, git_analyzer)
                
                # Create a compatible analysis result structure
                analysis_result = type('AnalysisResult', (), {
                    'breaking_changes': breaking_changes,
                    'files_analyzed': len(set(bc.file_path for bc in breaking_changes if bc.file_path)),
                    'analysis_duration': 0.0  # Not tracked by enhanced detector
                })()
                
                if analysis_result.breaking_changes:
                    all_breaking_changes.extend(analysis_result.breaking_changes)
                
                commit_analyses.append({
                    "commit_hash": commit.hexsha,
                    "short_hash": commit.hexsha[:8],
                    "message": commit.message.strip(),
                    "author": commit.author.name,
                    "date": commit.committed_datetime.isoformat(),
                    "breaking_changes_count": len(analysis_result.breaking_changes),
                    "files_analyzed": analysis_result.files_analyzed,
                    "analysis_duration": analysis_result.analysis_duration,
                    "breaking_changes": [serialize_breaking_change(bc) for bc in analysis_result.breaking_changes] if include_context else []
                })
                
            except Exception as e:
                logger.error(f"Error analyzing commit {commit.hexsha}: {e}")
                commit_analyses.append({
                    "commit_hash": commit.hexsha,
                    "short_hash": commit.hexsha[:8],
                    "message": commit.message.strip(),
                    "author": commit.author.name,
                    "date": commit.committed_datetime.isoformat(),
                    "breaking_changes_count": 0,
                    "files_analyzed": 0,
                    "error": str(e)
                })
        
        # Generate summary statistics
        critical_changes = [bc for bc in all_breaking_changes if bc.severity.value == "critical"]
        high_impact_changes = [bc for bc in all_breaking_changes if bc.severity.value in ["critical", "high"]]
        accidental_changes = [bc for bc in all_breaking_changes if bc.intent.value == "accidental"]
        
        # Generate recommendations
        recommendations = {
            "immediate_actions": _get_immediate_actions(all_breaking_changes),
            "migration_strategy": _get_migration_strategy(all_breaking_changes),
            "communication_plan": _get_communication_plan(all_breaking_changes),
            "testing_recommendations": _get_testing_recommendations(all_breaking_changes)
        }
        
        # Create response
        response = {
            "repo_id": repo_id,
            "analysis_type": "breaking_changes",
            "analysis_params": {
                "commit_hashes": commit_hashes if commit_hashes else None,
                "since_commit": since_commit,
                "days_back": days_back if not commit_hashes and not since_commit else None,
                "include_context": include_context,
                "ai_analysis": ai_analysis
            },
            "commits_analyzed": len(commits_to_analyze),
            "commits_details": commit_analyses,
            "breaking_changes": [serialize_breaking_change(bc) for bc in all_breaking_changes] if include_context else [],
            "summary": {
                "total_changes": len(all_breaking_changes),
                "critical_changes": len(critical_changes),
                "high_impact_changes": len(high_impact_changes),
                "accidental_changes": len(accidental_changes),
                "change_types": _get_change_type_distribution(all_breaking_changes),
                "affected_files": len(set(bc.file_path for bc in all_breaking_changes if bc.file_path))
            },
            "recommendations": recommendations,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return convert_numpy_types(response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing breaking changes: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze breaking changes: {str(e)}")

def _get_immediate_actions(breaking_changes) -> List[str]:
    """Get immediate actions based on breaking changes"""
    actions = []
    
    critical = [c for c in breaking_changes if c.severity.value == "critical"]
    if critical:
        actions.append("🚨 URGENT: Review critical breaking changes immediately")
        actions.append(f"📋 Analyze impact of {len(critical)} critical changes")
    
    accidental = [c for c in breaking_changes if c.intent.value == "accidental"]
    if accidental:
        actions.append(f"🔍 Investigate {len(accidental)} accidental breaking changes")
        actions.append("🛠️ Consider reverting or fixing accidental changes")
    
    api_changes = [c for c in breaking_changes if c.change_type.value == "api_signature_change"]
    if api_changes:
        actions.append("📢 Notify API consumers of upcoming changes")
        actions.append("📝 Update API documentation")
    
    if not actions:
        actions.append("✅ No immediate actions required")
    
    return actions

def _get_migration_strategy(breaking_changes) -> List[str]:
    """Get migration strategy recommendations"""
    strategies = []
    
    if not breaking_changes:
        return ["No migration required"]
    
    # Group by migration complexity
    complex_changes = [c for c in breaking_changes if c.migration_complexity in ["complex", "very_complex"]]
    moderate_changes = [c for c in breaking_changes if c.migration_complexity == "moderate"]
    
    if complex_changes:
        strategies.append("🎯 Phase 1: Address complex migrations first")
        strategies.append("📋 Create detailed migration guides for complex changes")
        strategies.append("🧪 Implement comprehensive testing for complex migrations")
    
    if moderate_changes:
        strategies.append("⚡ Phase 2: Handle moderate complexity migrations")
        strategies.append("🔄 Provide backward compatibility where possible")
    
    strategies.append("📚 Create migration documentation and examples")
    strategies.append("🤝 Provide support channels for migration assistance")
    
    return strategies

def _get_communication_plan(breaking_changes) -> List[str]:
    """Get communication plan for breaking changes"""
    plan = []
    
    if not breaking_changes:
        return ["No communication required"]
    
    high_impact = [c for c in breaking_changes if c.severity.value in ["critical", "high"]]
    
    if high_impact:
        plan.append("📢 Send advance notice to all affected users")
        plan.append("📧 Email notification 2 weeks before deployment")
        plan.append("📝 Publish detailed changelog with migration instructions")
    
    plan.append("🔗 Update documentation and examples")
    plan.append("💬 Monitor support channels for migration issues")
    plan.append("📊 Track adoption and migration progress")
    
    return plan

def _get_testing_recommendations(breaking_changes) -> List[str]:
    """Get testing recommendations for breaking changes"""
    recommendations = []
    
    if not breaking_changes:
        return ["Standard testing sufficient"]
    
    api_changes = [c for c in breaking_changes if c.change_type.value == "api_signature_change"]
    behavioral_changes = [c for c in breaking_changes if c.change_type.value == "behavioral_change"]
    
    if api_changes:
        recommendations.append("🧪 Run comprehensive API integration tests")
        recommendations.append("📝 Update contract tests for API changes")
    
    if behavioral_changes:
        recommendations.append("🔍 Perform thorough behavioral testing")
        recommendations.append("📊 Run performance regression tests")
    
    recommendations.append("🎯 Test all identified migration scenarios")
    recommendations.append("🔄 Verify backward compatibility where applicable")
    recommendations.append("🚀 Conduct staging environment validation")
    
    return recommendations

def _get_change_type_distribution(breaking_changes):
    """Get distribution of change types"""
    from collections import defaultdict
    distribution = defaultdict(int)
    
    for change in breaking_changes:
        distribution[change.change_type.value] += 1
    
    return dict(distribution)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
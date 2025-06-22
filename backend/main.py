from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any, Optional, Union
import tempfile
import shutil
import os
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
from src.breaking_change_detector import BreakingChangeDetector, CommitBreakingChangeAnalysis
from src.rag_system import RepositoryKnowledgeBase
from src.claude_analyzer import ClaudeAnalyzer, SmartContext
from src.embedding_engine import SemanticAnalyzer, TextEmbedder, CodeEmbedder
from src.database import DatabaseManager, User, Chat, ChatMessage, Commit, Repository, CommitAnalysis, RepositoryDashboard, FileContent, FileHistory, CommitFileCache
from src.storage_manager import RepositoryStorageManager
from src.github_service import GitHubService

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

# Legacy models (kept for backwards compatibility but will be deprecated)
class AnalyzeFileRequest(BaseModel):
    file_path: str
    max_commits: int = 50

class AnalyzeFunctionRequest(BaseModel):
    file_path: str
    function_name: str
    max_commits: int = 50

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

# Initialize global services
semantic_analyzer = SemanticAnalyzer()
claude_analyzer = ClaudeAnalyzer()
rag_system = RepositoryKnowledgeBase(semantic_analyzer, claude_analyzer=claude_analyzer)
breaking_change_detector = BreakingChangeDetector(semantic_analyzer)
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
            repo_rag_system = RepositoryKnowledgeBase(git_analyzer, claude_analyzer=claude_analyzer)
            rag_result = repo_rag_system.query(chat.repo_id, request.message, max_results=10)
            query_type = "repository"
            claude_enhanced = True  # RAG system doesn't track this, only enhanced endpoints do
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

# ===== LEGACY ENDPOINTS (DEPRECATED) =====

@app.post("/api/analyze-file/{repo_id}")
async def analyze_file_drift(repo_id: str, request: AnalyzeFileRequest):
    """[DEPRECATED] Analyze semantic drift for a specific file - use enhanced query endpoints instead"""
    logger.warning("analyze-file endpoint is deprecated. Use /api/repository/{repo_id}/file/{file_path}/insights instead")
    
    try:
        if repo_id not in active_repos:
            raise HTTPException(status_code=404, detail="Repository not found. Please clone first.")
        
        git_analyzer = active_repos[repo_id]
        
        # For now, return basic file information
        commits = git_analyzer.get_commits_for_file(request.file_path, request.max_commits)
        
        # Return simplified response
        return {
            "feature_id": f"file_{request.file_path}",
            "file_path": request.file_path,
            "function_name": None,
            "overall_drift": 0.0,
            "total_commits": len(commits),
            "drift_events": [],
            "change_summary": {"total_changes": len(commits)},
            "timeline": [],
            "risk_assessment": {"risk_level": "unknown"},
            "deprecated": True,
            "recommended_endpoint": f"/api/repository/{repo_id}/file/{request.file_path}/insights"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/analyze-function/{repo_id}")
async def analyze_function_drift(repo_id: str, request: AnalyzeFunctionRequest):
    """[DEPRECATED] Analyze semantic drift for a specific function - use enhanced query endpoints instead"""
    logger.warning("analyze-function endpoint is deprecated. Use enhanced query endpoints instead")
    
    try:
        if repo_id not in active_repos:
            raise HTTPException(status_code=404, detail="Repository not found. Please clone first.")
        
        git_analyzer = active_repos[repo_id]
        
        # For now, return basic function information
        commits = git_analyzer.get_commits_for_function(request.file_path, request.function_name, request.max_commits)
        
        # Return simplified response
        return {
            "feature_id": f"function_{request.function_name}",
            "file_path": request.file_path,
            "function_name": request.function_name,
            "overall_drift": 0.0,
            "total_commits": len(commits),
            "drift_events": [],
            "change_summary": {"total_changes": len(commits)},
            "timeline": [],
            "risk_assessment": {"risk_level": "unknown"},
            "deprecated": True,
            "recommended_endpoint": f"/api/query-repository/{repo_id}"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

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
    """Clean up repository resources"""
    try:
        if repo_id in active_repos:
            del active_repos[repo_id]
        
        if repo_id in temp_dirs:
            temp_dir = temp_dirs[repo_id]
            shutil.rmtree(temp_dir, ignore_errors=True)
            del temp_dirs[repo_id]
        
        return {"message": f"Repository {repo_id} cleaned up successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

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
    """Smart query using Smart RAG + Claude for direct answers"""
    try:
        if repo_id not in active_repos:
            raise HTTPException(status_code=404, detail="Repository not found. Please clone first.")
        
        git_analyzer = active_repos[repo_id]
        
        # Initialize RAG with git analyzer
        rag_system.git_analyzer = git_analyzer
        
        # Step 1: Smart context gathering with proper structure
        query_type, keywords, filters = rag_system._analyze_query(request.query)
        
        # Get repository context
        repo_context = rag_system._get_repo_context(repo_id)
        
        # Gather detailed context based on query type
        if query_type == 'file':
            context = rag_system._gather_file_context(repo_id, keywords, filters)
        elif query_type == 'commit':
            context = rag_system._gather_commit_context(repo_id, keywords, filters)
        elif query_type == 'author':
            context = rag_system._gather_author_context(repo_id, keywords, filters)
        elif query_type == 'security':
            context = rag_system._gather_security_context(repo_id, keywords, filters)
        elif query_type == 'diff':
            context = rag_system._gather_diff_context(repo_id, keywords, filters)
        else:
            context = rag_system._gather_general_context(repo_id, keywords, filters)
        
        # Calculate confidence
        confidence = rag_system._calculate_confidence(context, request.query)
        
        # Step 2: Create proper context object for Claude
        # Create rich context for Claude
        claude_context = SmartContext(
            query=request.query,
            query_type=query_type,
            files=context.get('files', []),
            commits=context.get('commits', []),
            repo_context=repo_context,
            confidence=confidence,
            reasoning=context.get('reasoning', f'{query_type} analysis')
        )
        
        # Step 3: Get Claude's analysis
        if claude_analyzer.available:
            claude_response = claude_analyzer.analyze(claude_context)
            
            return RepositoryQueryResponse(
                query=claude_response.query,
                response=claude_response.response,
                confidence=claude_response.confidence,
                sources=claude_response.sources,
                context_used=claude_response.context_used,
                suggestions=claude_response.suggestions,
                claude_enhanced=True
            )
        else:
            # Fallback without Claude
            fallback_response = claude_analyzer._create_fallback_response(claude_context)
            
            return RepositoryQueryResponse(
                query=fallback_response.query,
                response=fallback_response.response,
                confidence=fallback_response.confidence,
                sources=fallback_response.sources,
                context_used=fallback_response.context_used,
                suggestions=fallback_response.suggestions,
                claude_enhanced=False
            )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Smart query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
    except Exception as e:
        logger.error(f"Smart query failed: {str(e)}")
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
        
        # Analyze only new commits
        new_analyses = []
        for commit in recent_commits[:50]:  # Limit to 50 most recent for dashboard
            if commit.hexsha not in analyzed_commits:
                try:
                    logger.info(f"Analyzing new commit: {commit.hexsha[:8]}")
                    analysis = breaking_change_detector.analyze_commit(git_analyzer, commit.hexsha)
                    
                    # Save analysis to database
                    commit_analysis = CommitAnalysis(
                        id=f"{repo_id}_{commit.hexsha}",
                        repo_id=repo_id,
                        commit_hash=commit.hexsha,
                        overall_risk_score=float(analysis.overall_risk_score),
                        breaking_changes_count=len(analysis.breaking_changes),
                        complexity_score=float(analysis.complexity_score),
                        semantic_drift_score=float(analysis.semantic_drift_score),
                        files_changed=json.dumps(analysis.files_changed),
                        breaking_changes=json.dumps([{
                            "change_type": bc.change_type.value,
                            "risk_level": bc.risk_level.value,
                            "file_path": bc.file_path,
                            "description": bc.description
                        } for bc in analysis.breaking_changes]),
                        analyzed_at=datetime.now().isoformat()
                    )
                    
                    if db_manager.save_commit_analysis(commit_analysis):
                        analyzed_commits[commit.hexsha] = commit_analysis
                        new_analyses.append(commit_analysis)
                    
                except Exception as e:
                    logger.error(f"Error analyzing commit {commit.hexsha}: {e}")
                    continue
        
        # Generate dashboard from all analyzed commits
        risk_data = {
            "total_commits_analyzed": len(analyzed_commits),
            "high_risk_commits": 0,
            "breaking_changes_by_type": {},
            "risk_trend": [],
            "most_risky_files": {},
            "recent_high_risk_commits": [],
            "performance_stats": {
                "new_commits_analyzed": len(new_analyses),
                "cached_analyses_used": len(analyzed_commits) - len(new_analyses),
                "cache_hit_rate": round((len(analyzed_commits) - len(new_analyses)) / max(len(analyzed_commits), 1) * 100, 1)
            }
        }
        
        # Process cached analyses for dashboard
        for analysis in analyzed_commits.values():
            if analysis.overall_risk_score > 0.7:
                risk_data["high_risk_commits"] += 1
                risk_data["recent_high_risk_commits"].append({
                    "commit_hash": analysis.commit_hash[:8],
                    "risk_score": float(analysis.overall_risk_score),
                    "breaking_changes_count": analysis.breaking_changes_count
                })
            
            # Count breaking change types from stored data
            try:
                breaking_changes = json.loads(analysis.breaking_changes)
                for bc in breaking_changes:
                    change_type = bc.get("change_type", "unknown")
                    risk_data["breaking_changes_by_type"][change_type] = \
                        risk_data["breaking_changes_by_type"].get(change_type, 0) + 1
            except:
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
                "risk_score": float(analysis.overall_risk_score)
            })
        
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
    """Get detailed analysis of a specific commit with RAG context"""
    try:
        if repo_id not in active_repos:
            raise HTTPException(status_code=404, detail="Repository not found.")
        
        git_analyzer = active_repos[repo_id]
        
        # Get commit analysis
        commit_analysis = breaking_change_detector.analyze_commit(git_analyzer, commit_hash)
        
        # Get RAG context for this commit
        rag_query = f"analyze commit {commit_hash} changes risk breaking"
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
                    "risk_score": commit_analysis.overall_risk_score
                }
                enhanced_analysis = claude_analyzer.analyze_commit_breaking_changes(
                    commit_data, rag_result.context_used
                )
            except Exception as e:
                logger.warning(f"Claude analysis failed: {str(e)}")
        
        result = {
            "commit_analysis": asdict(commit_analysis),
            "rag_insights": {
                "response": rag_result.response,
                "confidence": rag_result.confidence,
                "sources": rag_result.sources,
                "suggestions": rag_result.suggestions
            },
            "enhanced_analysis": enhanced_analysis,
            "risk_assessment": {
                "is_high_risk": commit_analysis.overall_risk_score > 0.7,
                "risk_level": "High" if commit_analysis.overall_risk_score > 0.7 
                           else "Medium" if commit_analysis.overall_risk_score > 0.3 
                           else "Low",
                "total_breaking_changes": len(commit_analysis.breaking_changes),
                "files_at_risk": len(commit_analysis.files_changed)
            }
        }
        
        return convert_numpy_types(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Commit analysis failed: {str(e)}")

@app.get("/api/repository/{repo_id}/file/{file_path:path}/insights")
async def get_file_insights(
    repo_id: str, 
    file_path: str,
    current_user: str = Depends(require_repository_access)
):
    """Get comprehensive insights about a specific file"""
    try:
        if repo_id not in active_repos:
            raise HTTPException(status_code=404, detail="Repository not found.")
        
        git_analyzer = active_repos[repo_id]
        
        # Get file history and analysis
        repo = git_analyzer.repo
        file_commits = list(repo.iter_commits(paths=file_path, max_count=50))
        
        # RAG query for file-specific insights
        rag_query = f"file {file_path} changes history modifications risk"
        rag_result = rag_system.query(repo_id, rag_query, max_results=10)
        
        # Analyze file risk patterns
        risk_history = []
        total_risk = 0.0
        for commit in file_commits[:20]:
            try:
                analysis = breaking_change_detector.analyze_commit(git_analyzer, commit.hexsha)
                if file_path in analysis.files_changed:
                    risk_history.append({
                        "commit_hash": commit.hexsha[:8],
                        "timestamp": commit.committed_datetime.isoformat(),
                        "message": commit.message[:100],
                        "risk_score": float(analysis.overall_risk_score),
                        "breaking_changes": len([bc for bc in analysis.breaking_changes 
                                               if file_path in bc.file_path])
                    })
                    total_risk += analysis.overall_risk_score
            except Exception as e:
                continue
        
        # Calculate file risk metrics
        avg_risk = total_risk / len(risk_history) if risk_history else 0.0
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
                "risk_level": "High" if avg_risk > 0.7 else "Medium" if avg_risk > 0.3 else "Low",
                "modification_frequency": modification_frequency,
                "recent_activity_score": recent_activity,
                "is_high_activity": modification_frequency > 10
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
    """Enhanced RAG query with Claude integration and advanced context analysis"""
    try:
        if repo_id not in active_repos:
            raise HTTPException(status_code=404, detail="Repository not found.")
        
        # Get comprehensive RAG response
        rag_result = rag_system.query(repo_id, request.query, max_results=10)
        
        # Enhanced response with Claude if available
        enhanced_response = None
        if claude_analyzer.available and rag_result.confidence > 0.3:
            try:
                enhanced_response = claude_analyzer.enhance_rag_response(
                    request.query, 
                    rag_result.response, 
                    rag_result.sources[:8]  # Limit context for Claude
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
        
        # Analyze semantic patterns
        semantic_patterns = {
            "commit_message_themes": {},
            "file_change_patterns": {},
            "risk_evolution": [],
            "breaking_change_trends": {},
            "developer_impact": {}
        }
        
        for commit in recent_commits:
            try:
                analysis = breaking_change_detector.analyze_commit(git_analyzer, commit.hexsha)
                
                # Track risk evolution
                semantic_patterns["risk_evolution"].append({
                    "date": commit.committed_datetime.isoformat(),
                    "risk_score": float(analysis.overall_risk_score),
                    "breaking_changes": len(analysis.breaking_changes)
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
                
                # Track breaking change trends
                for bc in analysis.breaking_changes:
                    change_type = bc.change_type.value
                    semantic_patterns["breaking_change_trends"][change_type] = \
                        semantic_patterns["breaking_change_trends"].get(change_type, 0) + 1
                
                # Developer impact analysis
                author = commit.author.name
                if author not in semantic_patterns["developer_impact"]:
                    semantic_patterns["developer_impact"][author] = {
                        "commits": 0,
                        "total_risk": 0.0,
                        "breaking_changes": 0
                    }
                
                semantic_patterns["developer_impact"][author]["commits"] += 1
                semantic_patterns["developer_impact"][author]["total_risk"] += analysis.overall_risk_score
                semantic_patterns["developer_impact"][author]["breaking_changes"] += len(analysis.breaking_changes)
                
            except Exception as e:
                continue
        
        # Get top themes and patterns
        top_themes = sorted(semantic_patterns["commit_message_themes"].items(), 
                           key=lambda x: x[1], reverse=True)[:10]
        
        top_file_patterns = sorted(semantic_patterns["file_change_patterns"].items(), 
                                  key=lambda x: x[1], reverse=True)[:10]
        
        # Calculate developer risk scores
        for author, stats in semantic_patterns["developer_impact"].items():
            if stats["commits"] > 0:
                stats["avg_risk"] = float(stats["total_risk"] / stats["commits"])
            else:
                stats["avg_risk"] = 0.0
        
        top_risk_developers = sorted(
            [(author, stats) for author, stats in semantic_patterns["developer_impact"].items()],
            key=lambda x: x[1]["avg_risk"], reverse=True
        )[:5]
        
        # Enhanced RAG query about drift patterns
        drift_query = f"semantic drift patterns changes {days} days breaking changes trends"
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
                "risk_trend_summary": {
                    "avg_risk": float(np.mean([r["risk_score"] for r in semantic_patterns["risk_evolution"]])) if semantic_patterns["risk_evolution"] else 0,
                    "max_risk": float(max([r["risk_score"] for r in semantic_patterns["risk_evolution"]])) if semantic_patterns["risk_evolution"] else 0,
                    "total_breaking_changes": sum(r["breaking_changes"] for r in semantic_patterns["risk_evolution"])
                }
            },
            "developer_impact": {
                "top_risk_contributors": [
                    {
                        "author": author,
                        "avg_risk_score": stats["avg_risk"],
                        "total_commits": stats["commits"],
                        "breaking_changes": stats["breaking_changes"]
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
                        raise HTTPException(status_code=404, detail="No changes found for this file in this commit")
                    
                    diff_content = str(diffs[0])
                    return {
                        "repo_id": repo_id,
                        "file_path": file_path,
                        "commit_hash": commit_hash,
                        "parent_commit": parent_commit.hexsha,
                        "type": "diff",
                        "content": diff_content,
                        "change_type": diffs[0].change_type,
                        "insertions": diffs[0].diff.decode('utf-8').count('\n+') if diffs[0].diff else 0,
                        "deletions": diffs[0].diff.decode('utf-8').count('\n-') if diffs[0].diff else 0
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
    current_user: str = Depends(require_repository_access)
):
    """Get list of files changed in a specific commit"""
    try:
        if repo_id not in active_repos:
            raise HTTPException(status_code=404, detail="Repository not found")
        
        git_analyzer = active_repos[repo_id]
        
        # Use caching function to get commit files data
        cached_commit = get_or_cache_commit_files(repo_id, commit_hash, git_analyzer, include_diff_stats)
        if not cached_commit:
            raise HTTPException(status_code=404, detail="Commit not found or error processing commit")
        
        # Parse cached data
        files_changed = json.loads(cached_commit.files_data)
        summary_stats = json.loads(cached_commit.summary_stats)
        
        # Get commit info
        repo = git_analyzer.repo
        try:
            commit = repo.commit(commit_hash)
        except:
            raise HTTPException(status_code=404, detail="Commit not found")
        
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
            "files_changed": files_changed,
            "summary": summary_stats,
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
                            "base_commit": base_commit,
                            "head_commit": head_commit,
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
                            "base_commit": base_commit,
                            "head_commit": head_commit,
                            "file_path": file_path,
                            "change_type": "deleted",
                            "diff": diff_content
                        }
                    else:
                        return {
                            "repo_id": repo_id,
                            "base_commit": base_commit,
                            "head_commit": head_commit,
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
                    "base_commit": base_commit,
                    "head_commit": head_commit,
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
                
                if diff_item.change_type == 'R':
                    file_info["old_path"] = diff_item.a_path
                    file_info["new_path"] = diff_item.b_path
                
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
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any, Optional, Union
import tempfile
import shutil
import os
import subprocess
from datetime import datetime, timedelta
import json
import traceback
import logging
import numpy as np
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

from src.config import config
from src.git_analyzer import GitAnalyzer
from src.breaking_change_detector import BreakingChangeDetector, CommitBreakingChangeAnalysis
from src.rag_system import RepositoryKnowledgeBase
from src.claude_analyzer import ClaudeAnalyzer, SmartContext
from src.embedding_engine import SemanticAnalyzer, TextEmbedder, CodeEmbedder
from src.database import DatabaseManager
from src.storage_manager import RepositoryStorageManager

app = FastAPI(title="DiffSense API", description="Semantic drift detection for git repositories")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class AnalyzeRepositoryRequest(BaseModel):
    repo_url: HttpUrl
    file_path: Optional[str] = None
    function_name: Optional[str] = None
    max_commits: int = 50

class AnalyzeCommitRequest(BaseModel):
    commit_hash: str
    include_claude_analysis: bool = True

class AnalyzeCommitRangeRequest(BaseModel):
    start_commit: str
    end_commit: str
    max_commits: int = 100
    include_claude_analysis: bool = False  # Disabled by default for cleaner responses

class QueryRepositoryRequest(BaseModel):
    query: str
    max_results: int = 10
    include_claude_response: bool = False  # Disabled by default for cleaner responses

class AnalyzeFileRequest(BaseModel):
    repo_path: str
    file_path: str
    max_commits: int = 50

class AnalyzeFunctionRequest(BaseModel):
    repo_path: str
    file_path: str
    function_name: str
    max_commits: int = 30

class QueryRequest(BaseModel):
    query: str
    max_results: int = 10

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
rag_system = RepositoryKnowledgeBase(semantic_analyzer)
claude_analyzer = ClaudeAnalyzer()
breaking_change_detector = BreakingChangeDetector(semantic_analyzer)

# Initialize database and storage
db_manager = DatabaseManager()
storage_manager = RepositoryStorageManager()

# Legacy in-memory storage for backwards compatibility
active_repos: Dict[str, GitAnalyzer] = {}
temp_dirs: Dict[str, str] = {}

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "DiffSense API is running", 
        "timestamp": datetime.now().isoformat(),
        "config_status": config.validate_config()
    }

@app.post("/api/clone-repository")
async def clone_repository(request: AnalyzeRepositoryRequest, background_tasks: BackgroundTasks):
    """Clone a repository and return basic stats"""
    try:
        print(f"Received request: {request}")
        repo_url_str = str(request.repo_url)
        print(f"Repository URL: {repo_url_str}")
        
        # Generate repository ID and setup storage
        repo_id = storage_manager.generate_repo_id(repo_url_str)
        repo_path = storage_manager.get_repository_path(repo_id, "active")
        
        # Check if repository already exists in database
        existing_repo = db_manager.get_repository_by_url(repo_url_str)
        if existing_repo and os.path.exists(repo_path):
            print(f"Repository already exists: {repo_id}")
            # Load existing repository
            git_analyzer = GitAnalyzer(repo_path)
            active_repos[repo_id] = git_analyzer
            
            # Get repository stats
            stats = git_analyzer.analyze_repository_stats()
            
            return {
                "repo_id": repo_id,
                "status": "loaded_existing",
                "stats": RepositoryStatsResponse(**stats),
                "message": f"Loaded existing repository from {repo_url_str}"
            }
        
        # Clone repository to organized storage
        print(f"Cloning to: {repo_path}")
        git_analyzer = GitAnalyzer.clone_repository(repo_url_str, repo_path)
        print(f"Successfully created GitAnalyzer")
        
        # Store repository in database
        if not existing_repo:
            db_manager.add_repository(
                url=repo_url_str,
                name=repo_id,
                storage_path=repo_path,
                last_analyzed=datetime.now()
            )
        
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

@app.post("/api/analyze-file/{repo_id}")
async def analyze_file_drift(repo_id: str, request: AnalyzeFileRequest):
    """Analyze semantic drift for a specific file"""
    try:
        if repo_id not in active_repos:
            raise HTTPException(status_code=404, detail="Repository not found. Please clone first.")
        
        git_analyzer = active_repos[repo_id]
        
        # TODO: Re-enable drift analysis once transformers issue is resolved
        # detector = DriftDetector(git_analyzer)
        # feature_history = detector.analyze_file_drift(request.file_path, request.max_commits)
        
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
            "risk_assessment": {"risk_level": "unknown"}
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/analyze-function/{repo_id}")
async def analyze_function_drift(repo_id: str, request: AnalyzeFunctionRequest):
    """Analyze semantic drift for a specific function"""
    try:
        if repo_id not in active_repos:
            raise HTTPException(status_code=404, detail="Repository not found. Please clone first.")
        
        git_analyzer = active_repos[repo_id]
        
        # TODO: Re-enable drift analysis once transformers issue is resolved
        # detector = DriftDetector(git_analyzer)
        # feature_history = detector.analyze_function_drift(
        #     request.file_path, 
        #     request.function_name, 
        #     request.max_commits
        # )
        
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
            "risk_assessment": {"risk_level": "unknown"}
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/api/repository/{repo_id}/files")
async def list_repository_files(repo_id: str):
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
async def get_repository_stats(repo_id: str):
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
async def cleanup_repository(repo_id: str):
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
async def get_repository_commits(repo_id: str, limit: int = Query(50, description="Maximum number of commits to return")):
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

# Background task functions
async def index_repository_for_rag(repo_id: str, repo_url: str, git_analyzer):
    """Background task to index repository in RAG system"""
    try:
        rag_system.index_repository(git_analyzer, repo_id, repo_url)
        print(f"Successfully indexed repository {repo_id} for RAG")
    except Exception as e:
        print(f"Error indexing repository {repo_id} for RAG: {str(e)}")

async def store_repository_commits(repo_id: str, git_analyzer):
    """Background task to store repository commits in database"""
    try:
        # Get repository from database
        repo_url = None
        for rid, analyzer in active_repos.items():
            if rid == repo_id:
                # Find the repo URL from the git remote
                try:
                    remote_url = analyzer.repo.remotes.origin.url
                    repo_url = remote_url
                    break
                except:
                    pass
        
        if not repo_url:
            print(f"Could not find repo URL for {repo_id}")
            return
            
        repository = db_manager.get_repository_by_url(repo_url)
        if not repository:
            print(f"Repository {repo_id} not found in database")
            return
        
        # Store recent commits (limit to avoid overwhelming the database)
        commits = list(git_analyzer.repo.iter_commits(max_count=100))
        
        for commit in commits:
            try:
                # Check if commit already exists
                existing = db_manager.get_commit(repository.id, commit.hexsha)
                if existing:
                    continue
                
                # Add new commit
                db_manager.add_commit(
                    repository_id=repository.id,
                    commit_hash=commit.hexsha,
                    message=commit.message.strip(),
                    author=commit.author.name,
                    timestamp=datetime.fromtimestamp(commit.committed_date),
                    files_changed=len(commit.stats.files)
                )
                
                # Store file records for this commit
                for file_path, stats in commit.stats.files.items():
                    db_manager.add_file_record(
                        commit_id=db_manager.get_commit(repository.id, commit.hexsha).id,
                        file_path=file_path,
                        lines_added=stats['insertions'],
                        lines_deleted=stats['deletions']
                    )
                    
            except Exception as e:
                print(f"Error storing commit {commit.hexsha}: {str(e)}")
                continue
        
        print(f"Successfully stored commits for repository {repo_id}")
        
    except Exception as e:
        print(f"Error storing repository commits {repo_id}: {str(e)}")

async def cleanup_old_repos():
    """Background task to cleanup old temporary repositories"""
    try:
        # In production, implement proper cleanup logic
        # For now, just clean up very old temporary directories
        import time
        current_time = time.time()
        
        for repo_id, temp_dir in list(temp_dirs.items()):
            try:
                # Check if directory is older than 1 hour
                dir_age = current_time - os.path.getctime(temp_dir)
                if dir_age > 3600:  # 1 hour
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    del temp_dirs[repo_id]
                    if repo_id in active_repos:
                        del active_repos[repo_id]
                    print(f"Cleaned up old repository: {repo_id}")
            except Exception as e:
                print(f"Error cleaning up repository {repo_id}: {str(e)}")
                
    except Exception as e:
        print(f"Error in cleanup task: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check with system status"""
    return {
        "status": "healthy",
        "active_repositories": len(active_repos),
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "config": config.validate_config()
    }

@app.get("/api/test-git")
async def test_git():
    """Test git functionality"""
    try:
        import subprocess
        
        # Check if git is available
        result = subprocess.run(['git', '--version'], capture_output=True, text=True)
        git_version = result.stdout.strip() if result.returncode == 0 else "Git not found"
        
        # Test temp directory creation
        temp_dir = tempfile.mkdtemp(prefix="test_")
        temp_exists = os.path.exists(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return {
            "git_available": result.returncode == 0,
            "git_version": git_version,
            "temp_dir_creation": temp_exists,
            "python_version": os.sys.version
        }
    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

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
async def get_repository_summary(repo_id: str):
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
async def search_commits(repo_id: str, query: str = Query(...), max_results: int = Query(10)):
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
async def search_files(repo_id: str, query: str = Query(...), max_results: int = Query(10)):
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
async def get_risk_dashboard(repo_id: str):
    """Get comprehensive risk dashboard for repository"""
    try:
        if repo_id not in active_repos:
            raise HTTPException(status_code=404, detail="Repository not found.")
        
        git_analyzer = active_repos[repo_id]
        
        # Get recent commits for analysis
        repo = git_analyzer.repo
        recent_commits = list(repo.iter_commits(max_count=50))
        
        # Analyze recent commits for breaking changes
        risk_data = {
            "total_commits_analyzed": 0,
            "high_risk_commits": 0,
            "breaking_changes_by_type": {},
            "risk_trend": [],
            "most_risky_files": {},
            "recent_high_risk_commits": []
        }
        
        for commit in recent_commits[:20]:  # Analyze recent 20 commits
            try:
                analysis = breaking_change_detector.analyze_commit(git_analyzer, commit.hexsha)
                risk_data["total_commits_analyzed"] += 1
                
                if analysis.overall_risk_score > 0.7:
                    risk_data["high_risk_commits"] += 1
                    risk_data["recent_high_risk_commits"].append({
                        "commit_hash": analysis.commit_hash[:8],
                        "message": analysis.commit_message[:100],
                        "risk_score": float(analysis.overall_risk_score),
                        "breaking_changes_count": len(analysis.breaking_changes)
                    })
                
                # Count breaking change types
                for bc in analysis.breaking_changes:
                    change_type = bc.change_type.value
                    risk_data["breaking_changes_by_type"][change_type] = \
                        risk_data["breaking_changes_by_type"].get(change_type, 0) + 1
                
                # Track risky files
                for file_path in analysis.files_changed:
                    risk_data["most_risky_files"][file_path] = \
                        risk_data["most_risky_files"].get(file_path, 0) + float(analysis.overall_risk_score)
                
                # Add to trend
                risk_data["risk_trend"].append({
                    "commit_hash": analysis.commit_hash[:8],
                    "timestamp": analysis.timestamp,
                    "risk_score": float(analysis.overall_risk_score)
                })
                
            except Exception as e:
                print(f"Error analyzing commit {commit.hexsha}: {str(e)}")
                continue
        
        # Sort most risky files
        risk_data["most_risky_files"] = dict(
            sorted([(k, float(v)) for k, v in risk_data["most_risky_files"].items()], 
                   key=lambda x: x[1], reverse=True)[:10]
        )
        
        # Calculate overall repository risk score
        if risk_data["total_commits_analyzed"] > 0:
            risk_data["overall_risk_score"] = float(risk_data["high_risk_commits"] / risk_data["total_commits_analyzed"])
        else:
            risk_data["overall_risk_score"] = 0.0
        
        # Convert numpy types to Python native types for JSON serialization
        risk_data = convert_numpy_types(risk_data)
        
        return risk_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate risk dashboard: {str(e)}")

@app.get("/api/repository/{repo_id}/commit/{commit_hash}/analysis")
async def get_commit_analysis(repo_id: str, commit_hash: str):
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
async def get_file_insights(repo_id: str, file_path: str):
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
async def enhanced_rag_query(repo_id: str, request: QueryRequest):
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
async def get_semantic_drift_analysis(repo_id: str, days: int = Query(30, description="Number of days to analyze")):
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

# ...existing code...

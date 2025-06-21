from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any, Optional, Union
import tempfile
import shutil
import os
import subprocess
from datetime import datetime
import json
import traceback
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.config import config
from src.git_analyzer import GitAnalyzer
from src.breaking_change_detector import BreakingChangeDetector, CommitBreakingChangeAnalysis
from src.rag_system import RepositoryKnowledgeBase
from src.claude_analyzer import ClaudeAnalyzer
from src.embedding_engine import SemanticAnalyzer, TextEmbedder, CodeEmbedder

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
    include_claude_analysis: bool = True

class QueryRepositoryRequest(BaseModel):
    query: str
    max_results: int = 10
    include_claude_response: bool = True

class AnalyzeFileRequest(BaseModel):
    repo_path: str
    file_path: str
    max_commits: int = 50

class AnalyzeFunctionRequest(BaseModel):
    repo_path: str
    file_path: str
    function_name: str
    max_commits: int = 30

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
active_repos: Dict[str, GitAnalyzer] = {}
temp_dirs: Dict[str, str] = {}

# Initialize global services
semantic_analyzer = SemanticAnalyzer()
claude_analyzer = ClaudeAnalyzer()
rag_system = RepositoryKnowledgeBase(semantic_analyzer)
breaking_change_detector = BreakingChangeDetector(semantic_analyzer)

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
    temp_dir = None
    try:
        print(f"Received request: {request}")
        print(f"Repository URL: {request.repo_url}")
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="diffsense_")
        print(f"Created temp directory: {temp_dir}")
        
        # Clone repository
        repo_url_str = str(request.repo_url)
        print(f"Attempting to clone: {repo_url_str}")
        
        git_analyzer = GitAnalyzer.clone_repository(repo_url_str, temp_dir)
        print(f"Successfully created GitAnalyzer")
        
        # Store in global state
        repo_id = f"repo_{len(active_repos)}"
        active_repos[repo_id] = git_analyzer
        temp_dirs[repo_id] = temp_dir
        print(f"Stored as repo_id: {repo_id}")
        
        # Get repository stats
        stats = git_analyzer.analyze_repository_stats()
        print(f"Got repository stats: {stats}")
        
        # Index repository in RAG system for intelligent querying
        background_tasks.add_task(index_repository_for_rag, repo_id, repo_url_str, git_analyzer)
        
        # Schedule cleanup (in production, implement proper cleanup)
        background_tasks.add_task(cleanup_old_repos)
        
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
        
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"Cleaned up temp directory: {temp_dir}")
        
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

async def cleanup_old_repos():
    """Background task to cleanup old repositories"""
    # In production, implement proper cleanup based on timestamps
    # For demo, keep repos for the session
    pass

async def index_repository_for_rag(repo_id: str, repo_url: str, git_analyzer):
    """Background task to index repository in RAG system"""
    try:
        # Index repository for intelligent querying
        await rag_system.index_repository(repo_id, git_analyzer, repo_url)
        logger.info(f"Successfully indexed repository {repo_id} for RAG")
    except Exception as e:
        logger.error(f"Failed to index repository {repo_id} for RAG: {str(e)}")

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
        if request.include_claude_analysis and claude_analyzer.is_available():
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
            
            claude_response = claude_analyzer.analyze_breaking_changes(breaking_changes_data, commit_context)
            claude_analysis = {
                "content": claude_response.content,
                "confidence": claude_response.confidence,
                "suggestions": claude_response.suggestions,
                "metadata": claude_response.metadata
            }
        
        # Convert to response format
        breaking_changes_response = [
            BreakingChangeResponse(
                change_type=bc.change_type.value,
                risk_level=bc.risk_level.value,
                confidence=bc.confidence,
                file_path=bc.file_path,
                line_number=bc.line_number,
                description=bc.description,
                before_code=bc.before_code,
                after_code=bc.after_code,
                impact_analysis=bc.impact_analysis,
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
            overall_risk_score=analysis.overall_risk_score,
            breaking_changes=breaking_changes_response,
            files_changed=analysis.files_changed,
            lines_added=analysis.lines_added,
            lines_removed=analysis.lines_removed,
            complexity_score=analysis.complexity_score,
            semantic_drift_score=analysis.semantic_drift_score,
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
            if request.include_claude_analysis and claude_analyzer.is_available() and analysis.breaking_changes:
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
                
                claude_response = claude_analyzer.analyze_breaking_changes(breaking_changes_data, commit_context)
                claude_analysis = {
                    "content": claude_response.content,
                    "confidence": claude_response.confidence,
                    "suggestions": claude_response.suggestions
                }
            
            breaking_changes_response = [
                BreakingChangeResponse(
                    change_type=bc.change_type.value,
                    risk_level=bc.risk_level.value,
                    confidence=bc.confidence,
                    file_path=bc.file_path,
                    line_number=bc.line_number,
                    description=bc.description,
                    before_code=bc.before_code,
                    after_code=bc.after_code,
                    impact_analysis=bc.impact_analysis,
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
                overall_risk_score=analysis.overall_risk_score,
                breaking_changes=breaking_changes_response,
                files_changed=analysis.files_changed,
                lines_added=analysis.lines_added,
                lines_removed=analysis.lines_removed,
                complexity_score=analysis.complexity_score,
                semantic_drift_score=analysis.semantic_drift_score,
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
    """Query repository using RAG system"""
    try:
        if repo_id not in active_repos:
            raise HTTPException(status_code=404, detail="Repository not found. Please clone first.")
        
        # Query the RAG system
        result = rag_system.query(repo_id, request.query, request.max_results)
        
        # Enhance with Claude if requested
        claude_enhanced = False
        if request.include_claude_response and claude_analyzer.is_available():
            context = {
                "query": request.query,
                "rag_response": result.response,
                "sources": result.sources,
                "confidence": result.confidence
            }
            
            claude_response = claude_analyzer.answer_repository_question(request.query, context)
            
            # Merge Claude's response with RAG response
            enhanced_response = f"{result.response}\n\nEnhanced Analysis:\n{claude_response.content}"
            result = RepositoryQueryResponse(
                query=result.query,
                response=enhanced_response,
                confidence=max(result.confidence, claude_response.confidence),
                sources=result.sources,
                context_used=result.context_used,
                suggestions=result.suggestions + claude_response.suggestions,
                claude_enhanced=True
            )
            claude_enhanced = True
        
        return RepositoryQueryResponse(
            query=result.query,
            response=result.response,
            confidence=result.confidence,
            sources=result.sources,
            context_used=result.context_used,
            suggestions=result.suggestions,
            claude_enhanced=claude_enhanced
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
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
        if claude_analyzer.is_available():
            recent_changes = summary.get("recent_activity", [])
            claude_response = claude_analyzer.assess_repository_risk(summary, recent_changes)
            
            summary["claude_risk_assessment"] = {
                "content": claude_response.content,
                "confidence": claude_response.confidence,
                "suggestions": claude_response.suggestions,
                "metadata": claude_response.metadata
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
                        "risk_score": analysis.overall_risk_score,
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
                        risk_data["most_risky_files"].get(file_path, 0) + analysis.overall_risk_score
                
                # Add to trend
                risk_data["risk_trend"].append({
                    "commit_hash": analysis.commit_hash[:8],
                    "timestamp": analysis.timestamp,
                    "risk_score": analysis.overall_risk_score
                })
                
            except Exception as e:
                print(f"Error analyzing commit {commit.hexsha}: {str(e)}")
                continue
        
        # Sort most risky files
        risk_data["most_risky_files"] = dict(
            sorted(risk_data["most_risky_files"].items(), key=lambda x: x[1], reverse=True)[:10]
        )
        
        # Calculate overall repository risk score
        if risk_data["total_commits_analyzed"] > 0:
            risk_data["overall_risk_score"] = risk_data["high_risk_commits"] / risk_data["total_commits_analyzed"]
        else:
            risk_data["overall_risk_score"] = 0.0
        
        return risk_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate risk dashboard: {str(e)}")

# Background task functions
async def index_repository_for_rag(repo_id: str, repo_url: str, git_analyzer):
    """Background task to index repository in RAG system"""
    try:
        rag_system.index_repository(git_analyzer, repo_id, repo_url)
        print(f"Successfully indexed repository {repo_id} for RAG")
    except Exception as e:
        print(f"Error indexing repository {repo_id} for RAG: {str(e)}")

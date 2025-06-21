from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any, Optional
import tempfile
import shutil
import os
from datetime import datetime
import json

from src.git_analyzer import GitAnalyzer
from src.drift_detector import DriftDetector, FeatureHistory

app = FastAPI(title="DiffSense API", description="Semantic drift detection for git repositories")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
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

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "DiffSense API is running", "timestamp": datetime.now().isoformat()}

@app.post("/api/clone-repository")
async def clone_repository(request: AnalyzeRepositoryRequest, background_tasks: BackgroundTasks):
    """Clone a repository and return basic stats"""
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="diffsense_")
        
        # Clone repository
        repo_url_str = str(request.repo_url)
        git_analyzer = GitAnalyzer.clone_repository(repo_url_str, temp_dir)
        
        # Store in global state
        repo_id = f"repo_{len(active_repos)}"
        active_repos[repo_id] = git_analyzer
        temp_dirs[repo_id] = temp_dir
        
        # Get repository stats
        stats = git_analyzer.analyze_repository_stats()
        
        # Schedule cleanup (in production, implement proper cleanup)
        background_tasks.add_task(cleanup_old_repos)
        
        return {
            "repo_id": repo_id,
            "status": "cloned",
            "stats": RepositoryStatsResponse(**stats),
            "message": f"Successfully cloned repository from {repo_url_str}"
        }
        
    except Exception as e:
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Failed to clone repository: {str(e)}")

@app.post("/api/analyze-file/{repo_id}")
async def analyze_file_drift(repo_id: str, request: AnalyzeFileRequest):
    """Analyze semantic drift for a specific file"""
    try:
        if repo_id not in active_repos:
            raise HTTPException(status_code=404, detail="Repository not found. Please clone first.")
        
        git_analyzer = active_repos[repo_id]
        detector = DriftDetector(git_analyzer)
        
        # Analyze file drift
        feature_history = detector.analyze_file_drift(request.file_path, request.max_commits)
        
        # Generate timeline and risk assessment
        timeline = detector.generate_drift_timeline(feature_history)
        risk_assessment = detector.predict_breaking_change_risk(feature_history)
        
        # Convert drift events to response format
        drift_events = [
            DriftEventResponse(
                commit_hash=event.commit_hash,
                timestamp=event.timestamp.isoformat(),
                drift_score=event.drift_score,
                change_magnitude=event.change_magnitude,
                file_path=event.file_path,
                description=event.description,
                commit_message=event.commit_message,
                added_lines=event.added_lines,
                removed_lines=event.removed_lines
            )
            for event in feature_history.drift_events
        ]
        
        return FeatureHistoryResponse(
            feature_id=feature_history.feature_id,
            file_path=feature_history.file_path,
            function_name=feature_history.function_name,
            overall_drift=feature_history.overall_drift,
            total_commits=len(feature_history.commits),
            drift_events=drift_events,
            change_summary=feature_history.change_summary,
            timeline=timeline,
            risk_assessment=risk_assessment
        )
        
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
        detector = DriftDetector(git_analyzer)
        
        # Analyze function drift
        feature_history = detector.analyze_function_drift(
            request.file_path, 
            request.function_name, 
            request.max_commits
        )
        
        # Generate timeline and risk assessment
        timeline = detector.generate_drift_timeline(feature_history)
        risk_assessment = detector.predict_breaking_change_risk(feature_history)
        
        # Convert drift events to response format
        drift_events = [
            DriftEventResponse(
                commit_hash=event.commit_hash,
                timestamp=event.timestamp.isoformat(),
                drift_score=event.drift_score,
                change_magnitude=event.change_magnitude,
                file_path=event.file_path,
                description=event.description,
                commit_message=event.commit_message,
                added_lines=event.added_lines,
                removed_lines=event.removed_lines
            )
            for event in feature_history.drift_events
        ]
        
        return FeatureHistoryResponse(
            feature_id=feature_history.feature_id,
            file_path=feature_history.file_path,
            function_name=feature_history.function_name,
            overall_drift=feature_history.overall_drift,
            total_commits=len(feature_history.commits),
            drift_events=drift_events,
            change_summary=feature_history.change_summary,
            timeline=timeline,
            risk_assessment=risk_assessment
        )
        
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

@app.get("/api/health")
async def health_check():
    """Health check with system status"""
    return {
        "status": "healthy",
        "active_repositories": len(active_repos),
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

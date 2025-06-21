"""
Advanced Breaking Change Detection System
Analyzes commits for potential breaking changes using ML and heuristics
"""

import ast
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ChangeType(Enum):
    API_CHANGE = "api_change"
    SIGNATURE_CHANGE = "signature_change"
    BEHAVIOR_CHANGE = "behavior_change"
    DEPENDENCY_CHANGE = "dependency_change"
    CONFIG_CHANGE = "config_change"

class RiskLevel(Enum):
    """Risk levels for breaking changes"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class BreakingChange:
    """Represents a detected breaking change"""
    change_type: ChangeType
    risk_level: RiskLevel
    confidence: float
    file_path: str
    line_number: Optional[int]
    description: str
    before_code: Optional[str]
    after_code: Optional[str]
    impact_analysis: Dict[str, Any]
    mitigation_suggestions: List[str]
    related_files: List[str]

@dataclass
class CommitBreakingChangeAnalysis:
    """Analysis results for a single commit"""
    commit_hash: str
    commit_message: str
    timestamp: str
    author: str
    overall_risk_score: float
    breaking_changes: List[BreakingChange]
    files_changed: List[str]
    lines_added: int
    lines_removed: int
    complexity_score: float
    semantic_drift_score: float

class BreakingChangeDetector:
    """Detector for breaking changes using semantic analysis and heuristics"""
    
    def __init__(self, semantic_analyzer):
        self.semantic_analyzer = semantic_analyzer
        self.risk_threshold_high = 0.7
        self.risk_threshold_medium = 0.4
        logger.info("Initialized breaking change detector with semantic analysis")
    
    def analyze_commit(self, git_analyzer, commit_hash: str) -> CommitBreakingChangeAnalysis:
        """Analyze commit for breaking changes using semantic analysis and heuristics"""
        repo = git_analyzer.repo
        commit = repo.commit(commit_hash)
        
        # Get basic commit info
        stats = commit.stats.total
        files_changed = [item.a_path or item.b_path for item in commit.diff(commit.parents[0] if commit.parents else None)]
        
        # Extract diff information for analysis
        diff_infos = git_analyzer.extract_diff_info(commit)
        
        # Analyze each changed file for breaking changes
        breaking_changes = []
        semantic_drift_scores = []
        
        for diff_info in diff_infos:
            file_breaking_changes, file_drift_score = self._analyze_file_diff(diff_info)
            breaking_changes.extend(file_breaking_changes)
            if file_drift_score is not None:
                semantic_drift_scores.append(file_drift_score)
        
        # Calculate overall scores
        overall_semantic_drift = np.mean(semantic_drift_scores) if semantic_drift_scores else 0.0
        complexity_score = self._calculate_complexity_score(diff_infos, stats)
        overall_risk_score = self._calculate_overall_risk(breaking_changes, overall_semantic_drift, complexity_score)
        
        return CommitBreakingChangeAnalysis(
            commit_hash=commit_hash,
            commit_message=commit.message.strip(),
            timestamp=commit.committed_datetime.isoformat(),
            author=commit.author.name,
            overall_risk_score=overall_risk_score,
            breaking_changes=breaking_changes,
            files_changed=files_changed,
            lines_added=stats['insertions'],
            lines_removed=stats['deletions'],
            complexity_score=complexity_score,
            semantic_drift_score=overall_semantic_drift
        )
    
    def analyze_commit_range(self, git_analyzer, start_commit: str, end_commit: str) -> List[CommitBreakingChangeAnalysis]:
        """Analyze range of commits (stub implementation)"""
        repo = git_analyzer.repo
        commits = list(repo.iter_commits(f"{start_commit}..{end_commit}"))
        
        results = []
        for commit in commits:
            try:
                analysis = self.analyze_commit(git_analyzer, commit.hexsha)
                results.append(analysis)
            except Exception as e:
                logger.warning(f"Failed to analyze commit {commit.hexsha}: {str(e)}")
                continue
        
        return results

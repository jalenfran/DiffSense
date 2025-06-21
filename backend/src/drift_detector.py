from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import json

from .git_analyzer import DiffInfo, GitAnalyzer
from .embedding_engine import SemanticAnalyzer, EmbeddingResult

@dataclass
class DriftEvent:
    """Represents a significant semantic drift event"""
    commit_hash: str
    timestamp: datetime
    drift_score: float
    change_magnitude: float
    file_path: str
    description: str
    commit_message: str
    added_lines: int
    removed_lines: int

@dataclass
class FeatureHistory:
    """Complete history and analysis of a feature/function"""
    feature_id: str
    file_path: str
    function_name: Optional[str]
    commits: List[DiffInfo]
    embeddings: List[EmbeddingResult]
    drift_events: List[DriftEvent]
    overall_drift: float
    change_summary: Dict[str, Any]

class DriftDetector:
    """Main class for detecting and analyzing semantic drift"""
    
    def __init__(self, git_analyzer: GitAnalyzer):
        self.git_analyzer = git_analyzer
        self.semantic_analyzer = SemanticAnalyzer()
        self.drift_threshold = 0.3  # Threshold for significant drift
    
    def analyze_file_drift(self, file_path: str, max_commits: int = 50) -> FeatureHistory:
        """Analyze semantic drift for an entire file"""
        # Get commits that modified the file
        commits = self.git_analyzer.get_commits_for_file(file_path, max_commits)
        
        if not commits:
            raise ValueError(f"No commits found for file: {file_path}")
        
        # Extract diff information
        all_diffs = []
        for commit in commits:
            diffs = self.git_analyzer.extract_diff_info(commit, file_path)
            all_diffs.extend(diffs)
        
        # Sort by date (oldest first)
        all_diffs.sort(key=lambda x: x.commit_date)
        
        # Generate embeddings
        embeddings = []
        for diff in all_diffs:
            embedding = self.semantic_analyzer.generate_hybrid_embedding(
                (diff.added_lines, diff.removed_lines),
                diff.commit_message
            )
            embeddings.append(embedding)
        
        # Detect drift events
        drift_events = self._detect_drift_events(all_diffs, embeddings)
        
        # Calculate overall drift
        overall_drift = self._calculate_overall_drift(embeddings)
        
        # Generate change summary
        change_summary = self.semantic_analyzer.analyze_change_patterns(embeddings)
        
        return FeatureHistory(
            feature_id=f"file_{file_path.replace('/', '_')}",
            file_path=file_path,
            function_name=None,
            commits=all_diffs,
            embeddings=embeddings,
            drift_events=drift_events,
            overall_drift=overall_drift,
            change_summary=change_summary
        )
    
    def analyze_function_drift(self, file_path: str, function_name: str, max_commits: int = 30) -> FeatureHistory:
        """Analyze semantic drift for a specific function"""
        # Get commits that modified the function
        commits = self.git_analyzer.get_commits_for_function(file_path, function_name, max_commits)
        
        if not commits:
            raise ValueError(f"No commits found for function {function_name} in {file_path}")
        
        # Extract function-specific changes
        function_diffs = []
        for commit in commits:
            # Get the function content at this commit
            function_content = self.git_analyzer.get_function_content_at_commit(
                commit.hexsha, file_path, function_name
            )
            
            if function_content:
                # Create a simplified diff info for the function
                diff_info = DiffInfo(
                    commit_hash=commit.hexsha,
                    commit_message=commit.message.strip(),
                    commit_date=datetime.fromtimestamp(commit.committed_date),
                    author=commit.author.name,
                    file_path=file_path,
                    added_lines=[function_content],  # Simplified: treat as added content
                    removed_lines=[],
                    change_type='M',
                    lines_added=len(function_content.split('\n')),
                    lines_removed=0
                )
                function_diffs.append(diff_info)
        
        # Sort by date (oldest first)
        function_diffs.sort(key=lambda x: x.commit_date)
        
        # Generate embeddings for function versions
        embeddings = []
        for i, diff in enumerate(function_diffs):
            if i == 0:
                # First version - treat as entirely new
                embedding = self.semantic_analyzer.generate_hybrid_embedding(
                    (diff.added_lines, []),
                    diff.commit_message
                )
            else:
                # Compare with previous version
                prev_content = function_diffs[i-1].added_lines
                curr_content = diff.added_lines
                embedding = self.semantic_analyzer.generate_hybrid_embedding(
                    (curr_content, prev_content),
                    diff.commit_message
                )
            embeddings.append(embedding)
        
        # Detect drift events
        drift_events = self._detect_drift_events(function_diffs, embeddings)
        
        # Calculate overall drift
        overall_drift = self._calculate_overall_drift(embeddings)
        
        # Generate change summary
        change_summary = self.semantic_analyzer.analyze_change_patterns(embeddings)
        
        return FeatureHistory(
            feature_id=f"function_{file_path.replace('/', '_')}_{function_name}",
            file_path=file_path,
            function_name=function_name,
            commits=function_diffs,
            embeddings=embeddings,
            drift_events=drift_events,
            overall_drift=overall_drift,
            change_summary=change_summary
        )
    
    def _detect_drift_events(self, diffs: List[DiffInfo], embeddings: List[EmbeddingResult]) -> List[DriftEvent]:
        """Detect significant drift events"""
        drift_events = []
        
        if len(embeddings) < 2:
            return drift_events
        
        for i in range(1, len(embeddings)):
            # Calculate drift from previous commit
            similarity = self.semantic_analyzer.calculate_similarity(
                embeddings[i-1].embedding,
                embeddings[i].embedding
            )
            drift_score = 1 - similarity
            
            # Calculate change magnitude
            change_magnitude = self.semantic_analyzer.calculate_change_magnitude(
                embeddings[i-1].embedding,
                embeddings[i].embedding
            )
            
            # Check if this is a significant drift
            if drift_score > self.drift_threshold:
                diff_info = diffs[i]
                
                drift_event = DriftEvent(
                    commit_hash=diff_info.commit_hash,
                    timestamp=diff_info.commit_date,
                    drift_score=drift_score,
                    change_magnitude=change_magnitude,
                    file_path=diff_info.file_path,
                    description=self._generate_drift_description(drift_score, change_magnitude, diff_info),
                    commit_message=diff_info.commit_message,
                    added_lines=diff_info.lines_added,
                    removed_lines=diff_info.lines_removed
                )
                drift_events.append(drift_event)
        
        return drift_events
    
    def _generate_drift_description(self, drift_score: float, change_magnitude: float, diff_info: DiffInfo) -> str:
        """Generate human-readable description of drift event"""
        if drift_score > 0.7:
            severity = "major"
        elif drift_score > 0.5:
            severity = "significant"
        else:
            severity = "moderate"
        
        return f"Detected {severity} semantic drift (score: {drift_score:.2f}) in {diff_info.file_path}. " \
               f"{diff_info.lines_added} lines added, {diff_info.lines_removed} lines removed."
    
    def _calculate_overall_drift(self, embeddings: List[EmbeddingResult]) -> float:
        """Calculate overall drift from first to last commit"""
        if len(embeddings) < 2:
            return 0.0
        
        similarity = self.semantic_analyzer.calculate_similarity(
            embeddings[0].embedding,
            embeddings[-1].embedding
        )
        return 1 - similarity
    
    def generate_drift_timeline(self, feature_history: FeatureHistory) -> List[Dict[str, Any]]:
        """Generate timeline data for visualization"""
        timeline = []
        
        if len(feature_history.embeddings) < 2:
            return timeline
        
        # Calculate cumulative drift scores
        for i in range(1, len(feature_history.embeddings)):
            similarity = self.semantic_analyzer.calculate_similarity(
                feature_history.embeddings[0].embedding,  # Compare to original
                feature_history.embeddings[i].embedding
            )
            drift_score = 1 - similarity
            
            commit_info = feature_history.commits[i]
            
            timeline_point = {
                'timestamp': commit_info.commit_date.isoformat(),
                'commit_hash': commit_info.commit_hash,
                'commit_message': commit_info.commit_message,
                'author': commit_info.author,
                'drift_score': drift_score,
                'lines_added': commit_info.lines_added,
                'lines_removed': commit_info.lines_removed,
                'is_significant': drift_score > self.drift_threshold
            }
            timeline.append(timeline_point)
        
        return timeline
    
    def predict_breaking_change_risk(self, feature_history: FeatureHistory) -> Dict[str, Any]:
        """Predict if recent changes might be breaking (simplified heuristic)"""
        if not feature_history.drift_events:
            return {
                'risk_level': 'low',
                'risk_score': 0.1,
                'reasoning': 'No significant semantic drift detected'
            }
        
        recent_events = [e for e in feature_history.drift_events 
                        if (datetime.now() - e.timestamp).days <= 30]
        
        if not recent_events:
            return {
                'risk_level': 'low',
                'risk_score': 0.2,
                'reasoning': 'No recent significant changes'
            }
        
        # Calculate risk based on recent drift events
        max_recent_drift = max(event.drift_score for event in recent_events)
        avg_recent_drift = sum(event.drift_score for event in recent_events) / len(recent_events)
        
        # Simple heuristic for breaking change prediction
        risk_score = min(1.0, (max_recent_drift * 0.6) + (avg_recent_drift * 0.4))
        
        if risk_score > 0.7:
            risk_level = 'high'
            reasoning = f'High semantic drift detected in {len(recent_events)} recent commits'
        elif risk_score > 0.4:
            risk_level = 'medium'
            reasoning = f'Moderate semantic drift detected in recent changes'
        else:
            risk_level = 'low'
            reasoning = 'Recent changes show low semantic drift'
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'reasoning': reasoning,
            'recent_events_count': len(recent_events),
            'max_recent_drift': max_recent_drift
        }

def example_usage():
    """Example of how to use DriftDetector"""
    # Initialize with a git repository
    git_analyzer = GitAnalyzer("/path/to/your/repo")
    detector = DriftDetector(git_analyzer)
    
    try:
        # Analyze file drift
        file_history = detector.analyze_file_drift("src/main.py")
        print(f"File drift analysis:")
        print(f"- Overall drift: {file_history.overall_drift:.3f}")
        print(f"- Drift events: {len(file_history.drift_events)}")
        print(f"- Total commits: {len(file_history.commits)}")
        
        # Generate timeline
        timeline = detector.generate_drift_timeline(file_history)
        print(f"Timeline points: {len(timeline)}")
        
        # Predict breaking change risk
        risk_assessment = detector.predict_breaking_change_risk(file_history)
        print(f"Breaking change risk: {risk_assessment}")
        
    except ValueError as e:
        print(f"Analysis error: {e}")

if __name__ == "__main__":
    example_usage()

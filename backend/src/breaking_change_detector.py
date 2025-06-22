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
            file_breaking_changes, file_drift_score = self._analyze_file_diff(diff_info, git_analyzer)
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
    
    def _analyze_file_diff(self, diff_info, git_analyzer):
        """Analyze a single file diff for breaking changes"""
        breaking_changes = []
        
        file_path = diff_info.file_path
        commit_hash = diff_info.commit_hash
        
        # Get file content before and after the change
        commit = git_analyzer.repo.commit(commit_hash)
        parent = commit.parents[0] if commit.parents else None
        
        # Get before content (from parent commit)
        before_content = ""
        if parent:
            before_content = git_analyzer.get_file_content_at_commit(parent.hexsha, file_path) or ""
        
        # Get after content (from current commit)
        after_content = git_analyzer.get_file_content_at_commit(commit_hash, file_path) or ""
        
        # Language-specific analysis
        language = self._detect_language(file_path)
        
        # Analyze different types of breaking changes
        breaking_changes.extend(self._detect_api_changes(before_content, after_content, file_path, language))
        breaking_changes.extend(self._detect_signature_changes(before_content, after_content, file_path, language))
        breaking_changes.extend(self._detect_behavior_changes(before_content, after_content, file_path, language))
        breaking_changes.extend(self._detect_dependency_changes(before_content, after_content, file_path, language))
        
        # Calculate semantic drift score
        semantic_drift = self._calculate_semantic_drift(before_content, after_content)
        
        return breaking_changes, semantic_drift
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        ext_to_lang = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.cs': 'csharp',
            '.swift': 'swift',
            '.kt': 'kotlin'
        }
        
        import os
        ext = os.path.splitext(file_path)[1].lower()
        return ext_to_lang.get(ext, 'unknown')
    
    def _detect_api_changes(self, before: str, after: str, file_path: str, language: str) -> List[BreakingChange]:
        """Detect API-level breaking changes"""
        changes = []
        
        if language == 'python':
            changes.extend(self._detect_python_api_changes(before, after, file_path))
        elif language in ['javascript', 'typescript']:
            changes.extend(self._detect_js_api_changes(before, after, file_path))
        elif language == 'java':
            changes.extend(self._detect_java_api_changes(before, after, file_path))
        
        return changes
    
    def _detect_python_api_changes(self, before: str, after: str, file_path: str) -> List[BreakingChange]:
        """Detect Python API changes"""
        changes = []
        
        try:
            # Parse both versions
            before_tree = ast.parse(before) if before.strip() else None
            after_tree = ast.parse(after) if after.strip() else None
            
            if not before_tree or not after_tree:
                return changes
            
            # Extract functions and classes
            before_funcs = self._extract_python_functions(before_tree)
            after_funcs = self._extract_python_functions(after_tree)
            
            before_classes = self._extract_python_classes(before_tree)
            after_classes = self._extract_python_classes(after_tree)
            
            # Check for removed functions
            for func_name in before_funcs:
                if func_name not in after_funcs:
                    changes.append(BreakingChange(
                        change_type=ChangeType.API_CHANGE,
                        risk_level=RiskLevel.HIGH,
                        confidence=0.9,
                        file_path=file_path,
                        line_number=before_funcs[func_name].get('line'),
                        description=f"Function '{func_name}' was removed",
                        before_code=before_funcs[func_name].get('code'),
                        after_code=None,
                        impact_analysis={
                            'type': 'function_removal',
                            'affected_function': func_name,
                            'potential_impact': 'high'
                        },
                        mitigation_suggestions=[
                            f"Add deprecation warning before removing {func_name}",
                            "Provide migration guide for replacement function",
                            "Consider keeping function with compatibility wrapper"
                        ],
                        related_files=[]
                    ))
            
            # Check for removed classes
            for class_name in before_classes:
                if class_name not in after_classes:
                    changes.append(BreakingChange(
                        change_type=ChangeType.API_CHANGE,
                        risk_level=RiskLevel.CRITICAL,
                        confidence=0.95,
                        file_path=file_path,
                        line_number=before_classes[class_name].get('line'),
                        description=f"Class '{class_name}' was removed",
                        before_code=before_classes[class_name].get('code'),
                        after_code=None,
                        impact_analysis={
                            'type': 'class_removal',
                            'affected_class': class_name,
                            'potential_impact': 'critical'
                        },
                        mitigation_suggestions=[
                            f"Add deprecation warning before removing {class_name}",
                            "Provide migration guide for replacement class",
                            "Consider inheritance-based compatibility"
                        ],
                        related_files=[]
                    ))
            
        except SyntaxError:
            # If we can't parse, fall back to text-based analysis
            changes.extend(self._detect_text_based_changes(before, after, file_path))
        
        return changes
    
    def _detect_signature_changes(self, before: str, after: str, file_path: str, language: str) -> List[BreakingChange]:
        """Detect function signature changes"""
        changes = []
        
        if language == 'python':
            changes.extend(self._detect_python_signature_changes(before, after, file_path))
        
        return changes
    
    def _detect_python_signature_changes(self, before: str, after: str, file_path: str) -> List[BreakingChange]:
        """Detect Python function signature changes"""
        changes = []
        
        try:
            before_tree = ast.parse(before) if before.strip() else None
            after_tree = ast.parse(after) if after.strip() else None
            
            if not before_tree or not after_tree:
                return changes
            
            before_funcs = self._extract_python_functions(before_tree)
            after_funcs = self._extract_python_functions(after_tree)
            
            # Check for signature changes
            for func_name in before_funcs:
                if func_name in after_funcs:
                    before_sig = before_funcs[func_name].get('signature', '')
                    after_sig = after_funcs[func_name].get('signature', '')
                    
                    if before_sig != after_sig:
                        # Analyze the type of signature change
                        risk_level = self._assess_signature_change_risk(before_sig, after_sig)
                        
                        changes.append(BreakingChange(
                            change_type=ChangeType.SIGNATURE_CHANGE,
                            risk_level=risk_level,
                            confidence=0.8,
                            file_path=file_path,
                            line_number=after_funcs[func_name].get('line'),
                            description=f"Function '{func_name}' signature changed",
                            before_code=f"def {func_name}{before_sig}:",
                            after_code=f"def {func_name}{after_sig}:",
                            impact_analysis={
                                'type': 'signature_change',
                                'function': func_name,
                                'before_signature': before_sig,
                                'after_signature': after_sig
                            },
                            mitigation_suggestions=[
                                "Use keyword arguments for backward compatibility",
                                "Add default values for new parameters",
                                "Provide wrapper function for old signature"
                            ],
                            related_files=[]
                        ))
        
        except SyntaxError:
            pass
        
        return changes
    
    def _detect_behavior_changes(self, before: str, after: str, file_path: str, language: str) -> List[BreakingChange]:
        """Detect behavior changes that might break existing code"""
        changes = []
        
        # Look for changes in return statements, exceptions, etc.
        if language == 'python':
            changes.extend(self._detect_python_behavior_changes(before, after, file_path))
        
        return changes
    
    def _detect_python_behavior_changes(self, before: str, after: str, file_path: str) -> List[BreakingChange]:
        """Detect Python behavior changes"""
        changes = []
        
        # Check for new exceptions being raised
        before_exceptions = re.findall(r'raise\s+(\w+)', before)
        after_exceptions = re.findall(r'raise\s+(\w+)', after)
        
        new_exceptions = set(after_exceptions) - set(before_exceptions)
        
        for exception in new_exceptions:
            changes.append(BreakingChange(
                change_type=ChangeType.BEHAVIOR_CHANGE,
                risk_level=RiskLevel.MEDIUM,
                confidence=0.7,
                file_path=file_path,
                line_number=None,
                description=f"New exception '{exception}' is now raised",
                before_code=None,
                after_code=f"raise {exception}",
                impact_analysis={
                    'type': 'new_exception',
                    'exception_type': exception
                },
                mitigation_suggestions=[
                    "Document the new exception in API documentation",
                    "Update error handling in dependent code",
                    "Consider gradual rollout with warnings"
                ],
                related_files=[]
            ))
        
        return changes
    
    def _detect_dependency_changes(self, before: str, after: str, file_path: str, language: str) -> List[BreakingChange]:
        """Detect dependency changes"""
        changes = []
        
        if language == 'python':
            before_imports = re.findall(r'from\s+(\S+)\s+import|import\s+(\S+)', before)
            after_imports = re.findall(r'from\s+(\S+)\s+import|import\s+(\S+)', after)
            
            before_deps = set([imp[0] or imp[1] for imp in before_imports])
            after_deps = set([imp[0] or imp[1] for imp in after_imports])
            
            removed_deps = before_deps - after_deps
            added_deps = after_deps - before_deps
            
            for dep in removed_deps:
                changes.append(BreakingChange(
                    change_type=ChangeType.DEPENDENCY_CHANGE,
                    risk_level=RiskLevel.MEDIUM,
                    confidence=0.6,
                    file_path=file_path,
                    line_number=None,
                    description=f"Dependency '{dep}' was removed",
                    before_code=f"import {dep}",
                    after_code=None,
                    impact_analysis={
                        'type': 'dependency_removal',
                        'dependency': dep
                    },
                    mitigation_suggestions=[
                        "Update requirements.txt",
                        "Check if dependency is still needed",
                        "Update documentation"
                    ],
                    related_files=[]
                ))
        
        return changes
    
    def _calculate_semantic_drift(self, before: str, after: str) -> float:
        """Calculate semantic drift between two code versions"""
        try:
            before_embedding = self.semantic_analyzer.get_embedding(before)
            after_embedding = self.semantic_analyzer.get_embedding(after)
            
            if before_embedding is not None and after_embedding is not None:
                similarity = np.dot(before_embedding, after_embedding) / (
                    np.linalg.norm(before_embedding) * np.linalg.norm(after_embedding)
                )
                drift_score = 1.0 - similarity
                return max(0.0, min(1.0, drift_score))
        
        except Exception as e:
            logger.warning(f"Error calculating semantic drift: {e}")
        
        return 0.0
    
    def _calculate_complexity_score(self, diff_infos, stats) -> float:
        """Calculate complexity score based on changes"""
        total_lines_changed = stats.get('insertions', 0) + stats.get('deletions', 0)
        file_count = len(diff_infos)
        
        # Normalize complexity score
        complexity = (total_lines_changed * 0.1 + file_count * 0.2) / 10.0
        return min(1.0, complexity)
    
    def _calculate_overall_risk(self, breaking_changes, semantic_drift, complexity) -> float:
        """Calculate overall risk score"""
        if not breaking_changes:
            return semantic_drift * 0.5 + complexity * 0.3
        
        # Weight by risk levels
        risk_weights = {
            RiskLevel.LOW: 0.2,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.HIGH: 0.8,
            RiskLevel.CRITICAL: 1.0
        }
        
        max_risk = max(risk_weights[change.risk_level] for change in breaking_changes)
        avg_confidence = np.mean([change.confidence for change in breaking_changes])
        
        overall_risk = (max_risk * 0.5 + 
                       semantic_drift * 0.3 + 
                       complexity * 0.2) * avg_confidence
        
        return min(1.0, overall_risk)
    
    def _extract_python_functions(self, tree) -> Dict[str, Dict]:
        """Extract function definitions from Python AST"""
        functions = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Extract signature
                args = [arg.arg for arg in node.args.args]
                defaults = [ast.unparse(default) if hasattr(ast, 'unparse') else 'default' 
                           for default in node.args.defaults]
                
                signature = f"({', '.join(args)})"
                
                functions[node.name] = {
                    'line': node.lineno,
                    'signature': signature,
                    'code': f"def {node.name}{signature}:"
                }
        
        return functions
    
    def _extract_python_classes(self, tree) -> Dict[str, Dict]:
        """Extract class definitions from Python AST"""
        classes = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes[node.name] = {
                    'line': node.lineno,
                    'code': f"class {node.name}:"
                }
        
        return classes
    
    def _detect_js_api_changes(self, before: str, after: str, file_path: str) -> List[BreakingChange]:
        """Detect JavaScript API changes (simplified)"""
        changes = []
        
        # Simple regex-based detection for JS functions
        before_functions = re.findall(r'function\s+(\w+)\s*\(', before)
        after_functions = re.findall(r'function\s+(\w+)\s*\(', after)
        
        removed_functions = set(before_functions) - set(after_functions)
        
        for func_name in removed_functions:
            changes.append(BreakingChange(
                change_type=ChangeType.API_CHANGE,
                risk_level=RiskLevel.HIGH,
                confidence=0.7,
                file_path=file_path,
                line_number=None,
                description=f"JavaScript function '{func_name}' was removed",
                before_code=f"function {func_name}(...)",
                after_code=None,
                impact_analysis={'type': 'js_function_removal'},
                mitigation_suggestions=["Add deprecation warning", "Provide migration guide"],
                related_files=[]
            ))
        
        return changes
    
    def _detect_java_api_changes(self, before: str, after: str, file_path: str) -> List[BreakingChange]:
        """Detect Java API changes (simplified)"""
        changes = []
        
        # Simple regex-based detection for Java methods
        before_methods = re.findall(r'public\s+\w+\s+(\w+)\s*\(', before)
        after_methods = re.findall(r'public\s+\w+\s+(\w+)\s*\(', after)
        
        removed_methods = set(before_methods) - set(after_methods)
        
        for method_name in removed_methods:
            changes.append(BreakingChange(
                change_type=ChangeType.API_CHANGE,
                risk_level=RiskLevel.HIGH,
                confidence=0.7,
                file_path=file_path,
                line_number=None,
                description=f"Java method '{method_name}' was removed",
                before_code=f"public ... {method_name}(...)",
                after_code=None,
                impact_analysis={'type': 'java_method_removal'},
                mitigation_suggestions=["Add @Deprecated annotation", "Provide replacement method"],
                related_files=[]
            ))
        
        return changes
    
    def _detect_text_based_changes(self, before: str, after: str, file_path: str) -> List[BreakingChange]:
        """Fallback text-based analysis when parsing fails"""
        changes = []
        
        # Look for removed function/method definitions
        function_patterns = [
            r'def\s+(\w+)\s*\(',  # Python
            r'function\s+(\w+)\s*\(',  # JavaScript
            r'public\s+\w+\s+(\w+)\s*\('  # Java
        ]
        
        for pattern in function_patterns:
            before_matches = set(re.findall(pattern, before))
            after_matches = set(re.findall(pattern, after))
            
            removed = before_matches - after_matches
            
            for func_name in removed:
                changes.append(BreakingChange(
                    change_type=ChangeType.API_CHANGE,
                    risk_level=RiskLevel.MEDIUM,
                    confidence=0.5,
                    file_path=file_path,
                    line_number=None,
                    description=f"Function/method '{func_name}' may have been removed",
                    before_code=None,
                    after_code=None,
                    impact_analysis={'type': 'text_based_detection'},
                    mitigation_suggestions=["Manual verification needed"],
                    related_files=[]
                ))
        
        return changes
    
    def _assess_signature_change_risk(self, before_sig: str, after_sig: str) -> RiskLevel:
        """Assess the risk level of a signature change"""
        # Count parameters
        before_params = before_sig.count(',') + 1 if before_sig.strip('()') else 0
        after_params = after_sig.count(',') + 1 if after_sig.strip('()') else 0
        
        if after_params < before_params:
            return RiskLevel.HIGH  # Removed parameters
        elif after_params > before_params:
            return RiskLevel.MEDIUM  # Added parameters (might have defaults)
        else:
            return RiskLevel.LOW  # Same number, might be renamed

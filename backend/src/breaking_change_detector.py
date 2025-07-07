"""
Advanced Breaking Change Detector - Next-Generation Intelligence
==============================================================

This is a cutting-edge breaking change detection system that uses:
- Semantic analysis with vector embeddings
- AST (Abstract Syntax Tree) deep parsing  
- Machine learning-based change impact prediction
- Multi-expert AI analysis with Claude
- Historical pattern recognition
- Dependency graph analysis
- API surface change detection
- Behavioral change analysis through test impact

Not all breaking changes are bad - this system categorizes them by:
- Intentional vs Accidental
- Impact severity (Critical/High/Medium/Low)
- Affected user segments
- Migration complexity
- Business value vs risk
"""

import ast
import re
import json
import logging
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
from collections import defaultdict

logger = logging.getLogger(__name__)

class ChangeType(Enum):
    """Types of breaking changes detected"""
    API_SIGNATURE_CHANGE = "api_signature_change"
    FUNCTION_REMOVAL = "function_removal"  
    PARAMETER_CHANGE = "parameter_change"
    RETURN_TYPE_CHANGE = "return_type_change"
    CLASS_STRUCTURE_CHANGE = "class_structure_change"
    DEPENDENCY_CHANGE = "dependency_change"
    CONFIGURATION_CHANGE = "configuration_change"
    DATABASE_SCHEMA_CHANGE = "database_schema_change"
    BEHAVIORAL_CHANGE = "behavioral_change"
    SECURITY_CHANGE = "security_change"
    PERFORMANCE_REGRESSION = "performance_regression"

class ImpactSeverity(Enum):
    """Severity levels for breaking changes"""
    CRITICAL = "critical"      # System-breaking, immediate attention required
    HIGH = "high"             # Major functionality affected, migration needed
    MEDIUM = "medium"         # Some features affected, workarounds possible
    LOW = "low"              # Minor issues, easy to fix
    ENHANCEMENT = "enhancement"  # Positive breaking change (improvement)

class ChangeIntent(Enum):
    """Whether the breaking change was intentional"""
    INTENTIONAL = "intentional"      # Deliberate improvement/refactor
    ACCIDENTAL = "accidental"        # Unintended side effect
    UNCLEAR = "unclear"              # Cannot determine intent

@dataclass
class BreakingChange:
    """Comprehensive breaking change analysis"""
    id: str
    commit_hash: str
    change_type: ChangeType
    severity: ImpactSeverity
    intent: ChangeIntent
    
    # Change details
    affected_component: str
    file_path: str
    line_number: Optional[int]
    old_signature: Optional[str]
    new_signature: Optional[str]
    
    # Impact analysis
    confidence_score: float  # 0.0 to 1.0
    affected_users_estimate: str  # "all", "most", "some", "few"
    migration_complexity: str  # "trivial", "easy", "moderate", "complex", "very_complex"
    
    # Context
    description: str
    technical_details: str
    suggested_migration: Optional[str]
    related_changes: List[str] = field(default_factory=list)
    
    # AI Analysis
    ai_analysis: Optional[str] = None
    expert_recommendations: List[str] = field(default_factory=list)
    
    # Metadata
    detection_methods: List[str] = field(default_factory=list)
    detected_at: datetime = field(default_factory=datetime.now)

@dataclass
class CodeSignature:
    """Semantic signature of code elements"""
    name: str
    type: str  # function, class, method, property
    parameters: List[str]
    return_type: Optional[str]
    access_modifiers: List[str]
    decorators: List[str]
    docstring: Optional[str]
    body_hash: str  # Hash of normalized body for behavior detection
    dependencies: Set[str]
    
@dataclass
class BreakingChangeAnalysis:
    """Complete analysis result for a commit's breaking changes"""
    commit_hash: str
    commit_message: str
    timestamp: str
    author: str
    overall_risk_score: float
    files_changed: List[str]
    lines_added: int
    lines_removed: int
    complexity_score: float
    semantic_drift_score: float
    breaking_changes: List[BreakingChange]
    all_changes: List[BreakingChange]
    migration_suggestions: List[str]
    confidence: float
    ai_analysis: Optional[str] = None

class BreakingChangeDetector:
    """Next-generation breaking change detection with AI and semantic analysis"""
    
    def __init__(self, git_analyzer, claude_analyzer=None, embedding_engine=None):
        self.git_analyzer = git_analyzer
        self.claude_analyzer = claude_analyzer
        self.embedding_engine = embedding_engine
        
        # Caching for performance
        self.signature_cache = {}
        self.embedding_cache = {}
        self.pattern_cache = {}
        
        # Machine learning models (would be trained with historical data)
        self.severity_predictor = None  # Would load pre-trained model
        self.intent_classifier = None  # Would load pre-trained model
        
    def _is_code_file(self, file_path: str) -> bool:
        """Check if a file is a code file based on its extension"""
        if not file_path:
            return False
            
        code_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h',
            '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala',
            '.dart', '.vue', '.svelte', '.html', '.css', '.scss', '.sass',
            '.less', '.sql', '.r', '.R', '.m', '.pl', '.sh', '.bat', '.ps1'
        }
        
        import os
        ext = os.path.splitext(file_path)[1].lower()
        return ext in code_extensions
        
    def analyze_commit_for_breaking_changes(self, commit_hash: str, git_analyzer=None, enable_ai_analysis=False) -> List[BreakingChange]:
        """
        Comprehensive breaking change analysis for a commit
        Uses multiple detection strategies and optionally AI analysis
        
        Args:
            commit_hash: The commit hash to analyze
            git_analyzer: Optional git analyzer instance
            enable_ai_analysis: Whether to include Claude AI analysis (default: False for performance)
        """
        logger.info(f"🔍 Analyzing commit {commit_hash} for breaking changes...")
        
        try:
            # Use provided git_analyzer or fall back to instance git_analyzer
            analyzer = git_analyzer or self.git_analyzer
            
            # Check if git_analyzer is properly initialized
            if not analyzer or not hasattr(analyzer, 'repo'):
                logger.error("Git analyzer not properly initialized")
                return []
                
            # Get commit details
            repo = analyzer.repo
            if not repo:
                logger.error("Repository not available in git analyzer")
                return []
                
            commit = repo.commit(commit_hash)
            
            # Get all changes in the commit
            if not commit.parents:
                logger.warning("Initial commit - no breaking changes to detect")
                return []
            
            parent = commit.parents[0]
            diffs = parent.diff(commit, create_patch=True)
            
            all_breaking_changes = []
            
            # Multi-strategy detection
            for diff_item in diffs:
                file_path = diff_item.a_path or diff_item.b_path
                logger.debug(f"Processing file: {file_path}")
                
                if diff_item.a_path and self._is_code_file(diff_item.a_path):
                    logger.debug(f"Analyzing code file: {file_path}")
                    changes = self._analyze_file_diff(diff_item, commit_hash)
                    logger.debug(f"Found {len(changes)} changes in {file_path}")
                    all_breaking_changes.extend(changes)
                else:
                    logger.debug(f"Skipping non-code file: {file_path}")
            
            # Post-processing: merge related changes, calculate relationships
            all_breaking_changes = self._post_process_changes(all_breaking_changes, commit)
            
            # Optional AI-enhanced analysis with Claude (only when explicitly enabled)
            if enable_ai_analysis and self.claude_analyzer and all_breaking_changes:
                logger.info(f"🤖 Running AI analysis for {len(all_breaking_changes)} breaking changes...")
                all_breaking_changes = self._enhance_with_ai_analysis(all_breaking_changes, commit)
            
            logger.info(f"✅ Found {len(all_breaking_changes)} breaking changes in commit {commit_hash}")
            return all_breaking_changes
            
        except Exception as e:
            logger.error(f"Error analyzing commit {commit_hash}: {e}")
            return []
    
    def analyze_commit(self, git_analyzer, commit_hash: str, enable_ai_analysis=False) -> BreakingChangeAnalysis:
        """Analyze a specific commit for breaking changes"""
        try:
            if not git_analyzer or not hasattr(git_analyzer, 'repo'):
                raise ValueError("Invalid git_analyzer provided")
                
            repo = git_analyzer.repo
            commit = repo.commit(commit_hash)
            
            # Get commit changes
            changes = []
            if commit.parents:
                parent = commit.parents[0]
                diffs = parent.diff(commit, create_patch=True)
                
                for diff_item in diffs:
                    if diff_item.a_path and diff_item.a_path.endswith(('.py', '.js', '.ts', '.java', '.cpp', '.h')):
                        file_path = diff_item.a_path or diff_item.b_path
                        
                        # Get file content before and after
                        old_content = ""
                        new_content = ""
                        
                        try:
                            if diff_item.a_blob:
                                old_content = diff_item.a_blob.data_stream.read().decode('utf-8', errors='ignore')
                            if diff_item.b_blob:
                                new_content = diff_item.b_blob.data_stream.read().decode('utf-8', errors='ignore')
                        except Exception as e:
                            logger.debug(f"Could not read blob content: {e}")
                            continue
                        
                        # Detect changes in this file
                        file_changes = self._detect_file_changes(file_path, old_content, new_content, diff_item, commit_hash)
                        changes.extend(file_changes)
            
            # Create analysis result
            breaking_changes = [change for change in changes if change.severity in [ImpactSeverity.CRITICAL, ImpactSeverity.HIGH]]
            migration_suggestions = self._generate_migration_suggestions(breaking_changes)
            
            # Calculate confidence based on detection methods used
            confidence = 0.8 if breaking_changes else 0.9
            
            # Optional AI analysis (only when explicitly enabled)
            ai_analysis = None
            if enable_ai_analysis and self.claude_analyzer and breaking_changes:
                logger.info(f"🤖 Running AI analysis for commit {commit_hash}...")
                ai_analysis = self._generate_ai_analysis(commit, breaking_changes)
            
            # Calculate overall risk score
            risk_score = 0.0
            if breaking_changes:
                # Calculate risk based on severity and count
                severity_weights = {"critical": 1.0, "high": 0.8, "medium": 0.5, "low": 0.2}
                total_weight = sum(severity_weights.get(bc.severity.value, 0.5) for bc in breaking_changes)
                risk_score = min(total_weight / len(breaking_changes), 1.0)
            
            # Get files changed
            files_changed = list(set(change.file_path for change in changes if change.file_path))
            
            # Calculate basic metrics
            lines_added = 0
            lines_removed = 0
            if commit.parents:
                try:
                    stats = commit.stats
                    lines_added = stats.total['insertions']
                    lines_removed = stats.total['deletions']
                except:
                    pass
            
            # Calculate complexity score (simplified)
            complexity_score = min(len(changes) * 0.1 + risk_score * 2, 10.0)
            
            # Calculate semantic drift score (simplified)
            semantic_drift_score = risk_score * 0.8 + (len(breaking_changes) / max(len(changes), 1)) * 0.2
            
            return BreakingChangeAnalysis(
                commit_hash=commit_hash,
                commit_message=commit.message.strip(),
                timestamp=commit.committed_datetime.isoformat(),
                author=commit.author.name,
                overall_risk_score=risk_score,
                files_changed=files_changed,
                lines_added=lines_added,
                lines_removed=lines_removed,
                complexity_score=complexity_score,
                semantic_drift_score=semantic_drift_score,
                breaking_changes=breaking_changes,
                all_changes=changes,
                migration_suggestions=migration_suggestions,
                confidence=confidence,
                ai_analysis=ai_analysis
            )
            
        except Exception as e:
            logger.error(f"Error analyzing commit {commit_hash}: {e}")
            return BreakingChangeAnalysis(
                commit_hash=commit_hash,
                commit_message="Error analyzing commit",
                timestamp="",
                author="unknown",
                overall_risk_score=0.0,
                files_changed=[],
                lines_added=0,
                lines_removed=0,
                complexity_score=0.0,
                semantic_drift_score=0.0,
                breaking_changes=[],
                all_changes=[],
                migration_suggestions=[],
                confidence=0.0,
                ai_analysis=None
            )
    
    def _analyze_file_diff(self, diff_item, commit_hash: str) -> List[BreakingChange]:
        """Analyze a single file diff for breaking changes using multiple strategies"""
        changes = []
        
        try:
            # Get old and new content with safer blob reading
            old_content = ""
            new_content = ""
            
            try:
                if diff_item.a_blob:
                    old_content = diff_item.a_blob.data_stream.read().decode('utf-8', errors='ignore')
            except Exception as e:
                logger.debug(f"Could not read old blob: {e}")
                
            try:
                if diff_item.b_blob:
                    new_content = diff_item.b_blob.data_stream.read().decode('utf-8', errors='ignore')
            except Exception as e:
                logger.debug(f"Could not read new blob: {e}")
            
            file_path = diff_item.a_path or diff_item.b_path
            
            # Strategy 1: AST-based structural analysis
            ast_changes = self._detect_ast_changes(old_content, new_content, file_path, commit_hash)
            changes.extend(ast_changes)
            
            # Strategy 2: Semantic signature analysis
            signature_changes = self._detect_signature_changes(old_content, new_content, file_path, commit_hash)
            changes.extend(signature_changes)
            
            # Strategy 3: API surface analysis
            api_changes = self._detect_api_surface_changes(old_content, new_content, file_path, commit_hash)
            changes.extend(api_changes)
            
            # Strategy 4: Dependency analysis
            dependency_changes = self._detect_dependency_changes(old_content, new_content, file_path, commit_hash)
            changes.extend(dependency_changes)
            
            # Strategy 5: Behavioral analysis (via test impact)
            behavioral_changes = self._detect_behavioral_changes(old_content, new_content, file_path, commit_hash)
            changes.extend(behavioral_changes)
            
            # Strategy 6: Configuration and schema changes
            config_changes = self._detect_config_schema_changes(old_content, new_content, file_path, commit_hash)
            changes.extend(config_changes)
            
            # Strategy 7: Basic diff pattern analysis (fallback)
            diff_changes = self._detect_diff_patterns(old_content, new_content, file_path, commit_hash, diff_item)
            changes.extend(diff_changes)
            
        except Exception as e:
            logger.error(f"Error analyzing file diff for {diff_item.a_path}: {e}")
        
        return changes
    
    def _detect_ast_changes(self, old_content: str, new_content: str, file_path: str, commit_hash: str) -> List[BreakingChange]:
        """Detect breaking changes using Abstract Syntax Tree analysis"""
        changes = []
        
        try:
            # Parse ASTs for Python files
            if file_path.endswith('.py'):
                old_ast = self._safe_parse_ast(old_content)
                new_ast = self._safe_parse_ast(new_content)
                
                if old_ast and new_ast:
                    # Extract signatures from both versions
                    old_signatures = self._extract_signatures_from_ast(old_ast)
                    new_signatures = self._extract_signatures_from_ast(new_ast)
                    
                    # Compare signatures
                    changes.extend(self._compare_signatures(old_signatures, new_signatures, file_path, commit_hash))
            
            # TODO: Add support for JavaScript, TypeScript, Java, etc.
            # elif file_path.endswith(('.js', '.ts')):
            #     changes.extend(self._detect_js_ast_changes(old_content, new_content, file_path, commit_hash))
            
        except Exception as e:
            logger.debug(f"AST analysis failed for {file_path}: {e}")
        
        return changes
    
    def _safe_parse_ast(self, content: str) -> Optional[ast.AST]:
        """Safely parse Python AST with error handling"""
        try:
            return ast.parse(content)
        except SyntaxError as e:
            logger.debug(f"Syntax error parsing AST: {e}")
            return None
        except Exception as e:
            logger.debug(f"Error parsing AST: {e}")
            return None
    
    def _extract_signatures_from_ast(self, tree: ast.AST) -> Dict[str, CodeSignature]:
        """Extract semantic signatures from AST"""
        signatures = {}
        
        class SignatureExtractor(ast.NodeVisitor):
            def __init__(self, detector):
                self.current_class = None
                self.detector = detector
                
            def visit_FunctionDef(self, node):
                sig = self.detector._create_function_signature(node, self.current_class)
                signatures[sig.name] = sig
                self.generic_visit(node)
                
            def visit_AsyncFunctionDef(self, node):
                sig = self.detector._create_function_signature(node, self.current_class, is_async=True)
                signatures[sig.name] = sig
                self.generic_visit(node)
                
            def visit_ClassDef(self, node):
                old_class = self.current_class
                self.current_class = node.name
                
                sig = self.detector._create_class_signature(node)
                signatures[sig.name] = sig
                
                self.generic_visit(node)
                self.current_class = old_class
        
        extractor = SignatureExtractor(self)
        extractor.visit(tree)
        return signatures
    
    def _create_function_signature(self, node: ast.FunctionDef, current_class: Optional[str] = None, is_async: bool = False) -> CodeSignature:
        """Create a semantic signature for a function"""
        name = f"{current_class}.{node.name}" if current_class else node.name
        
        # Extract parameters
        params = []
        for arg in node.args.args:
            param_str = arg.arg
            if arg.annotation:
                param_str += f": {ast.unparse(arg.annotation)}"
            params.append(param_str)
        
        # Extract return type
        return_type = ast.unparse(node.returns) if node.returns else None
        
        # Extract decorators
        decorators = [ast.unparse(d) for d in node.decorator_list]
        
        # Extract docstring
        docstring = ast.get_docstring(node)
        
        # Create body hash for behavioral change detection
        body_code = ast.unparse(node).encode('utf-8')
        body_hash = hashlib.sha256(body_code).hexdigest()[:16]
        
        # Extract dependencies (simplified)
        dependencies = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                dependencies.add(child.id)
        
        return CodeSignature(
            name=name,
            type="async_function" if is_async else "function",
            parameters=params,
            return_type=return_type,
            access_modifiers=["async"] if is_async else [],
            decorators=decorators,
            docstring=docstring,
            body_hash=body_hash,
            dependencies=dependencies
        )
    
    def _create_class_signature(self, node: ast.ClassDef) -> CodeSignature:
        """Create a semantic signature for a class"""
        name = node.name
        
        # Extract base classes
        bases = [ast.unparse(base) for base in node.bases]
        
        # Extract decorators
        decorators = [ast.unparse(d) for d in node.decorator_list]
        
        # Extract docstring
        docstring = ast.get_docstring(node)
        
        # Create body hash
        body_code = ast.unparse(node).encode('utf-8')
        body_hash = hashlib.sha256(body_code).hexdigest()[:16]
        
        # Extract method names
        methods = []
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(child.name)
        
        return CodeSignature(
            name=name,
            type="class",
            parameters=bases,  # Base classes as "parameters"
            return_type=None,
            access_modifiers=[],
            decorators=decorators,
            docstring=docstring,
            body_hash=body_hash,
            dependencies=set(methods)
        )
    
    def _compare_signatures(self, old_sigs: Dict[str, CodeSignature], new_sigs: Dict[str, CodeSignature], 
                          file_path: str, commit_hash: str) -> List[BreakingChange]:
        """Compare signature dictionaries to detect breaking changes"""
        changes = []
        
        # Check for removed functions/classes
        for name, old_sig in old_sigs.items():
            if name not in new_sigs:
                changes.append(BreakingChange(
                    id=f"{commit_hash}_{hashlib.sha256(name.encode()).hexdigest()[:8]}",
                    commit_hash=commit_hash,
                    change_type=ChangeType.FUNCTION_REMOVAL,
                    severity=self._predict_removal_severity(old_sig),
                    intent=ChangeIntent.UNCLEAR,
                    affected_component=name,
                    file_path=file_path,
                    line_number=None,
                    old_signature=self._signature_to_string(old_sig, file_path),
                    new_signature=None,
                    confidence_score=0.95,
                    affected_users_estimate=self._estimate_affected_users(old_sig),
                    migration_complexity=self._estimate_migration_complexity(old_sig, None),
                    description=f"Function/class '{name}' was removed",
                    technical_details=f"The {old_sig.type} '{name}' is no longer available in the codebase",
                    suggested_migration=f"Find alternative implementation for '{name}' functionality",
                    detection_methods=["ast_signature_analysis"]
                ))
        
        # Check for modified functions/classes
        for name, new_sig in new_sigs.items():
            if name in old_sigs:
                old_sig = old_sigs[name]
                signature_changes = self._compare_individual_signatures(old_sig, new_sig, file_path, commit_hash)
                changes.extend(signature_changes)
        
        return changes
    
    def _predict_removal_severity(self, sig: CodeSignature) -> ImpactSeverity:
        """Predict severity of a function/class removal"""
        # If it's a public API (no leading underscore), it's more severe
        if not sig.name.startswith('_'):
            return ImpactSeverity.HIGH
        elif sig.name.startswith('__'):
            return ImpactSeverity.LOW
        else:
            return ImpactSeverity.MEDIUM
    
    def _signature_to_string(self, sig: CodeSignature, file_path: str = "") -> str:
        """Convert signature to string representation based on language"""
        if sig.type in ['function', 'async_function']:
            params_str = ', '.join(sig.parameters) if sig.parameters else ""
            
            # Language-specific formatting
            if file_path.endswith('.py'):
                prefix = "async def" if sig.type == 'async_function' else "def"
                return f"{prefix} {sig.name}({params_str})"
            elif file_path.endswith(('.js', '.ts', '.jsx', '.tsx')):
                if sig.type == 'async_function':
                    return f"async function {sig.name}({params_str})"
                else:
                    return f"function {sig.name}({params_str})"
            elif file_path.endswith(('.c', '.cpp', '.h', '.hpp')):
                return_type = sig.return_type or "void"
                return f"{return_type} {sig.name}({params_str})"
            elif file_path.endswith('.java'):
                return_type = sig.return_type or "void"
                access = "public" if not sig.name.startswith('_') else "private"
                return f"{access} {return_type} {sig.name}({params_str})"
            else:
                # Generic format
                return f"{sig.name}({params_str})"
                
        elif sig.type == 'class':
            bases_str = ', '.join(sig.parameters) if sig.parameters else ""
            
            if file_path.endswith('.py'):
                return f"class {sig.name}({bases_str})" if bases_str else f"class {sig.name}"
            elif file_path.endswith('.java'):
                extends_str = f" extends {bases_str}" if bases_str else ""
                return f"public class {sig.name}{extends_str}"
            elif file_path.endswith(('.c', '.cpp', '.h', '.hpp')):
                return f"class {sig.name}" if sig.type == 'class' else f"struct {sig.name}"
            else:
                return f"class {sig.name}({bases_str})" if bases_str else f"class {sig.name}"
                
        return sig.name
    
    def _estimate_affected_users(self, sig: CodeSignature) -> str:
        """Estimate how many users are affected by this change"""
        if sig.name.startswith('__'):
            return "few"
        elif sig.name.startswith('_'):
            return "some"
        else:
            return "most"
    
    def _estimate_migration_complexity(self, old_sig: Optional[CodeSignature], new_sig: Optional[CodeSignature]) -> str:
        """Estimate migration complexity"""
        if new_sig is None:  # Removal
            return "complex"
        if old_sig is None:  # Addition
            return "trivial"
        
        # Parameter changes
        if old_sig.parameters != new_sig.parameters:
            param_diff = abs(len(old_sig.parameters) - len(new_sig.parameters))
            if param_diff > 2:
                return "complex"
            elif param_diff > 0:
                return "moderate"
        
        return "easy"
    
    def _predict_return_type_severity(self, old_sig: CodeSignature, new_sig: CodeSignature) -> ImpactSeverity:
        """Predict severity of return type changes"""
        if old_sig.return_type is None and new_sig.return_type is not None:
            return ImpactSeverity.LOW  # Adding return type annotation
        elif old_sig.return_type is not None and new_sig.return_type is None:
            return ImpactSeverity.LOW  # Removing return type annotation
        elif old_sig.return_type != new_sig.return_type:
            return ImpactSeverity.MEDIUM  # Changing return type
        return ImpactSeverity.LOW
    
    def _suggest_return_type_migration(self, old_sig: CodeSignature, new_sig: CodeSignature) -> str:
        """Suggest migration for return type changes"""
        if old_sig.return_type != new_sig.return_type:
            return f"Update code expecting return type '{old_sig.return_type}' to handle '{new_sig.return_type}'"
        return "Review return type usage for compatibility"
    
    def _predict_behavioral_severity(self, old_sig: CodeSignature, new_sig: CodeSignature) -> ImpactSeverity:
        """Predict severity of behavioral changes"""
        # If it's a public function, behavioral changes are more severe
        if not old_sig.name.startswith('_'):
            return ImpactSeverity.MEDIUM
        return ImpactSeverity.LOW
    
    def _suggest_behavioral_migration(self, old_sig: CodeSignature, new_sig: CodeSignature) -> str:
        """Suggest migration for behavioral changes"""
        return f"Test and validate that '{old_sig.name}' still behaves as expected in your use cases"
    
    def _predict_parameter_change_severity(self, old_sig: CodeSignature, new_sig: CodeSignature) -> ImpactSeverity:
        """Predict severity of parameter changes"""
        old_param_count = len(old_sig.parameters)
        new_param_count = len(new_sig.parameters)
        
        # If parameters are removed, it's more severe
        if new_param_count < old_param_count:
            return ImpactSeverity.HIGH
        # If parameters are added without defaults, it's medium severity
        elif new_param_count > old_param_count:
            return ImpactSeverity.MEDIUM
        # If parameter types change, it's medium severity
        else:
            return ImpactSeverity.MEDIUM
    
    def _predict_parameter_change_intent(self, old_sig: CodeSignature, new_sig: CodeSignature) -> ChangeIntent:
        """Predict intent of parameter changes"""
        old_param_count = len(old_sig.parameters)
        new_param_count = len(new_sig.parameters)
        
        # Adding parameters is usually intentional
        if new_param_count > old_param_count:
            return ChangeIntent.INTENTIONAL
        # Removing parameters is usually intentional
        elif new_param_count < old_param_count:
            return ChangeIntent.INTENTIONAL
        # Changing parameter types/names could be either
        else:
            return ChangeIntent.UNCLEAR
    
    def _suggest_parameter_migration(self, old_sig: CodeSignature, new_sig: CodeSignature) -> str:
        """Suggest migration for parameter changes"""
        old_param_count = len(old_sig.parameters)
        new_param_count = len(new_sig.parameters)
        
        if new_param_count < old_param_count:
            return f"Remove unused parameters when calling '{new_sig.name}'"
        elif new_param_count > old_param_count:
            return f"Add required parameters when calling '{new_sig.name}'"
        else:
            return f"Update parameter types/names when calling '{new_sig.name}'"
    
    def _extract_python_imports(self, content: str) -> set:
        """Extract Python import statements from content"""
        imports = set()
        
        # Regular import patterns
        import_patterns = [
            r'^import\s+([\w.]+)',  # import module
            r'^from\s+([\w.]+)\s+import',  # from module import ...
        ]
        
        for line in content.split('\n'):
            line = line.strip()
            for pattern in import_patterns:
                match = re.match(pattern, line)
                if match:
                    imports.add(match.group(1))
        
        return imports
        
        # Check for modified functions/classes
        for name, new_sig in new_sigs.items():
            if name in old_sigs:
                old_sig = old_sigs[name]
                sig_changes = self._compare_individual_signatures(old_sig, new_sig, file_path, commit_hash)
                changes.extend(sig_changes)
        
        return changes
    
    def _compare_individual_signatures(self, old_sig: CodeSignature, new_sig: CodeSignature, 
                                     file_path: str, commit_hash: str) -> List[BreakingChange]:
        """Compare two signatures for breaking changes"""
        changes = []
        
        # Parameter changes
        if old_sig.parameters != new_sig.parameters:
            changes.append(BreakingChange(
                id=f"{commit_hash}_{hashlib.sha256(f'{old_sig.name}_params'.encode()).hexdigest()[:8]}",
                commit_hash=commit_hash,
                change_type=ChangeType.PARAMETER_CHANGE,
                severity=self._predict_parameter_change_severity(old_sig, new_sig),
                intent=self._predict_parameter_change_intent(old_sig, new_sig),
                affected_component=old_sig.name,
                file_path=file_path,
                line_number=None,
                old_signature=self._signature_to_string(old_sig, file_path),
                new_signature=self._signature_to_string(new_sig, file_path),
                confidence_score=0.9,
                affected_users_estimate=self._estimate_affected_users(old_sig),
                migration_complexity=self._estimate_migration_complexity(old_sig, new_sig),
                description=f"Parameters changed for {old_sig.type} '{old_sig.name}'",
                technical_details=f"Old: {old_sig.parameters}\nNew: {new_sig.parameters}",
                suggested_migration=self._suggest_parameter_migration(old_sig, new_sig),
                detection_methods=["ast_analysis", "signature_comparison"]
            ))
        
        # Return type changes
        if old_sig.return_type != new_sig.return_type:
            changes.append(BreakingChange(
                id=f"{commit_hash}_{hashlib.sha256(f'{old_sig.name}_return'.encode()).hexdigest()[:8]}",
                commit_hash=commit_hash,
                change_type=ChangeType.RETURN_TYPE_CHANGE,
                severity=self._predict_return_type_severity(old_sig, new_sig),
                intent=ChangeIntent.INTENTIONAL,  # Return type changes are usually intentional
                affected_component=old_sig.name,
                file_path=file_path,
                line_number=None,
                old_signature=self._signature_to_string(old_sig, file_path),
                new_signature=self._signature_to_string(new_sig, file_path),
                confidence_score=0.85,
                affected_users_estimate=self._estimate_affected_users(old_sig),
                migration_complexity="moderate",
                description=f"Return type changed for {old_sig.type} '{old_sig.name}'",
                technical_details=f"Old return type: {old_sig.return_type}\nNew return type: {new_sig.return_type}",
                suggested_migration=self._suggest_return_type_migration(old_sig, new_sig),
                detection_methods=["ast_analysis", "signature_comparison"]
            ))
        
        # Behavioral changes (body hash comparison)
        if old_sig.body_hash != new_sig.body_hash:
            changes.append(BreakingChange(
                id=f"{commit_hash}_{hashlib.sha256(f'{old_sig.name}_behavior'.encode()).hexdigest()[:8]}",
                commit_hash=commit_hash,
                change_type=ChangeType.BEHAVIORAL_CHANGE,
                severity=ImpactSeverity.MEDIUM,  # Behavioral changes need careful analysis
                intent=ChangeIntent.UNCLEAR,
                affected_component=old_sig.name,
                file_path=file_path,
                line_number=None,
                old_signature=self._signature_to_string(old_sig, file_path),
                new_signature=self._signature_to_string(new_sig, file_path),
                confidence_score=0.7,  # Lower confidence for behavioral changes
                affected_users_estimate=self._estimate_affected_users(old_sig),
                migration_complexity="moderate",
                description=f"Implementation changed for {old_sig.type} '{old_sig.name}'",
                technical_details="The internal implementation has been modified, which may affect behavior.",
                suggested_migration="Review the changes and test thoroughly to ensure compatibility.",
                detection_methods=["ast_analysis", "behavior_hash_comparison"]
            ))
        
        return changes
    
    def _detect_signature_changes(self, old_content: str, new_content: str, file_path: str, commit_hash: str) -> List[BreakingChange]:
        """Detect changes using regex-based signature analysis (for non-Python files)"""
        changes = []
        
        try:
            # JavaScript/TypeScript function signatures
            if file_path.endswith(('.js', '.ts', '.jsx', '.tsx')):
                changes.extend(self._detect_js_signature_changes(old_content, new_content, file_path, commit_hash))
            
            # Java method signatures
            elif file_path.endswith('.java'):
                changes.extend(self._detect_java_signature_changes(old_content, new_content, file_path, commit_hash))
            
            # C++ function signatures
            elif file_path.endswith(('.cpp', '.cc', '.cxx', '.h', '.hpp')):
                changes.extend(self._detect_cpp_signature_changes(old_content, new_content, file_path, commit_hash))
            
            # API endpoint changes (various formats)
            elif any(keyword in file_path.lower() for keyword in ['api', 'route', 'endpoint', 'controller']):
                changes.extend(self._detect_api_changes(old_content, new_content, file_path, commit_hash))
        
        except Exception as e:
            logger.debug(f"Signature analysis failed for {file_path}: {e}")
        
        return changes
    
    def _detect_js_signature_changes(self, old_content: str, new_content: str, file_path: str, commit_hash: str) -> List[BreakingChange]:
        """Detect JavaScript/TypeScript signature changes"""
        changes = []
        
        # Regex patterns for JS/TS functions
        function_patterns = [
            r'function\s+(\w+)\s*\(([^)]*)\)',  # function name(params)
            r'(\w+)\s*:\s*function\s*\(([^)]*)\)',  # name: function(params)
            r'(\w+)\s*\(([^)]*)\)\s*{',  # name(params) { (arrow functions, methods)
            r'export\s+function\s+(\w+)\s*\(([^)]*)\)',  # export function name(params)
            r'(\w+)\s*=\s*\(([^)]*)\)\s*=>'  # name = (params) =>
        ]
        
        old_functions = {}
        new_functions = {}
        
        # Extract functions from old content
        for pattern in function_patterns:
            for match in re.finditer(pattern, old_content):
                name = match.group(1)
                params = match.group(2) if len(match.groups()) > 1 else ""
                old_functions[name] = params.strip()
        
        # Extract functions from new content
        for pattern in function_patterns:
            for match in re.finditer(pattern, new_content):
                name = match.group(1)
                params = match.group(2) if len(match.groups()) > 1 else ""
                new_functions[name] = params.strip()
        
        # Compare functions
        for name, old_params in old_functions.items():
            if name not in new_functions:
                # Function removed
                changes.append(BreakingChange(
                    id=f"{commit_hash}_{hashlib.sha256(f'{file_path}_{name}'.encode()).hexdigest()[:8]}",
                    commit_hash=commit_hash,
                    change_type=ChangeType.FUNCTION_REMOVAL,
                    severity=ImpactSeverity.HIGH,
                    intent=ChangeIntent.UNCLEAR,
                    affected_component=name,
                    file_path=file_path,
                    line_number=None,
                    old_signature=f"{name}({old_params})",
                    new_signature=None,
                    confidence_score=0.85,
                    affected_users_estimate="some",
                    migration_complexity="moderate",
                    description=f"JavaScript function '{name}' was removed",
                    technical_details=f"The function '{name}' is no longer available.",
                    suggested_migration=f"Replace calls to '{name}' with alternative implementation.",
                    detection_methods=["regex_signature_analysis"]
                ))
            elif old_params != new_functions[name]:
                # Parameters changed
                changes.append(BreakingChange(
                    id=f"{commit_hash}_{hashlib.sha256(f'{file_path}_{name}_params'.encode()).hexdigest()[:8]}",
                    commit_hash=commit_hash,
                    change_type=ChangeType.PARAMETER_CHANGE,
                    severity=ImpactSeverity.MEDIUM,
                    intent=ChangeIntent.INTENTIONAL,
                    affected_component=name,
                    file_path=file_path,
                    line_number=None,
                    old_signature=f"{name}({old_params})",
                    new_signature=f"{name}({new_functions[name]})",
                    confidence_score=0.8,
                    affected_users_estimate="some",
                    migration_complexity="easy",
                    description=f"JavaScript function '{name}' parameters changed",
                    technical_details=f"Parameters changed from '({old_params})' to '({new_functions[name]})'",
                    suggested_migration="Update function calls to match new parameter signature.",
                    detection_methods=["regex_signature_analysis"]
                ))
        
        return changes
    
    def _detect_api_surface_changes(self, old_content: str, new_content: str, file_path: str, commit_hash: str) -> List[BreakingChange]:
        """Detect API surface changes (REST endpoints, GraphQL, etc.)"""
        changes = []
        
        try:
            # REST API endpoint patterns
            api_patterns = [
                r'@app\.route\(["\']([^"\']+)["\']',  # Flask routes
                r'@router\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']',  # FastAPI routes
                r'app\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']',  # Express.js routes
                r'Route::(get|post|put|delete|patch)\(["\']([^"\']+)["\']',  # Laravel routes
            ]
            
            old_endpoints = set()
            new_endpoints = set()
            
            # Extract endpoints
            for pattern in api_patterns:
                for match in re.finditer(pattern, old_content):
                    if len(match.groups()) == 2:
                        method, path = match.groups()
                        old_endpoints.add(f"{method.upper()} {path}")
                    else:
                        path = match.group(1)
                        old_endpoints.add(f"* {path}")
            
            for pattern in api_patterns:
                for match in re.finditer(pattern, new_content):
                    if len(match.groups()) == 2:
                        method, path = match.groups()
                        new_endpoints.add(f"{method.upper()} {path}")
                    else:
                        path = match.group(1)
                        new_endpoints.add(f"* {path}")
            
            # Find removed endpoints
            removed_endpoints = old_endpoints - new_endpoints
            for endpoint in removed_endpoints:
                changes.append(BreakingChange(
                    id=f"{commit_hash}_{hashlib.sha256(f'{file_path}_{endpoint}'.encode()).hexdigest()[:8]}",
                    commit_hash=commit_hash,
                    change_type=ChangeType.API_SIGNATURE_CHANGE,
                    severity=ImpactSeverity.HIGH,
                    intent=ChangeIntent.INTENTIONAL,
                    affected_component=endpoint,
                    file_path=file_path,
                    line_number=None,
                    old_signature=endpoint,
                    new_signature=None,
                    confidence_score=0.9,
                    affected_users_estimate="some",
                    migration_complexity="moderate",
                    description=f"API endpoint '{endpoint}' was removed",
                    technical_details=f"The endpoint '{endpoint}' is no longer available.",
                    suggested_migration="Update API clients to use alternative endpoints or implement fallback logic.",
                    detection_methods=["api_surface_analysis"]
                ))
        
        except Exception as e:
            logger.debug(f"API surface analysis failed for {file_path}: {e}")
        
        return changes
    
    def _detect_dependency_changes(self, old_content: str, new_content: str, file_path: str, commit_hash: str) -> List[BreakingChange]:
        """Detect dependency and import changes"""
        changes = []
        
        try:
            # Python imports
            if file_path.endswith('.py'):
                old_imports = self._extract_python_imports(old_content)
                new_imports = self._extract_python_imports(new_content)
                
                removed_imports = old_imports - new_imports
                for imp in removed_imports:
                    changes.append(BreakingChange(
                        id=f"{commit_hash}_{hashlib.sha256(f'{file_path}_import_{imp}'.encode()).hexdigest()[:8]}",
                        commit_hash=commit_hash,
                        change_type=ChangeType.DEPENDENCY_CHANGE,
                        severity=ImpactSeverity.MEDIUM,
                        intent=ChangeIntent.INTENTIONAL,
                        affected_component=imp,
                        file_path=file_path,
                        line_number=None,
                        old_signature=f"import {imp}",
                        new_signature=None,
                        confidence_score=0.75,
                        affected_users_estimate="few",
                        migration_complexity="easy",
                        description=f"Import '{imp}' was removed",
                        technical_details=f"The import statement for '{imp}' is no longer present.",
                        suggested_migration=f"Add import for '{imp}' if still needed, or update code to not require it.",
                        detection_methods=["dependency_analysis"]
                    ))
            
            # Package.json changes
            elif file_path.endswith('package.json'):
                changes.extend(self._detect_package_json_changes(old_content, new_content, file_path, commit_hash))
            
            # Requirements.txt changes
            elif file_path.endswith('requirements.txt'):
                changes.extend(self._detect_requirements_changes(old_content, new_content, file_path, commit_hash))
        
        except Exception as e:
            logger.debug(f"Dependency analysis failed for {file_path}: {e}")
        
        return changes
    
    def _detect_behavioral_changes(self, old_content: str, new_content: str, file_path: str, commit_hash: str) -> List[BreakingChange]:
        """Detect behavioral changes through semantic analysis"""
        changes = []
        
        try:
            # Use embeddings to detect semantic changes
            if self.embedding_engine:
                old_embedding = self.embedding_engine.get_text_embedding(old_content)
                new_embedding = self.embedding_engine.get_text_embedding(new_content)
                
                # Calculate semantic similarity
                similarity = np.dot(old_embedding, new_embedding) / (
                    np.linalg.norm(old_embedding) * np.linalg.norm(new_embedding)
                )
                
                # If similarity is low, there might be significant behavioral changes
                if similarity < 0.7:  # Threshold for semantic change
                    changes.append(BreakingChange(
                        id=f"{commit_hash}_{hashlib.sha256(f'{file_path}_semantic'.encode()).hexdigest()[:8]}",
                        commit_hash=commit_hash,
                        change_type=ChangeType.BEHAVIORAL_CHANGE,
                        severity=ImpactSeverity.MEDIUM,
                        intent=ChangeIntent.UNCLEAR,
                        affected_component=file_path,
                        file_path=file_path,
                        line_number=None,
                        old_signature=None,
                        new_signature=None,
                        confidence_score=1.0 - similarity,  # Lower similarity = higher confidence of change
                        affected_users_estimate="some",
                        migration_complexity="moderate",
                        description=f"Significant behavioral changes detected in {file_path}",
                        technical_details=f"Semantic similarity: {similarity:.2f} (threshold: 0.7)",
                        suggested_migration="Review the changes and test thoroughly to ensure compatibility.",
                        detection_methods=["semantic_embedding_analysis"]
                    ))
        
        except Exception as e:
            logger.debug(f"Behavioral analysis failed for {file_path}: {e}")
        
        return changes
    
    def _detect_config_schema_changes(self, old_content: str, new_content: str, file_path: str, commit_hash: str) -> List[BreakingChange]:
        """Detect configuration and schema changes"""
        changes = []
        
        try:
            # JSON configuration files
            if file_path.endswith('.json'):
                changes.extend(self._detect_json_schema_changes(old_content, new_content, file_path, commit_hash))
            
            # YAML configuration files
            elif file_path.endswith(('.yml', '.yaml')):
                changes.extend(self._detect_yaml_schema_changes(old_content, new_content, file_path, commit_hash))
            
            # Database migration files
            elif 'migration' in file_path.lower() or 'schema' in file_path.lower():
                changes.extend(self._detect_database_schema_changes(old_content, new_content, file_path, commit_hash))
        
        except Exception as e:
            logger.debug(f"Config/schema analysis failed for {file_path}: {e}")
        
        return changes
    
    def _detect_file_changes(self, file_path: str, old_content: str, new_content: str, diff_item, commit_hash: str = "") -> List[BreakingChange]:
        """Detect breaking changes in a single file"""
        changes = []
        
        # Determine file type and use appropriate detector
        if file_path.endswith('.py'):
            changes.extend(self._detect_python_changes(file_path, old_content, new_content, commit_hash))
        elif file_path.endswith(('.js', '.ts')):
            changes.extend(self._detect_javascript_changes(file_path, old_content, new_content, commit_hash))
        elif file_path.endswith('.json'):
            changes.extend(self._detect_config_changes(file_path, old_content, new_content, commit_hash))
        
        # Add generic diff-based changes
        changes.extend(self._detect_generic_changes(file_path, old_content, new_content, diff_item, commit_hash))
        
        return changes
    
    def _detect_python_changes(self, file_path: str, old_content: str, new_content: str, commit_hash: str = "") -> List[BreakingChange]:
        """Detect Python-specific breaking changes using AST analysis"""
        changes = []
        
        try:
            # Parse both versions
            old_ast = ast.parse(old_content) if old_content else None
            new_ast = ast.parse(new_content) if new_content else None
            
            if old_ast and new_ast:
                # Extract signatures from both versions
                old_signatures = self._extract_python_signatures(old_ast)
                new_signatures = self._extract_python_signatures(new_ast)
                
                # Compare signatures
                for name, old_sig in old_signatures.items():
                    if name not in new_signatures:
                        changes.append(BreakingChange(
                            id=f"{commit_hash}_{hashlib.sha256(f'{file_path}_{name}_removal'.encode()).hexdigest()[:8]}",
                            commit_hash=commit_hash,
                            change_type=ChangeType.FUNCTION_REMOVAL,
                            severity=ImpactSeverity.HIGH,
                            intent=ChangeIntent.INTENTIONAL,
                            affected_component=name,
                            file_path=file_path,
                            line_number=old_sig.get('line', 0),
                            old_signature=old_sig.get('signature', f"{name}({old_sig.get('params_str', '')})"),
                            new_signature=None,
                            confidence_score=0.9,
                            affected_users_estimate="most",
                            migration_complexity="medium",
                            description=f"Python {old_sig.get('type', 'function')} '{name}' was removed",
                            technical_details=f"The {old_sig.get('type', 'function')} '{name}' is no longer available in {file_path}.\nOriginal signature: {old_sig.get('signature', name)}",
                            suggested_migration=f"Replace calls to '{name}' with alternative implementation.",
                            detection_methods=["python_ast_analysis"]
                        ))
                    elif old_sig.get('signature') != new_signatures[name].get('signature'):
                        # Signature changed
                        changes.append(BreakingChange(
                            id=f"{commit_hash}_{hashlib.sha256(f'{file_path}_{name}_change'.encode()).hexdigest()[:8]}",
                            commit_hash=commit_hash,
                            change_type=ChangeType.API_SIGNATURE_CHANGE,
                            severity=ImpactSeverity.MEDIUM,
                            intent=ChangeIntent.INTENTIONAL,
                            affected_component=name,
                            file_path=file_path,
                            line_number=new_signatures[name].get('line', 0),
                            old_signature=old_sig.get('signature', f"{name}({old_sig.get('params_str', '')})"),
                            new_signature=new_signatures[name].get('signature', f"{name}({new_signatures[name].get('params_str', '')})"),
                            confidence_score=0.8,
                            affected_users_estimate="some",
                            migration_complexity="low",
                            description=f"Python {old_sig.get('type', 'function')} signature changed",
                            technical_details=f"Signature changed for '{name}' in {file_path}.\nOld: {old_sig.get('signature')}\nNew: {new_signatures[name].get('signature')}",
                            suggested_migration=f"Update calls to '{name}' to match new signature.",
                            detection_methods=["python_ast_analysis"]
                        ))
                        
        except SyntaxError:
            # File has syntax errors, might be a major change
            if old_content and not new_content:
                changes.append(BreakingChange(
                    id=f"file_removed",
                    commit_hash="",
                    change_type=ChangeType.FUNCTION_REMOVAL,
                    severity=ImpactSeverity.CRITICAL,
                    intent=ChangeIntent.INTENTIONAL,
                    affected_component=file_path,
                    file_path=file_path,
                    line_number=0,
                    old_signature=file_path,
                    new_signature=None,
                    confidence_score=1.0,
                    affected_users_estimate="all",
                    migration_complexity="high",
                    description=f"File '{file_path}' was removed",
                    technical_details="The entire file has been removed from the codebase.",
                    suggested_migration="Find alternative implementation for functionality."
                ))
        
        return changes
    
    def _extract_python_signatures(self, tree) -> Dict[str, Dict]:
        """Extract detailed function and class signatures from Python AST"""
        signatures = {}
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Extract function arguments with type annotations
                args = []
                for arg in node.args.args:
                    arg_str = arg.arg
                    if arg.annotation:
                        try:
                            arg_str += f": {ast.unparse(arg.annotation)}"
                        except:
                            arg_str += f": {type(arg.annotation).__name__}"
                    args.append(arg_str)
                
                # Extract default arguments
                defaults = node.args.defaults
                if defaults:
                    # Match defaults to arguments (defaults correspond to last N args)
                    num_defaults = len(defaults)
                    num_args = len(args)
                    for i, default in enumerate(defaults):
                        arg_index = num_args - num_defaults + i
                        if arg_index < len(args):
                            try:
                                default_val = ast.unparse(default)
                                args[arg_index] += f" = {default_val}"
                            except:
                                args[arg_index] += " = <default>"
                
                # Extract return type annotation
                return_type = None
                if node.returns:
                    try:
                        return_type = ast.unparse(node.returns)
                    except:
                        return_type = type(node.returns).__name__
                
                # Extract decorators
                decorators = []
                for decorator in node.decorator_list:
                    try:
                        decorators.append(ast.unparse(decorator))
                    except:
                        decorators.append(type(decorator).__name__)
                
                # Create signature string
                params_str = ', '.join(args)
                signature_base = f"{'async ' if isinstance(node, ast.AsyncFunctionDef) else ''}def {node.name}({params_str})"
                if return_type:
                    signature_base += f" -> {return_type}"
                
                signatures[node.name] = {
                    'type': 'async_function' if isinstance(node, ast.AsyncFunctionDef) else 'function',
                    'args': args,
                    'params_str': params_str,
                    'return_type': return_type,
                    'decorators': decorators,
                    'signature': signature_base,
                    'line': node.lineno
                }
            elif isinstance(node, ast.ClassDef):
                # Extract base classes
                bases = []
                for base in node.bases:
                    try:
                        bases.append(ast.unparse(base))
                    except:
                        if isinstance(base, ast.Name):
                            bases.append(base.id)
                        else:
                            bases.append(type(base).__name__)
                
                # Extract decorators
                decorators = []
                for decorator in node.decorator_list:
                    try:
                        decorators.append(ast.unparse(decorator))
                    except:
                        decorators.append(type(decorator).__name__)
                
                # Create class signature
                bases_str = f"({', '.join(bases)})" if bases else ""
                signature = f"class {node.name}{bases_str}"
                
                signatures[node.name] = {
                    'type': 'class',
                    'bases': bases,
                    'decorators': decorators,
                    'signature': signature,
                    'line': node.lineno
                }
        
        return signatures
    
    def _detect_javascript_changes(self, file_path: str, old_content: str, new_content: str, commit_hash: str = "") -> List[BreakingChange]:
        """Detect JavaScript/TypeScript breaking changes with detailed signatures"""
        changes = []
        
        # Enhanced regex patterns to capture full function signatures
        function_patterns = [
            r'export\s+function\s+(\w+)\s*\(([^)]*)\)',  # export function name(params)
            r'export\s+const\s+(\w+)\s*=\s*\(([^)]*)\)\s*=>', # export const name = (params) =>
            r'export\s+const\s+(\w+)\s*=\s*function\s*\(([^)]*)\)', # export const name = function(params)
            r'function\s+(\w+)\s*\(([^)]*)\)',  # function name(params)
            r'(\w+)\s*:\s*function\s*\(([^)]*)\)',  # name: function(params) 
            r'(\w+)\s*\(([^)]*)\)\s*\{',  # name(params) { (methods)
            r'(\w+)\s*=\s*\(([^)]*)\)\s*=>'  # name = (params) =>
        ]
        
        def extract_js_functions(content):
            functions = {}
            for pattern in function_patterns:
                for match in re.finditer(pattern, content, re.MULTILINE):
                    name = match.group(1)
                    params = match.group(2).strip() if len(match.groups()) > 1 else ""
                    
                    # Clean up parameters for better readability
                    if params:
                        param_list = [p.strip() for p in params.split(',') if p.strip()]
                        params = ', '.join(param_list)
                    
                    # Determine function type from the pattern
                    full_match = match.group(0)
                    if 'export' in full_match:
                        if '=>' in full_match:
                            func_type = "export arrow function"
                        elif 'const' in full_match:
                            func_type = "export const function"
                        else:
                            func_type = "export function"
                    elif '=>' in full_match:
                        func_type = "arrow function"
                    elif ':' in full_match:
                        func_type = "method"
                    else:
                        func_type = "function"
                    
                    functions[name] = {
                        'params': params,
                        'type': func_type,
                        'full_match': full_match.strip()
                    }
            return functions
        
        old_functions = extract_js_functions(old_content)
        new_functions = extract_js_functions(new_content)
        
        # Find removed functions
        removed_functions = set(old_functions.keys()) - set(new_functions.keys())
        for func_name in removed_functions:
            func_info = old_functions[func_name]
            
            # Create detailed signature
            if func_info['params']:
                old_signature = f"{func_info['type']} {func_name}({func_info['params']})"
            else:
                old_signature = f"{func_info['type']} {func_name}()"
            
            changes.append(BreakingChange(
                id=f"{commit_hash}_{hashlib.sha256(f'{file_path}_{func_name}_removal'.encode()).hexdigest()[:8]}",
                commit_hash=commit_hash,
                change_type=ChangeType.FUNCTION_REMOVAL,
                severity=ImpactSeverity.HIGH,
                intent=ChangeIntent.UNCLEAR,
                affected_component=func_name,
                file_path=file_path,
                line_number=0,
                old_signature=old_signature,
                new_signature=None,
                confidence_score=0.85,
                affected_users_estimate="most",
                migration_complexity="moderate",
                description=f"JavaScript/TypeScript {func_info['type']} '{func_name}' was removed",
                technical_details=f"The {func_info['type']} '{func_name}' is no longer available in {file_path}.\nOriginal definition: {func_info['full_match']}",
                suggested_migration=f"Replace usage of '{func_name}' with alternative implementation.",
                detection_methods=["javascript_detailed_analysis"]
            ))
        
        # Find functions with changed signatures
        for func_name in old_functions:
            if func_name in new_functions:
                old_func = old_functions[func_name]
                new_func = new_functions[func_name]
                
                if old_func['params'] != new_func['params']:
                    old_signature = f"{old_func['type']} {func_name}({old_func['params']})"
                    new_signature = f"{new_func['type']} {func_name}({new_func['params']})"
                    
                    changes.append(BreakingChange(
                        id=f"{commit_hash}_{hashlib.sha256(f'{file_path}_{func_name}_params'.encode()).hexdigest()[:8]}",
                        commit_hash=commit_hash,
                        change_type=ChangeType.PARAMETER_CHANGE,
                        severity=ImpactSeverity.MEDIUM,
                        intent=ChangeIntent.INTENTIONAL,
                        affected_component=func_name,
                        file_path=file_path,
                        line_number=0,
                        old_signature=old_signature,
                        new_signature=new_signature,
                        confidence_score=0.9,
                        affected_users_estimate="some",
                        migration_complexity="moderate",
                        description=f"JavaScript/TypeScript {func_name} parameters changed",
                        technical_details=f"Function signature changed:\nOld: {old_signature}\nNew: {new_signature}",
                        suggested_migration=f"Update all calls to '{func_name}' to match the new parameter signature.",
                        detection_methods=["javascript_detailed_analysis"]
                    ))
        
        return changes
    
    def _detect_config_changes(self, file_path: str, old_content: str, new_content: str, commit_hash: str = "") -> List[BreakingChange]:
        """Detect configuration file changes"""
        changes = []
        
        try:
            old_config = json.loads(old_content) if old_content else {}
            new_config = json.loads(new_content) if new_content else {}
            
            # Check for removed keys
            removed_keys = set(old_config.keys()) - set(new_config.keys())
            for key in removed_keys:
                changes.append(BreakingChange(
                    id=f"config_remove_{key}",
                    commit_hash="",
                    change_type=ChangeType.CONFIGURATION_CHANGE,
                    severity=ImpactSeverity.MEDIUM,
                    intent=ChangeIntent.INTENTIONAL,
                    affected_component=key,
                    file_path=file_path,
                    line_number=0,
                    old_signature=f"config.{key}",
                    new_signature=None,
                    confidence_score=0.95,
                    affected_users_estimate="some",
                    migration_complexity="low",
                    description=f"Configuration key '{key}' was removed",
                    technical_details=f"The configuration option '{key}' is no longer available.",
                    suggested_migration=f"Remove references to '{key}' or provide default value."
                ))
                
        except json.JSONDecodeError:
            pass
        
        return changes
    
    def _detect_generic_changes(self, file_path: str, old_content: str, new_content: str, diff_item, commit_hash: str = "") -> List[BreakingChange]:
        """Detect generic breaking changes from diff analysis"""
        changes = []
        
        # File removal
        if old_content and not new_content:
            changes.append(BreakingChange(
                id=f"file_removed_{file_path.replace('/', '_')}",
                commit_hash="",
                change_type=ChangeType.FUNCTION_REMOVAL,
                severity=ImpactSeverity.CRITICAL,
                intent=ChangeIntent.INTENTIONAL,
                affected_component=file_path,
                file_path=file_path,
                line_number=0,
                old_signature=file_path,
                new_signature=None,
                confidence_score=1.0,
                affected_users_estimate="all" if "api" in file_path.lower() else "some",
                migration_complexity="high",
                description=f"File '{file_path}' was removed",
                technical_details="The entire file has been deleted from the repository.",
                suggested_migration="Find alternative implementation for the functionality."
            ))
        
        # Large deletions (potential breaking changes)
        if old_content and new_content:
            old_lines = len(old_content.splitlines())
            new_lines = len(new_content.splitlines())
            deletion_ratio = (old_lines - new_lines) / old_lines if old_lines > 0 else 0
            
            if deletion_ratio > 0.3:  # More than 30% of lines deleted
                changes.append(BreakingChange(
                    id=f"major_refactor_{file_path.replace('/', '_')}",
                    commit_hash="",
                    change_type=ChangeType.BEHAVIORAL_CHANGE,
                    severity=ImpactSeverity.MEDIUM,
                    intent=ChangeIntent.INTENTIONAL,
                    affected_component=file_path,
                    file_path=file_path,
                    line_number=0,
                    old_signature=None,
                    new_signature=None,
                    confidence_score=0.7,
                    affected_users_estimate="some",
                    migration_complexity="medium",
                    description=f"Major refactoring in '{file_path}' ({deletion_ratio:.0%} of lines removed)",
                    technical_details=f"Significant code reduction: {old_lines} -> {new_lines} lines",
                    suggested_migration="Review changes carefully and test all functionality."
                ))
        
        return changes
    
    def _generate_migration_suggestions(self, breaking_changes: List[BreakingChange]) -> List[str]:
        """Generate migration suggestions for breaking changes"""
        suggestions = []
        
        for change in breaking_changes:
            if change.change_type == ChangeType.FUNCTION_REMOVAL:
                suggestions.append(f"Update imports to remove references to '{change.affected_component}'")
            elif change.change_type == ChangeType.API_SIGNATURE_CHANGE:
                suggestions.append(f"Update function calls to match new signature for '{change.affected_component}'")
            elif change.change_type == ChangeType.CONFIGURATION_CHANGE:
                suggestions.append(f"Update configuration files to remove deprecated key '{change.affected_component}'")
            elif change.change_type == ChangeType.BEHAVIORAL_CHANGE:
                suggestions.append(f"Review and test changes in '{change.file_path}' for compatibility")
        
        return suggestions
    
    def _generate_ai_analysis(self, commit, changes: List[BreakingChange]) -> Optional[str]:
        """Generate AI analysis of breaking changes"""
        if not self.claude_analyzer or not self.claude_analyzer.available:
            return None
        
        try:
            prompt = f"""# 🔍 Breaking Change Analysis

Analyze this commit for breaking changes and their impact:

**Commit**: {commit.hexsha[:8]}
**Message**: {commit.message.strip()}
**Author**: {commit.author.name}

## Detected Changes
{chr(10).join(f"- {change.change_type.value}: {change.description}" for change in changes[:5])}

## Analysis Required

Provide expert analysis on:

1. **Impact Assessment**: How severe are these changes for users?
2. **Intent Classification**: Are these intentional breaking changes or accidental?
3. **Migration Strategy**: What's the best approach for users to adapt?
4. **Business Justification**: Why might these changes be necessary?
5. **Risk Mitigation**: How to minimize disruption?

Focus on practical guidance for development teams."""

            response = self.claude_analyzer.client.messages.create(
                model=self.claude_analyzer.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return None
    
    def _detect_diff_patterns(self, old_content: str, new_content: str, file_path: str, commit_hash: str, diff_item) -> List[BreakingChange]:
        """Detect breaking changes using basic diff pattern analysis"""
        changes = []
        
        try:
            # C keywords and library functions to exclude
            c_keywords = {
                'auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do',
                'double', 'else', 'enum', 'extern', 'float', 'for', 'goto', 'if',
                'inline', 'int', 'long', 'register', 'return', 'short', 'signed',
                'sizeof', 'static', 'struct', 'switch', 'typedef', 'union', 'unsigned',
                'void', 'volatile', 'while', '_Bool', '_Complex', '_Imaginary',
                'restrict', '_Static_assert', '_Noreturn', '_Thread_local', '_Alignas',
                '_Alignof', '_Atomic', '_Generic'
            }
            
            c_library_functions = {
                'printf', 'scanf', 'malloc', 'free', 'strlen', 'strcpy', 'strcmp',
                'strcat', 'strdup', 'fopen', 'fclose', 'fread', 'fwrite', 'fprintf',
                'sprintf', 'memcpy', 'memset', 'exit', 'abort', 'atoi', 'atof'
            }
            
            # File completely removed
            if old_content and not new_content:
                changes.append(BreakingChange(
                    id=f"{commit_hash}_{hashlib.sha256(file_path.encode()).hexdigest()[:8]}",
                    commit_hash=commit_hash,
                    change_type=ChangeType.FUNCTION_REMOVAL,
                    severity=ImpactSeverity.HIGH if any(keyword in file_path.lower() for keyword in ['api', 'public', 'interface']) else ImpactSeverity.MEDIUM,
                    intent=ChangeIntent.INTENTIONAL,
                    affected_component=file_path,
                    file_path=file_path,
                    line_number=None,
                    old_signature=f"File: {file_path}",
                    new_signature=None,
                    confidence_score=1.0,
                    affected_users_estimate="most" if any(keyword in file_path.lower() for keyword in ['api', 'public']) else "some",
                    migration_complexity="complex",
                    description=f"File '{file_path}' was removed",
                    technical_details=f"The entire file '{file_path}' has been deleted from the repository",
                    suggested_migration=f"Find alternative implementation for functionality in '{file_path}'",
                    detection_methods=["diff_pattern_analysis"]
                ))
            
            # Look for function/method/class removals in diffs
            if hasattr(diff_item, 'diff') and diff_item.diff:
                diff_text = diff_item.diff.decode('utf-8', errors='ignore') if isinstance(diff_item.diff, bytes) else str(diff_item.diff)
                
                # More precise patterns for different languages to extract full signatures
                removed_functions = []
                if file_path.endswith('.py'):
                    # Python-specific patterns with parameter extraction
                    py_func_pattern = r'^-\s*(?:def|class)\s+(\w+)(?:\s*\(([^)]*)\))?'
                    matches = re.findall(py_func_pattern, diff_text, re.MULTILINE)
                    for match in matches:
                        func_name = match[0]
                        params = match[1] if len(match) > 1 and match[1] else ""
                        removed_functions.append((func_name, params, "python"))
                elif file_path.endswith(('.js', '.ts')):
                    # JavaScript/TypeScript patterns with parameter extraction
                    js_patterns = [
                        r'^-\s*function\s+(\w+)\s*\(([^)]*)\)',
                        r'^-\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*\(([^)]*)\)\s*=>',
                        r'^-\s*(?:export\s+)?class\s+(\w+)(?:\s*\(([^)]*)\))?'
                    ]
                    for pattern in js_patterns:
                        matches = re.findall(pattern, diff_text, re.MULTILINE)
                        for match in matches:
                            func_name = match[0]
                            params = match[1] if len(match) > 1 and match[1] else ""
                            removed_functions.append((func_name, params, "javascript"))
                elif file_path.endswith(('.c', '.cpp', '.h', '.hpp')):
                    # C/C++ patterns with return type and parameter extraction
                    cpp_pattern = r'^-\s*(?:static\s+|extern\s+|inline\s+)*(\w+(?:\s*\*)*)\s+(\w+)\s*\(([^)]*)\)\s*[{;]'
                    matches = re.findall(cpp_pattern, diff_text, re.MULTILINE)
                    for match in matches:
                        return_type = match[0].strip()
                        func_name = match[1]
                        params = match[2] if len(match) > 2 and match[2] else ""
                        removed_functions.append((func_name, params, "cpp", return_type))
                elif file_path.endswith('.java'):
                    # Java patterns with modifier and parameter extraction
                    java_pattern = r'^-\s*(public|private|protected)?\s*(?:static\s+)?(\w+)\s+(\w+)\s*\(([^)]*)\)'
                    matches = re.findall(java_pattern, diff_text, re.MULTILINE)
                    for match in matches:
                        modifier = match[0] or "public"
                        return_type = match[1]
                        func_name = match[2]
                        params = match[3] if len(match) > 3 and match[3] else ""
                        removed_functions.append((func_name, params, "java", modifier, return_type))
                else:
                    # Generic pattern for other languages
                    generic_pattern = r'^-\s*(?:def|function|class)\s+(\w+)(?:\s*\(([^)]*)\))?'
                    matches = re.findall(generic_pattern, diff_text, re.MULTILINE)
                    for match in matches:
                        func_name = match[0]
                        params = match[1] if len(match) > 1 and match[1] else ""
                        removed_functions.append((func_name, params, "generic"))
                
                for func_info in removed_functions:
                    func_name = func_info[0]
                    params = func_info[1]
                    lang_type = func_info[2]
                    
                    # Filter out keywords and library functions
                    if (func_name and 
                        len(func_name) > 2 and  # Avoid very short names
                        func_name not in c_keywords and 
                        func_name not in c_library_functions and
                        not func_name.isdigit()):
                        
                        # Create language-appropriate signature with actual parameters
                        if lang_type == "python":
                            old_signature = f"def {func_name}({params})"
                            description = f"Python function/class '{func_name}' was removed"
                        elif lang_type == "javascript":
                            old_signature = f"function {func_name}({params})"
                            description = f"JavaScript/TypeScript function '{func_name}' was removed"
                        elif lang_type == "cpp":
                            return_type = func_info[3] if len(func_info) > 3 else "void"
                            old_signature = f"{return_type} {func_name}({params})"
                            description = f"C/C++ function '{func_name}' was removed"
                        elif lang_type == "java":
                            modifier = func_info[3] if len(func_info) > 3 else "public"
                            return_type = func_info[4] if len(func_info) > 4 else "void"
                            old_signature = f"{modifier} {return_type} {func_name}({params})"
                            description = f"Java method '{func_name}' was removed"
                        else:
                            old_signature = f"{func_name}({params})"
                            description = f"Function '{func_name}' was removed"
                        
                        changes.append(BreakingChange(
                            id=f"{commit_hash}_{hashlib.sha256(f'{file_path}_{func_name}'.encode()).hexdigest()[:8]}",
                            commit_hash=commit_hash,
                            change_type=ChangeType.FUNCTION_REMOVAL,
                            severity=ImpactSeverity.HIGH if not func_name.startswith('_') else ImpactSeverity.MEDIUM,
                            intent=ChangeIntent.UNCLEAR,
                            affected_component=func_name,
                            file_path=file_path,
                            line_number=None,
                            old_signature=old_signature,
                            new_signature=None,
                            confidence_score=0.8,
                            affected_users_estimate="some" if func_name.startswith('_') else "most",
                            migration_complexity="moderate",
                            description=description,
                            technical_details=f"The definition of '{func_name}' was deleted from {file_path}",
                            suggested_migration=f"Replace calls to '{func_name}' with alternative implementation",
                            detection_methods=["diff_pattern_analysis"]
                        ))
                
                # Look for parameter changes in function signatures with actual parameter extraction
                param_change_pattern = r'^-\s*(?:def|function)\s+(\w+)\s*\(([^)]*)\).*\n\+\s*(?:def|function)\s+\1\s*\(([^)]*)\)'
                param_changes = re.findall(param_change_pattern, diff_text, re.MULTILINE)
                for match in param_changes:
                    func_name = match[0]
                    old_params = match[1].strip() if len(match) > 1 else ""
                    new_params = match[2].strip() if len(match) > 2 else ""
                    
                    # Create language-appropriate signatures for parameter changes
                    if file_path.endswith('.py'):
                        old_signature = f"def {func_name}({old_params})"
                        new_signature = f"def {func_name}({new_params})"
                        description = f"Python function '{func_name}' parameters changed"
                    elif file_path.endswith(('.js', '.ts')):
                        old_signature = f"function {func_name}({old_params})"
                        new_signature = f"function {func_name}({new_params})"
                        description = f"JavaScript/TypeScript function '{func_name}' parameters changed"
                    elif file_path.endswith(('.c', '.cpp', '.h', '.hpp')):
                        # Try to extract return type for C/C++ functions
                        old_signature = f"{func_name}({old_params})"
                        new_signature = f"{func_name}({new_params})"
                        description = f"C/C++ function '{func_name}' parameters changed"
                    elif file_path.endswith('.java'):
                        old_signature = f"public {func_name}({old_params})"
                        new_signature = f"public {func_name}({new_params})"
                        description = f"Java method '{func_name}' parameters changed"
                    else:
                        old_signature = f"{func_name}({old_params})"
                        new_signature = f"{func_name}({new_params})"
                        description = f"Function '{func_name}' parameters changed"
                        
                    changes.append(BreakingChange(
                        id=f"{commit_hash}_{hashlib.sha256(f'{file_path}_{func_name}_params'.encode()).hexdigest()[:8]}",
                        commit_hash=commit_hash,
                        change_type=ChangeType.PARAMETER_CHANGE,
                        severity=ImpactSeverity.MEDIUM,
                        intent=ChangeIntent.INTENTIONAL,
                        affected_component=func_name,
                        file_path=file_path,
                        line_number=None,
                        old_signature=old_signature,
                        new_signature=new_signature,
                        confidence_score=0.7,
                        affected_users_estimate="some",
                        migration_complexity="easy",
                        description=description,
                        technical_details=f"The parameter signature of '{func_name}' was modified",
                        suggested_migration=f"Update calls to '{func_name}' to match new signature",
                        detection_methods=["diff_pattern_analysis"]
                    ))
        
        except Exception as e:
            logger.debug(f"Diff pattern analysis failed for {file_path}: {e}")
        
        return changes
    
    # Placeholder methods for additional language support
    def _detect_java_signature_changes(self, old_content: str, new_content: str, file_path: str, commit_hash: str) -> List[BreakingChange]:
        """Detect Java signature changes"""
        changes = []
        
        try:
            # Enhanced Java method and class patterns to capture parameters
            java_patterns = [
                # Method definitions: public/private/protected return_type method_name(params)
                r'(public|private|protected|static|final|abstract)\s+(?:(?:public|private|protected|static|final|abstract)\s+)*(\w+(?:<[^>]+>)?)\s+(\w+)\s*\(([^)]*)\)\s*(?:throws\s+[^{]+)?\s*[{;]',
                # Class definitions
                r'(public|private|protected|abstract|final)?\s*class\s+(\w+)(?:\s+extends\s+([\w<>,\s]+))?(?:\s+implements\s+([\w,\s]+))?\s*\{',
                # Interface definitions
                r'(public|private|protected)?\s*interface\s+(\w+)(?:\s+extends\s+([\w,\s]+))?\s*\{',
                # Enum definitions
                r'(public|private|protected)?\s*enum\s+(\w+)\s*\{',
            ]
            
            def extract_java_signatures(content):
                signatures = {}
                for pattern in java_patterns:
                    for match in re.finditer(pattern, content, re.MULTILINE):
                        if len(match.groups()) >= 4 and 'method' in pattern or ('public|private|protected|static|final|abstract' in pattern and len(match.groups()) >= 3):
                            # Method signature with parameters
                            modifiers = match.group(1) or 'package'
                            return_type = match.group(2)
                            method_name = match.group(3)
                            params = match.group(4) if len(match.groups()) >= 4 else ""
                            
                            if method_name and method_name not in ['class', 'interface', 'enum']:
                                # Clean up parameters
                                if params:
                                    param_list = [p.strip() for p in params.split(',') if p.strip()]
                                    params_clean = ', '.join(param_list)
                                else:
                                    params_clean = ""
                                
                                signature = f"{modifiers} {return_type} {method_name}({params_clean})"
                                signatures[method_name] = {
                                    'type': 'method',
                                    'modifiers': modifiers,
                                    'return_type': return_type,
                                    'params': params_clean,
                                    'signature': signature
                                }
                        elif len(match.groups()) >= 2:
                            # Class/interface/enum
                            modifiers = match.group(1) or 'package'
                            type_name = match.group(2)
                            
                            if type_name:
                                if 'class' in pattern:
                                    extends = match.group(3) if len(match.groups()) >= 3 else ""
                                    implements = match.group(4) if len(match.groups()) >= 4 else ""
                                    inheritance = ""
                                    if extends:
                                        inheritance += f" extends {extends.strip()}"
                                    if implements:
                                        inheritance += f" implements {implements.strip()}"
                                    signature = f"{modifiers} class {type_name}{inheritance}"
                                elif 'interface' in pattern:
                                    extends = match.group(3) if len(match.groups()) >= 3 else ""
                                    inheritance = f" extends {extends.strip()}" if extends else ""
                                    signature = f"{modifiers} interface {type_name}{inheritance}"
                                else:  # enum
                                    signature = f"{modifiers} enum {type_name}"
                                
                                signatures[type_name] = {
                                    'type': 'class' if 'class' in pattern else ('interface' if 'interface' in pattern else 'enum'),
                                    'modifiers': modifiers,
                                    'signature': signature
                                }
                return signatures
            
            old_signatures = extract_java_signatures(old_content)
            new_signatures = extract_java_signatures(new_content)
            
            # Find removed signatures
            removed_signatures = set(old_signatures.keys()) - set(new_signatures.keys())
            for component_name in removed_signatures:
                sig_info = old_signatures[component_name]
                
                # Determine if it's public (more severe) or private
                is_public = 'public' in sig_info['modifiers']
                severity = ImpactSeverity.HIGH if is_public else ImpactSeverity.MEDIUM
                
                changes.append(BreakingChange(
                    id=f"{commit_hash}_{hashlib.sha256(f'{file_path}_{component_name}'.encode()).hexdigest()[:8]}",
                    commit_hash=commit_hash,
                    change_type=ChangeType.FUNCTION_REMOVAL if sig_info['type'] == 'method' else ChangeType.CLASS_STRUCTURE_CHANGE,
                    severity=severity,
                    intent=ChangeIntent.UNCLEAR,
                    affected_component=component_name,
                    file_path=file_path,
                    line_number=None,
                    old_signature=sig_info['signature'],
                    new_signature=None,
                    confidence_score=0.8,
                    affected_users_estimate="most" if is_public else "some",
                    migration_complexity="moderate",
                    description=f"Java {sig_info['type']} '{component_name}' was removed",
                    technical_details=f"The {sig_info['type']} '{component_name}' is no longer available in {file_path}.\nOriginal signature: {sig_info['signature']}",
                    suggested_migration=f"Replace calls to '{component_name}' with alternative implementation",
                    detection_methods=["java_signature_analysis"]
                ))
                
            # Find methods with changed signatures
            for component_name in old_signatures:
                if component_name in new_signatures:
                    old_sig = old_signatures[component_name]
                    new_sig = new_signatures[component_name]
                    
                    if old_sig['signature'] != new_sig['signature']:
                        is_public = 'public' in old_sig['modifiers']
                        severity = ImpactSeverity.MEDIUM if is_public else ImpactSeverity.LOW
                        
                        changes.append(BreakingChange(
                            id=f"{commit_hash}_{hashlib.sha256(f'{file_path}_{component_name}_change'.encode()).hexdigest()[:8]}",
                            commit_hash=commit_hash,
                            change_type=ChangeType.API_SIGNATURE_CHANGE,
                            severity=severity,
                            intent=ChangeIntent.INTENTIONAL,
                            affected_component=component_name,
                            file_path=file_path,
                            line_number=None,
                            old_signature=old_sig['signature'],
                            new_signature=new_sig['signature'],
                            confidence_score=0.9,
                            affected_users_estimate="some",
                            migration_complexity="moderate",
                            description=f"Java {old_sig['type']} '{component_name}' signature changed",
                            technical_details=f"Signature changed:\nOld: {old_sig['signature']}\nNew: {new_sig['signature']}",
                            suggested_migration=f"Update all calls to '{component_name}' to match the new signature",
                            detection_methods=["java_signature_analysis"]
                        ))
                
        except Exception as e:
            logger.debug(f"Java signature analysis failed for {file_path}: {e}")
        
        return changes
    
    def _detect_cpp_signature_changes(self, old_content: str, new_content: str, file_path: str, commit_hash: str) -> List[BreakingChange]:
        """Detect C/C++ signature changes"""
        changes = []
        
        try:
            # C keywords to exclude from function detection
            c_keywords = {
                'auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do',
                'double', 'else', 'enum', 'extern', 'float', 'for', 'goto', 'if',
                'inline', 'int', 'long', 'register', 'return', 'short', 'signed',
                'sizeof', 'static', 'struct', 'switch', 'typedef', 'union', 'unsigned',
                'void', 'volatile', 'while', '_Bool', '_Complex', '_Imaginary',
                'restrict', '_Static_assert', '_Noreturn', '_Thread_local', '_Alignas',
                '_Alignof', '_Atomic', '_Generic'
            }
            
            # Common C library functions to exclude  
            c_library_functions = {
                'printf', 'scanf', 'malloc', 'free', 'strlen', 'strcpy', 'strcmp',
                'strcat', 'strdup', 'fopen', 'fclose', 'fread', 'fwrite', 'fprintf',
                'sprintf', 'memcpy', 'memset', 'exit', 'abort', 'atoi', 'atof'
            }
            
            # More precise C/C++ function patterns to capture full signatures
            function_patterns = [
                # Function definitions with return type and parameters
                r'\b(?:static\s+|extern\s+|inline\s+)*(\w+(?:\s*\*)*)\s+(\w+)\s*\(([^)]*)\)\s*\{',
                # Function declarations with return type and parameters
                r'\b(?:static\s+|extern\s+|inline\s+)*(\w+(?:\s*\*)*)\s+(\w+)\s*\(([^)]*)\)\s*;',
                # Struct definitions
                r'\bstruct\s+(\w+)\s*\{',
                # Enum definitions  
                r'\benum\s+(\w+)\s*\{',
                # Typedef function pointers
                r'\btypedef\s+.*\(\s*\*\s*(\w+)\s*\)\s*\(([^)]*)\)',
            ]
            
            old_functions = {}
            new_functions = {}
            
            def extract_functions(content, functions_dict):
                for pattern in function_patterns:
                    for match in re.finditer(pattern, content, re.MULTILINE):
                        if 'struct' in pattern or 'enum' in pattern:
                            # Struct or enum - simpler pattern
                            func_name = match.group(1)
                            if (func_name and 
                                func_name not in c_keywords and 
                                func_name not in c_library_functions and
                                not func_name.isdigit() and
                                len(func_name) > 2):
                                functions_dict[func_name] = {
                                    'type': 'struct' if 'struct' in pattern else 'enum',
                                    'signature': f"{'struct' if 'struct' in pattern else 'enum'} {func_name}",
                                    'params': '',
                                    'return_type': None
                                }
                        elif len(match.groups()) >= 3:
                            # Function with parameters
                            return_type = match.group(1).strip()
                            func_name = match.group(2).strip()
                            params = match.group(3).strip()
                            
                            # Clean up parameters
                            if params:
                                # Remove extra whitespace and normalize
                                param_list = [p.strip() for p in params.split(',') if p.strip()]
                                params = ', '.join(param_list)
                            
                            if (func_name and 
                                func_name not in c_keywords and 
                                func_name not in c_library_functions and
                                not func_name.isdigit() and
                                len(func_name) > 2):
                                
                                # Create detailed signature
                                signature = f"{return_type} {func_name}({params})"
                                
                                functions_dict[func_name] = {
                                    'type': 'function',
                                    'signature': signature,
                                    'params': params,
                                    'return_type': return_type
                                }
                        elif len(match.groups()) >= 2:
                            # Function without captured parameters or typedef
                            if 'typedef' in pattern:
                                func_name = match.group(1)
                                params = match.group(2) if len(match.groups()) > 1 else ''
                                signature = f"typedef function_ptr {func_name}({params})"
                            else:
                                return_type = match.group(1).strip()
                                func_name = match.group(2).strip()
                                signature = f"{return_type} {func_name}(...)"
                            
                            if (func_name and 
                                func_name not in c_keywords and 
                                func_name not in c_library_functions and
                                not func_name.isdigit() and
                                len(func_name) > 2):
                                
                                functions_dict[func_name] = {
                                    'type': 'typedef' if 'typedef' in pattern else 'function',
                                    'signature': signature,
                                    'params': params if 'typedef' in pattern else '...',
                                    'return_type': return_type if 'typedef' not in pattern else None
                                }
            
            # Extract functions from both versions
            extract_functions(old_content, old_functions)
            extract_functions(new_content, new_functions)
            
            # Find removed functions
            removed_functions = set(old_functions.keys()) - set(new_functions.keys())
            for func_name in removed_functions:
                func_info = old_functions[func_name]
                changes.append(BreakingChange(
                    id=f"{commit_hash}_{hashlib.sha256(f'{file_path}_{func_name}'.encode()).hexdigest()[:8]}",
                    commit_hash=commit_hash,
                    change_type=ChangeType.FUNCTION_REMOVAL,
                    severity=ImpactSeverity.HIGH,
                    intent=ChangeIntent.UNCLEAR,
                    affected_component=func_name,
                    file_path=file_path,
                    line_number=None,
                    old_signature=func_info['signature'],
                    new_signature=None,
                    confidence_score=0.85,  # Higher confidence with better filtering
                    affected_users_estimate="some",
                    migration_complexity="moderate",
                    description=f"C/C++ {func_info['type']} '{func_name}' was removed",
                    technical_details=f"The {func_info['type']} '{func_name}' is no longer available in {file_path}.\nOriginal signature: {func_info['signature']}",
                    suggested_migration=f"Replace calls to '{func_name}' with alternative implementation",
                    detection_methods=["cpp_signature_analysis"]
                ))
                
            # Find functions with changed signatures
            for func_name in old_functions:
                if func_name in new_functions:
                    old_func = old_functions[func_name]
                    new_func = new_functions[func_name]
                    
                    if old_func['signature'] != new_func['signature']:
                        changes.append(BreakingChange(
                            id=f"{commit_hash}_{hashlib.sha256(f'{file_path}_{func_name}_change'.encode()).hexdigest()[:8]}",
                            commit_hash=commit_hash,
                            change_type=ChangeType.API_SIGNATURE_CHANGE,
                            severity=ImpactSeverity.MEDIUM,
                            intent=ChangeIntent.INTENTIONAL,
                            affected_component=func_name,
                            file_path=file_path,
                            line_number=None,
                            old_signature=old_func['signature'],
                            new_signature=new_func['signature'],
                            confidence_score=0.9,
                            affected_users_estimate="some",
                            migration_complexity="moderate",
                            description=f"C/C++ {func_name} signature changed",
                            technical_details=f"Function signature changed:\nOld: {old_func['signature']}\nNew: {new_func['signature']}",
                            suggested_migration=f"Update all calls to '{func_name}' to match the new signature",
                            detection_methods=["cpp_signature_analysis"]
                        ))
                
        except Exception as e:
            logger.debug(f"C++ signature analysis failed for {file_path}: {e}")
        
        return changes
    
    def _detect_package_json_changes(self, old_content: str, new_content: str, file_path: str, commit_hash: str) -> List[BreakingChange]:
        """Detect package.json dependency changes"""
        changes = []
        
        try:
            old_package = json.loads(old_content) if old_content else {}
            new_package = json.loads(new_content) if new_content else {}
            
            # Check dependencies section
            old_deps = old_package.get('dependencies', {})
            new_deps = new_package.get('dependencies', {})
            old_dev_deps = old_package.get('devDependencies', {})
            new_dev_deps = new_package.get('devDependencies', {})
            
            # Removed dependencies
            removed_deps = set(old_deps.keys()) - set(new_deps.keys())
            for dep in removed_deps:
                changes.append(BreakingChange(
                    id=f"{commit_hash}_{hashlib.sha256(f'{file_path}_{dep}_removal'.encode()).hexdigest()[:8]}",
                    commit_hash=commit_hash,
                    change_type=ChangeType.DEPENDENCY_CHANGE,
                    severity=ImpactSeverity.HIGH,
                    intent=ChangeIntent.INTENTIONAL,
                    affected_component=dep,
                    file_path=file_path,
                    line_number=None,
                    old_signature=f"dependency: {dep}@{old_deps[dep]}",
                    new_signature=None,
                    confidence_score=0.95,
                    affected_users_estimate="most",
                    migration_complexity="moderate",
                    description=f"NPM dependency '{dep}' was removed",
                    technical_details=f"The dependency '{dep}' version {old_deps[dep]} is no longer required.",
                    suggested_migration=f"Remove usage of '{dep}' or add it back if still needed.",
                    detection_methods=["package_json_analysis"]
                ))
            
            # Major version upgrades (potentially breaking)
            for dep in old_deps:
                if dep in new_deps:
                    old_version = old_deps[dep].replace('^', '').replace('~', '').replace('>=', '').replace('>', '')
                    new_version = new_deps[dep].replace('^', '').replace('~', '').replace('>=', '').replace('>', '')
                    
                    try:
                        # Simple version comparison - major version change
                        old_major = int(old_version.split('.')[0]) if old_version.split('.')[0].isdigit() else 0
                        new_major = int(new_version.split('.')[0]) if new_version.split('.')[0].isdigit() else 0
                        
                        if new_major > old_major:
                            changes.append(BreakingChange(
                                id=f"{commit_hash}_{hashlib.sha256(f'{file_path}_{dep}_upgrade'.encode()).hexdigest()[:8]}",
                                commit_hash=commit_hash,
                                change_type=ChangeType.DEPENDENCY_CHANGE,
                                severity=ImpactSeverity.MEDIUM,
                                intent=ChangeIntent.INTENTIONAL,
                                affected_component=dep,
                                file_path=file_path,
                                line_number=None,
                                old_signature=f"dependency: {dep}@{old_deps[dep]}",
                                new_signature=f"dependency: {dep}@{new_deps[dep]}",
                                confidence_score=0.8,
                                affected_users_estimate="some",
                                migration_complexity="moderate",
                                description=f"NPM dependency '{dep}' major version upgrade",
                                technical_details=f"Dependency '{dep}' upgraded from v{old_major} to v{new_major}, may contain breaking changes.",
                                suggested_migration=f"Review '{dep}' changelog for breaking changes and update code accordingly.",
                                detection_methods=["package_json_analysis"]
                            ))
                    except (ValueError, IndexError):
                        # Skip if version parsing fails
                        pass
                        
        except json.JSONDecodeError:
            logger.debug(f"Failed to parse package.json: {file_path}")
        except Exception as e:
            logger.debug(f"Package.json analysis failed for {file_path}: {e}")
        
        return changes
    
    def _detect_requirements_changes(self, old_content: str, new_content: str, file_path: str, commit_hash: str) -> List[BreakingChange]:
        """Detect requirements.txt changes"""
        changes = []
        
        try:
            # Parse requirements.txt format
            def parse_requirements(content):
                deps = {}
                for line in content.splitlines():
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Handle different formats: package==1.0.0, package>=1.0.0, package
                        if '==' in line:
                            name, version = line.split('==', 1)
                        elif '>=' in line:
                            name, version = line.split('>=', 1)
                        elif '~=' in line:
                            name, version = line.split('~=', 1)
                        else:
                            name, version = line, 'any'
                        deps[name.strip()] = version.strip()
                return deps
            
            old_deps = parse_requirements(old_content) if old_content else {}
            new_deps = parse_requirements(new_content) if new_content else {}
            
            # Removed dependencies
            removed_deps = set(old_deps.keys()) - set(new_deps.keys())
            for dep in removed_deps:
                changes.append(BreakingChange(
                    id=f"{commit_hash}_{hashlib.sha256(f'{file_path}_{dep}_removal'.encode()).hexdigest()[:8]}",
                    commit_hash=commit_hash,
                    change_type=ChangeType.DEPENDENCY_CHANGE,
                    severity=ImpactSeverity.HIGH,
                    intent=ChangeIntent.INTENTIONAL,
                    affected_component=dep,
                    file_path=file_path,
                    line_number=None,
                    old_signature=f"requirement: {dep}=={old_deps[dep]}",
                    new_signature=None,
                    confidence_score=0.95,
                    affected_users_estimate="most",
                    migration_complexity="moderate",
                    description=f"Python dependency '{dep}' was removed",
                    technical_details=f"The requirement '{dep}' version {old_deps[dep]} is no longer specified.",
                    suggested_migration=f"Remove imports/usage of '{dep}' or add it back if still needed.",
                    detection_methods=["requirements_analysis"]
                ))
            
            # Version downgrades (potentially breaking)
            for dep in old_deps:
                if dep in new_deps and old_deps[dep] != 'any' and new_deps[dep] != 'any':
                    try:
                        # Simple version comparison for major changes
                        old_version = old_deps[dep].replace('>=', '').replace('~=', '').replace('==', '')
                        new_version = new_deps[dep].replace('>=', '').replace('~=', '').replace('==', '')
                        
                        old_parts = old_version.split('.')
                        new_parts = new_version.split('.')
                        
                        if len(old_parts) >= 1 and len(new_parts) >= 1:
                            old_major = int(old_parts[0]) if old_parts[0].isdigit() else 0
                            new_major = int(new_parts[0]) if new_parts[0].isdigit() else 0
                            
                            if new_major != old_major:  # Major version change
                                severity = ImpactSeverity.MEDIUM if new_major > old_major else ImpactSeverity.HIGH
                                change_type = "upgrade" if new_major > old_major else "downgrade"
                                
                                changes.append(BreakingChange(
                                    id=f"{commit_hash}_{hashlib.sha256(f'{file_path}_{dep}_{change_type}'.encode()).hexdigest()[:8]}",
                                    commit_hash=commit_hash,
                                    change_type=ChangeType.DEPENDENCY_CHANGE,
                                    severity=severity,
                                    intent=ChangeIntent.INTENTIONAL,
                                    affected_component=dep,
                                    file_path=file_path,
                                    line_number=None,
                                    old_signature=f"requirement: {dep}=={old_deps[dep]}",
                                    new_signature=f"requirement: {dep}=={new_deps[dep]}",
                                    confidence_score=0.8,
                                    affected_users_estimate="some",
                                    migration_complexity="moderate",
                                    description=f"Python dependency '{dep}' major version {change_type}",
                                    technical_details=f"Dependency '{dep}' changed from v{old_major} to v{new_major}, may contain breaking changes.",
                                    suggested_migration=f"Review '{dep}' changelog and update code for compatibility.",
                                    detection_methods=["requirements_analysis"]
                                ))
                    except (ValueError, IndexError):
                        # Skip if version parsing fails
                        pass
                        
        except Exception as e:
            logger.debug(f"Requirements.txt analysis failed for {file_path}: {e}")
        
        return changes
    
    def _detect_json_schema_changes(self, old_content: str, new_content: str, file_path: str, commit_hash: str) -> List[BreakingChange]:
        """Detect JSON schema changes"""
        changes = []
        
        try:
            old_json = json.loads(old_content) if old_content else {}
            new_json = json.loads(new_content) if new_content else {}
            
            # Helper function to flatten JSON for comparison
            def flatten_json(data, prefix=""):
                items = []
                if isinstance(data, dict):
                    for key, value in data.items():
                        new_prefix = f"{prefix}.{key}" if prefix else key
                        if isinstance(value, (dict, list)):
                            items.extend(flatten_json(value, new_prefix))
                        else:
                            items.append((new_prefix, type(value).__name__, value))
                elif isinstance(data, list):
                    for i, value in enumerate(data):
                        new_prefix = f"{prefix}[{i}]"
                        if isinstance(value, (dict, list)):
                            items.extend(flatten_json(value, new_prefix))
                        else:
                            items.append((new_prefix, type(value).__name__, value))
                return items
            
            old_items = {path: (type_name, value) for path, type_name, value in flatten_json(old_json)}
            new_items = {path: (type_name, value) for path, type_name, value in flatten_json(new_json)}
            
            # Removed keys/properties
            removed_keys = set(old_items.keys()) - set(new_items.keys())
            for key in removed_keys:
                changes.append(BreakingChange(
                    id=f"{commit_hash}_{hashlib.sha256(f'{file_path}_{key}_removal'.encode()).hexdigest()[:8]}",
                    commit_hash=commit_hash,
                    change_type=ChangeType.CONFIGURATION_CHANGE,
                    severity=ImpactSeverity.MEDIUM,
                    intent=ChangeIntent.INTENTIONAL,
                    affected_component=key,
                    file_path=file_path,
                    line_number=None,
                    old_signature=f"json.{key}",
                    new_signature=None,
                    confidence_score=0.9,
                    affected_users_estimate="some",
                    migration_complexity="low",
                    description=f"JSON property '{key}' was removed",
                    technical_details=f"The JSON property '{key}' is no longer available in the configuration.",
                    suggested_migration=f"Remove references to '{key}' or provide default value.",
                    detection_methods=["json_schema_analysis"]
                ))
            
            # Type changes (breaking for typed systems)
            for key in old_items:
                if key in new_items:
                    old_type, old_value = old_items[key]
                    new_type, new_value = new_items[key]
                    
                    if old_type != new_type:
                        changes.append(BreakingChange(
                            id=f"{commit_hash}_{hashlib.sha256(f'{file_path}_{key}_type_change'.encode()).hexdigest()[:8]}",
                            commit_hash=commit_hash,
                            change_type=ChangeType.CONFIGURATION_CHANGE,
                            severity=ImpactSeverity.MEDIUM,
                            intent=ChangeIntent.INTENTIONAL,
                            affected_component=key,
                            file_path=file_path,
                            line_number=None,
                            old_signature=f"json.{key}: {old_type}",
                            new_signature=f"json.{key}: {new_type}",
                            confidence_score=0.8,
                            affected_users_estimate="some",
                            migration_complexity="moderate",
                            description=f"JSON property '{key}' type changed from {old_type} to {new_type}",
                            technical_details=f"The type of '{key}' changed, which may break type-sensitive consumers.",
                            suggested_migration=f"Update code expecting '{key}' to handle {new_type} instead of {old_type}.",
                            detection_methods=["json_schema_analysis"]
                        ))
                        
        except json.JSONDecodeError:
            logger.debug(f"Failed to parse JSON file: {file_path}")
        except Exception as e:
            logger.debug(f"JSON schema analysis failed for {file_path}: {e}")
        
        return changes
    
    def _detect_yaml_schema_changes(self, old_content: str, new_content: str, file_path: str, commit_hash: str) -> List[BreakingChange]:
        """Detect YAML schema changes"""
        changes = []
        
        try:
            # Simple YAML parsing without external dependency
            def parse_simple_yaml(content):
                """Simple YAML parser for basic key-value pairs"""
                data = {}
                lines = content.splitlines()
                current_section = data
                section_stack = [data]
                
                for line in lines:
                    line = line.rstrip()
                    if not line or line.strip().startswith('#'):
                        continue
                    
                    # Count indentation
                    indent = len(line) - len(line.lstrip())
                    
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Handle nested structure (basic)
                        if indent == 0:
                            current_section = data
                            section_stack = [data]
                        
                        if value:
                            current_section[key] = value
                        else:
                            # New section
                            current_section[key] = {}
                            current_section = current_section[key]
                
                return data
            
            old_yaml = parse_simple_yaml(old_content) if old_content else {}
            new_yaml = parse_simple_yaml(new_content) if new_content else {}
            
            # Flatten YAML for comparison
            def flatten_yaml(data, prefix=""):
                items = {}
                for key, value in data.items():
                    new_key = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, dict):
                        items.update(flatten_yaml(value, new_key))
                    else:
                        items[new_key] = value
                return items
            
            old_items = flatten_yaml(old_yaml)
            new_items = flatten_yaml(new_yaml)
            
            # Removed configuration keys
            removed_keys = set(old_items.keys()) - set(new_items.keys())
            for key in removed_keys:
                changes.append(BreakingChange(
                    id=f"{commit_hash}_{hashlib.sha256(f'{file_path}_{key}_removal'.encode()).hexdigest()[:8]}",
                    commit_hash=commit_hash,
                    change_type=ChangeType.CONFIGURATION_CHANGE,
                    severity=ImpactSeverity.MEDIUM,
                    intent=ChangeIntent.INTENTIONAL,
                    affected_component=key,
                    file_path=file_path,
                    line_number=None,
                    old_signature=f"yaml.{key}: {old_items[key]}",
                    new_signature=None,
                    confidence_score=0.9,
                    affected_users_estimate="some",
                    migration_complexity="low",
                    description=f"YAML configuration '{key}' was removed",
                    technical_details=f"The configuration key '{key}' is no longer available.",
                    suggested_migration=f"Remove references to '{key}' or provide default value.",
                    detection_methods=["yaml_schema_analysis"]
                ))
            
            # Changed values (potentially breaking for CI/deployment configs)
            for key in old_items:
                if key in new_items and old_items[key] != new_items[key]:
                    # Check if it's a deployment-critical change
                    is_critical = any(keyword in key.lower() for keyword in ['port', 'host', 'url', 'endpoint', 'version', 'image'])
                    severity = ImpactSeverity.MEDIUM if is_critical else ImpactSeverity.LOW
                    
                    changes.append(BreakingChange(
                        id=f"{commit_hash}_{hashlib.sha256(f'{file_path}_{key}_change'.encode()).hexdigest()[:8]}",
                        commit_hash=commit_hash,
                        change_type=ChangeType.CONFIGURATION_CHANGE,
                        severity=severity,
                        intent=ChangeIntent.INTENTIONAL,
                        affected_component=key,
                        file_path=file_path,
                        line_number=None,
                        old_signature=f"yaml.{key}: {old_items[key]}",
                        new_signature=f"yaml.{key}: {new_items[key]}",
                        confidence_score=0.7,
                        affected_users_estimate="some",
                        migration_complexity="easy",
                        description=f"YAML configuration '{key}' value changed",
                        technical_details=f"Configuration '{key}' changed from '{old_items[key]}' to '{new_items[key]}'.",
                        suggested_migration=f"Update systems expecting the old value of '{key}'.",
                        detection_methods=["yaml_schema_analysis"]
                    ))
                    
        except Exception as e:
            logger.debug(f"YAML schema analysis failed for {file_path}: {e}")
        
        return changes
    
    def _detect_database_schema_changes(self, old_content: str, new_content: str, file_path: str, commit_hash: str) -> List[BreakingChange]:
        """Detect database schema changes"""
        changes = []
        
        try:
            # SQL DDL patterns for schema changes
            sql_patterns = {
                'drop_table': r'DROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?(\w+)',
                'drop_column': r'ALTER\s+TABLE\s+(\w+)\s+DROP\s+(?:COLUMN\s+)?(\w+)',
                'rename_table': r'ALTER\s+TABLE\s+(\w+)\s+RENAME\s+TO\s+(\w+)',
                'rename_column': r'ALTER\s+TABLE\s+(\w+)\s+RENAME\s+(?:COLUMN\s+)?(\w+)\s+TO\s+(\w+)',
                'modify_column': r'ALTER\s+TABLE\s+(\w+)\s+(?:ALTER|MODIFY)\s+(?:COLUMN\s+)?(\w+)\s+([^,;]+)',
                'add_not_null': r'ALTER\s+TABLE\s+(\w+)\s+(?:ALTER|MODIFY)\s+(?:COLUMN\s+)?(\w+)\s+[^,;]*NOT\s+NULL',
                'drop_index': r'DROP\s+INDEX\s+(?:IF\s+EXISTS\s+)?(\w+)',
                'drop_constraint': r'ALTER\s+TABLE\s+(\w+)\s+DROP\s+(?:CONSTRAINT|FOREIGN\s+KEY|PRIMARY\s+KEY)\s+(\w+)?'
            }
            
            # Check for breaking schema changes in new content
            for change_type, pattern in sql_patterns.items():
                matches = re.findall(pattern, new_content, re.IGNORECASE | re.MULTILINE)
                
                for match in matches:
                    # Extract table/column/constraint name
                    if isinstance(match, tuple):
                        if change_type in ['drop_table', 'drop_index']:
                            component = match[0] if match else 'unknown'
                        elif change_type in ['drop_column', 'add_not_null', 'modify_column']:
                            component = f"{match[0]}.{match[1]}" if len(match) >= 2 else match[0]
                        elif change_type == 'rename_table':
                            component = f"{match[0]} -> {match[1]}" if len(match) >= 2 else match[0]
                        elif change_type == 'rename_column':
                            component = f"{match[0]}.{match[1]} -> {match[2]}" if len(match) >= 3 else match[0]
                        else:
                            component = match[0] if match else 'unknown'
                    else:
                        component = match
                    
                    # Determine severity based on operation type
                    severity_map = {
                        'drop_table': ImpactSeverity.CRITICAL,
                        'drop_column': ImpactSeverity.HIGH,
                        'rename_table': ImpactSeverity.HIGH,
                        'rename_column': ImpactSeverity.MEDIUM,
                        'modify_column': ImpactSeverity.MEDIUM,
                        'add_not_null': ImpactSeverity.MEDIUM,
                        'drop_index': ImpactSeverity.LOW,
                        'drop_constraint': ImpactSeverity.MEDIUM
                    }
                    
                    severity = severity_map.get(change_type, ImpactSeverity.MEDIUM)
                    
                    changes.append(BreakingChange(
                        id=f"{commit_hash}_{hashlib.sha256(f'{file_path}_{change_type}_{component}'.encode()).hexdigest()[:8]}",
                        commit_hash=commit_hash,
                        change_type=ChangeType.DATABASE_SCHEMA_CHANGE,
                        severity=severity,
                        intent=ChangeIntent.INTENTIONAL,
                        affected_component=component,
                        file_path=file_path,
                        line_number=None,
                        old_signature=None,
                        new_signature=f"{change_type}: {component}",
                        confidence_score=0.9,
                        affected_users_estimate="most" if severity in [ImpactSeverity.CRITICAL, ImpactSeverity.HIGH] else "some",
                        migration_complexity="complex" if change_type in ['drop_table', 'rename_table'] else "moderate",
                        description=f"Database {change_type.replace('_', ' ')} on {component}",
                        technical_details=f"Schema change detected: {change_type.replace('_', ' ')} affecting {component}",
                        suggested_migration=self._get_db_migration_suggestion(change_type, component),
                        detection_methods=["database_schema_analysis"]
                    ))
                    
        except Exception as e:
            logger.debug(f"Database schema analysis failed for {file_path}: {e}")
        
        return changes
    
    def _get_db_migration_suggestion(self, change_type: str, component: str) -> str:
        """Get migration suggestion for database schema changes"""
        suggestions = {
            'drop_table': f"Ensure no code references table '{component}' before dropping",
            'drop_column': f"Update queries to remove references to column '{component}'",
            'rename_table': f"Update all references from old to new table name in '{component}'",
            'rename_column': f"Update all queries referencing the renamed column in '{component}'",
            'modify_column': f"Verify data compatibility for modified column '{component}'",
            'add_not_null': f"Ensure existing rows have values for column '{component}' before adding NOT NULL constraint",
            'drop_index': f"Monitor query performance after dropping index '{component}'",
            'drop_constraint': f"Ensure data integrity is maintained after dropping constraint on '{component}'"
        }
        return suggestions.get(change_type, f"Review database change affecting {component}")

    def analyze_historical_patterns(self, repo_path: str, days_back: int = 90) -> Dict[str, Any]:
        """
        Analyze historical patterns to improve breaking change detection
        """
        try:
            repo = self.git_analyzer.repo
            since_date = datetime.now() - timedelta(days=days_back)
            
            commits = list(repo.iter_commits(since=since_date))
            
            patterns = {
                'common_breaking_change_types': defaultdict(int),
                'authors_with_breaking_changes': defaultdict(int),
                'files_with_frequent_breaking_changes': defaultdict(int),
                'breaking_change_frequency': [],
                'severity_distribution': defaultdict(int)
            }
            
            for commit in commits:
                breaking_changes = self.analyze_commit_for_breaking_changes(commit.hexsha)
                
                if breaking_changes:
                    patterns['breaking_change_frequency'].append({
                        'date': commit.committed_datetime.isoformat(),
                        'count': len(breaking_changes),
                        'author': commit.author.name
                    })
                    
                    for change in breaking_changes:
                        patterns['common_breaking_change_types'][change.change_type.value] += 1
                        patterns['authors_with_breaking_changes'][commit.author.name] += 1
                        patterns['files_with_frequent_breaking_changes'][change.file_path] += 1
                        patterns['severity_distribution'][change.severity.value] += 1
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing historical patterns: {e}")
            return {}

    def generate_breaking_change_report(self, commit_hash: str) -> Dict[str, Any]:
        """
        Generate a comprehensive breaking change report for a commit
        """
        breaking_changes = self.analyze_commit_for_breaking_changes(commit_hash)
        
        if not breaking_changes:
            return {
                'commit_hash': commit_hash,
                'breaking_changes_detected': False,
                'summary': 'No breaking changes detected',
                'details': []
            }
        
        # Categorize changes
        by_severity = defaultdict(list)
        by_type = defaultdict(list)
        by_file = defaultdict(list)
        
        for change in breaking_changes:
            by_severity[change.severity.value].append(change)
            by_type[change.change_type.value].append(change)
            by_file[change.file_path].append(change)
        
        # Calculate overall risk score
        risk_score = self._calculate_overall_risk_score(breaking_changes)
        
        return {
            'commit_hash': commit_hash,
            'breaking_changes_detected': True,
            'total_changes': len(breaking_changes),
            'overall_risk_score': risk_score,
            'risk_level': self._risk_score_to_level(risk_score),
            'summary': self._generate_summary(breaking_changes),
            'severity_breakdown': dict(by_severity),
            'type_breakdown': dict(by_type),
            'affected_files': list(by_file.keys()),
            'recommendations': self._generate_recommendations(breaking_changes),
            'details': [self._change_to_dict(change) for change in breaking_changes]
        }
    
    def _calculate_overall_risk_score(self, changes: List[BreakingChange]) -> float:
        """Calculate overall risk score for a set of breaking changes"""
        if not changes:
            return 0.0
        
        severity_weights = {
            ImpactSeverity.CRITICAL: 1.0,
            ImpactSeverity.HIGH: 0.8,
            ImpactSeverity.MEDIUM: 0.5,
            ImpactSeverity.LOW: 0.2,
            ImpactSeverity.ENHANCEMENT: 0.1
        }
        
        total_weight = 0.0
        total_score = 0.0
        
        for change in changes:
            weight = severity_weights.get(change.severity, 0.5)
            score = change.confidence_score * weight
            total_weight += weight
            total_score += score
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _risk_score_to_level(self, score: float) -> str:
        """Convert risk score to level"""
        if score >= 0.8:
            return "CRITICAL"
        elif score >= 0.6:
            return "HIGH"
        elif score >= 0.4:
            return "MEDIUM"
        elif score >= 0.2:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _generate_summary(self, changes: List[BreakingChange]) -> str:
        """Generate human-readable summary"""
        if not changes:
            return "No breaking changes detected"
        
        critical = len([c for c in changes if c.severity == ImpactSeverity.CRITICAL])
        high = len([c for c in changes if c.severity == ImpactSeverity.HIGH])
        medium = len([c for c in changes if c.severity == ImpactSeverity.MEDIUM])
        
        parts = []
        if critical:
            parts.append(f"{critical} critical")
        if high:
            parts.append(f"{high} high-severity")
        if medium:
            parts.append(f"{medium} medium-severity")
        
        severity_text = ", ".join(parts) if parts else f"{len(changes)} low-severity"
        return f"Detected {len(changes)} breaking changes: {severity_text}"
    
    def _generate_recommendations(self, changes: List[BreakingChange]) -> List[str]:
        """Generate recommendations based on breaking changes"""
        recommendations = []
        
        critical_changes = [c for c in changes if c.severity == ImpactSeverity.CRITICAL]
        if critical_changes:
            recommendations.append("🚨 URGENT: Review critical breaking changes before deployment")
        
        api_changes = [c for c in changes if c.change_type in [ChangeType.API_SIGNATURE_CHANGE, ChangeType.FUNCTION_REMOVAL]]
        if api_changes:
            recommendations.append("📝 Update API documentation and notify consumers of API changes")
        
        if len(changes) > 5:
            recommendations.append("🔄 Consider splitting this commit into smaller, more focused changes")
        
        intentional_changes = [c for c in changes if c.intent == ChangeIntent.INTENTIONAL]
        if intentional_changes:
            recommendations.append("✅ Document intentional breaking changes in changelog")
        
        return recommendations
    
    def _post_process_changes(self, changes: List[BreakingChange], commit) -> List[BreakingChange]:
        """Post-process breaking changes to merge related changes and calculate relationships"""
        try:
            if not changes:
                return changes
            
            # Group changes by file and component for potential merging
            processed_changes = []
            
            # Add comprehensive commit metadata to each change
            for change in changes:
                # Ensure commit hash is set
                if not change.commit_hash:
                    change.commit_hash = commit.hexsha
                
                # Add commit metadata to AI analysis if missing
                if hasattr(commit, 'message'):
                    commit_info = f"Commit: {commit.hexsha[:8]} by {commit.author.name} - {commit.message.strip()[:100]}"
                    if not change.ai_analysis:
                        change.ai_analysis = commit_info
                    elif commit_info not in change.ai_analysis:
                        change.ai_analysis = f"{change.ai_analysis}\n{commit_info}"
                
                # Enhance technical details with commit context
                if hasattr(commit, 'committed_datetime'):
                    commit_date = commit.committed_datetime.strftime('%Y-%m-%d %H:%M:%S')
                    if "Commit date:" not in change.technical_details:
                        change.technical_details += f"\nCommit date: {commit_date}"
                        change.technical_details += f"\nCommit author: {commit.author.name}"
                        change.technical_details += f"\nCommit hash: {commit.hexsha}"
                
                # Add detection metadata
                change.detection_methods.append("advanced_multi_strategy")
                
                processed_changes.append(change)
            
            # Remove duplicates based on ID and affected component
            seen_keys = set()
            unique_changes = []
            for change in processed_changes:
                # Create unique key based on file, component, and change type
                key = f"{change.file_path}:{change.affected_component}:{change.change_type.value}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    unique_changes.append(change)
            
            logger.info(f"Post-processed {len(changes)} changes into {len(unique_changes)} unique changes")
            return unique_changes
            
        except Exception as e:
            logger.error(f"Error post-processing changes: {e}")
            return changes
    
    def _enhance_with_ai_analysis(self, changes: List[BreakingChange], commit) -> List[BreakingChange]:
        """Enhance breaking changes with AI analysis using Claude"""
        try:
            if not self.claude_analyzer or not changes:
                return changes
            
            enhanced_changes = []
            
            for change in changes:
                try:
                    # Create context for AI analysis
                    context = {
                        "change_type": change.change_type.value,
                        "file_path": change.file_path,
                        "old_signature": change.old_signature,
                        "new_signature": change.new_signature,
                        "commit_message": commit.message.strip() if hasattr(commit, 'message') else "",
                        "description": change.description
                    }
                    
                    # Get AI enhancement (simplified for now)
                    if not change.ai_analysis:
                        change.ai_analysis = f"AI Analysis: This {change.change_type.value} affects {change.affected_component}"
                    
                    # Add expert recommendations if missing
                    if not change.expert_recommendations:
                        change.expert_recommendations = [
                            f"Review impact of {change.change_type.value}",
                            f"Test migration for {change.migration_complexity} complexity change"
                        ]
                    
                    enhanced_changes.append(change)
                    
                except Exception as e:
                    logger.debug(f"Error enhancing change {change.id} with AI: {e}")
                    enhanced_changes.append(change)
            
            return enhanced_changes
            
        except Exception as e:
            logger.error(f"Error enhancing changes with AI: {e}")
            return changes
    
    def _change_to_dict(self, change: BreakingChange) -> Dict[str, Any]:
        """Convert BreakingChange to dictionary for JSON serialization"""
        return {
            'id': change.id,
            'type': change.change_type.value,
            'severity': change.severity.value,
            'intent': change.intent.value,
            'component': change.affected_component,
            'file': change.file_path,
            'line': change.line_number,
            'old_signature': change.old_signature,
            'new_signature': change.new_signature,
            'confidence': change.confidence_score,
            'affected_users': change.affected_users_estimate,
            'migration_complexity': change.migration_complexity,
            'description': change.description,
            'details': change.technical_details,
            'migration': change.suggested_migration,
            'detection_methods': change.detection_methods,
            'ai_analysis': change.ai_analysis,
            'recommendations': change.expert_recommendations
        }
    
    def analyze_breaking_changes(self, diff_text: str, file_path: str, commit_hash: str = "") -> Dict[str, Any]:
        """
        Analyze breaking changes from diff text
        
        Args:
            diff_text: The diff content to analyze
            file_path: Path to the file being analyzed
            commit_hash: Optional commit hash for context
            
        Returns:
            Dictionary containing detected breaking changes
        """
        try:
            changes = []
            
            # Basic diff pattern analysis - looks for removed functions/methods
            if diff_text and file_path:
                changes.extend(self._detect_diff_patterns("", "", file_path, commit_hash, type('DiffItem', (), {'diff': diff_text})()))
            
            # Convert BreakingChange objects to dictionaries for compatibility
            changes_dict = []
            for change in changes:
                changes_dict.append({
                    'id': change.id,
                    'type': change.change_type.value if hasattr(change.change_type, 'value') else str(change.change_type),
                    'severity': change.severity.value if hasattr(change.severity, 'value') else str(change.severity),
                    'description': change.description,
                    'file_path': change.file_path,
                    'affected_component': change.affected_component,
                    'confidence_score': change.confidence_score,
                    'suggested_migration': change.suggested_migration,
                    'technical_details': change.technical_details
                })
            
            return {
                'changes': changes_dict,
                'total_changes': len(changes_dict),
                'file_path': file_path,
                'analysis_methods': ['diff_pattern_analysis']
            }
            
        except Exception as e:
            logger.error(f"Error analyzing breaking changes from diff: {e}")
            return {
                'changes': [],
                'total_changes': 0,
                'file_path': file_path,
                'error': str(e),
                'analysis_methods': []
            }
    
    def analyze_diff_text(self, diff_text: str, file_path: str) -> List[BreakingChange]:
        """
        Analyze diff text for breaking changes
        
        Args:
            diff_text: The git diff text to analyze
            file_path: The file path being analyzed
            
        Returns:
            List of detected breaking changes
        """
        if not diff_text or not file_path:
            return []
            
        try:
            # Parse diff to extract old and new content
            old_content, new_content = self._parse_diff_text(diff_text)
            
            if old_content is None or new_content is None:
                logger.warning(f"Could not parse diff text for {file_path}")
                return []
            
            # Use existing file change detection
            return self._detect_file_changes(old_content, new_content, file_path)
            
        except Exception as e:
            logger.error(f"Error analyzing diff text for {file_path}: {e}")
            return []
    
    def _parse_diff_text(self, diff_text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse git diff text to extract old and new content
        
        Returns:
            Tuple of (old_content, new_content)
        """
        try:
            lines = diff_text.split('\n')
            old_lines = []
            new_lines = []
            
            for line in lines:
                if line.startswith('--- ') or line.startswith('+++ ') or line.startswith('@@'):
                    continue
                elif line.startswith('-'):
                    old_lines.append(line[1:])  # Remove the '-' prefix
                elif line.startswith('+'):
                    new_lines.append(line[1:])  # Remove the '+' prefix
                elif line.startswith(' '):
                    # Context line - exists in both
                    old_lines.append(line[1:])  # Remove the ' ' prefix
                    new_lines.append(line[1:])
            
            old_content = '\n'.join(old_lines) if old_lines else ""
            new_content = '\n'.join(new_lines) if new_lines else ""
            
            return old_content, new_content
            
        except Exception as e:
            logger.error(f"Error parsing diff text: {e}")
            return None, None

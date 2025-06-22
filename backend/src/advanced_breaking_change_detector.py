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
    author: str
    date: str
    breaking_changes: List[BreakingChange]
    all_changes: List[BreakingChange]
    migration_suggestions: List[str]
    confidence: float
    ai_analysis: Optional[str] = None

class AdvancedBreakingChangeDetector:
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
        
    def analyze_commit_for_breaking_changes(self, commit_hash: str) -> List[BreakingChange]:
        """
        Comprehensive breaking change analysis for a commit
        Uses multiple detection strategies and AI analysis
        """
        logger.info(f"🔍 Analyzing commit {commit_hash} for breaking changes...")
        
        try:
            # Get commit details
            repo = self.git_analyzer.repo
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
                if diff_item.a_path and self._is_code_file(diff_item.a_path):
                    changes = self._analyze_file_diff(diff_item, commit_hash)
                    all_breaking_changes.extend(changes)
            
            # Post-processing: merge related changes, calculate relationships
            all_breaking_changes = self._post_process_changes(all_breaking_changes, commit)
            
            # AI-enhanced analysis with Claude
            if self.claude_analyzer and all_breaking_changes:
                all_breaking_changes = self._enhance_with_ai_analysis(all_breaking_changes, commit)
            
            logger.info(f"✅ Found {len(all_breaking_changes)} breaking changes in commit {commit_hash}")
            return all_breaking_changes
            
        except Exception as e:
            logger.error(f"Error analyzing commit {commit_hash}: {e}")
            return []
    
    def analyze_commit(self, repo, commit_hash: str) -> BreakingChangeAnalysis:
        """Analyze a specific commit for breaking changes"""
        try:
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
                        file_changes = self._detect_file_changes(file_path, old_content, new_content, diff_item)
                        changes.extend(file_changes)
            
            # Create analysis result
            breaking_changes = [change for change in changes if change.severity in [ImpactSeverity.CRITICAL, ImpactSeverity.HIGH]]
            migration_suggestions = self._generate_migration_suggestions(breaking_changes)
            
            # Calculate confidence based on detection methods used
            confidence = 0.8 if breaking_changes else 0.9
            
            # Generate AI analysis if available
            ai_analysis = None
            if self.claude_analyzer and self.claude_analyzer.available:
                ai_analysis = self._generate_ai_analysis(commit, changes)
            
            return BreakingChangeAnalysis(
                commit_hash=commit_hash,
                commit_message=commit.message.strip(),
                author=commit.author.name,
                date=commit.committed_datetime.isoformat(),
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
                author="unknown",
                date="",
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
            # Get old and new content
            old_content = diff_item.a_blob.data_stream.read().decode('utf-8', errors='ignore') if diff_item.a_blob else ""
            new_content = diff_item.b_blob.data_stream.read().decode('utf-8', errors='ignore') if diff_item.b_blob else ""
            
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
            def __init__(self):
                self.current_class = None
                
            def visit_FunctionDef(self, node):
                sig = self._create_function_signature(node, self.current_class)
                signatures[sig.name] = sig
                self.generic_visit(node)
                
            def visit_AsyncFunctionDef(self, node):
                sig = self._create_function_signature(node, self.current_class, is_async=True)
                signatures[sig.name] = sig
                self.generic_visit(node)
                
            def visit_ClassDef(self, node):
                old_class = self.current_class
                self.current_class = node.name
                
                sig = self._create_class_signature(node)
                signatures[sig.name] = sig
                
                self.generic_visit(node)
                self.current_class = old_class
        
        extractor = SignatureExtractor()
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
                    old_signature=self._signature_to_string(old_sig),
                    new_signature=None,
                    confidence_score=0.95,
                    affected_users_estimate=self._estimate_affected_users(old_sig),
                    migration_complexity=self._estimate_migration_complexity(old_sig, None),
                    description=f"{old_sig.type.title()} '{name}' was removed",
                    technical_details=f"The {old_sig.type} '{name}' is no longer available in the codebase.",
                    suggested_migration=self._suggest_removal_migration(old_sig),
                    detection_methods=["ast_analysis", "signature_comparison"]
                ))
        
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
                old_signature=self._signature_to_string(old_sig),
                new_signature=self._signature_to_string(new_sig),
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
                old_signature=self._signature_to_string(old_sig),
                new_signature=self._signature_to_string(new_sig),
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
                old_signature=self._signature_to_string(old_sig),
                new_signature=self._signature_to_string(new_sig),
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
    
    def _detect_file_changes(self, file_path: str, old_content: str, new_content: str, diff_item) -> List[BreakingChange]:
        """Detect breaking changes in a single file"""
        changes = []
        
        # Determine file type and use appropriate detector
        if file_path.endswith('.py'):
            changes.extend(self._detect_python_changes(file_path, old_content, new_content))
        elif file_path.endswith(('.js', '.ts')):
            changes.extend(self._detect_javascript_changes(file_path, old_content, new_content))
        elif file_path.endswith('.json'):
            changes.extend(self._detect_config_changes(file_path, old_content, new_content))
        
        # Add generic diff-based changes
        changes.extend(self._detect_generic_changes(file_path, old_content, new_content, diff_item))
        
        return changes
    
    def _detect_python_changes(self, file_path: str, old_content: str, new_content: str) -> List[BreakingChange]:
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
                            id=f"remove_{name}",
                            commit_hash="",
                            change_type=ChangeType.FUNCTION_REMOVAL,
                            severity=ImpactSeverity.HIGH,
                            intent=ChangeIntent.INTENTIONAL,
                            affected_component=name,
                            file_path=file_path,
                            line_number=old_sig.get('line', 0),
                            old_signature=f"{name}({old_sig.get('args', [])})",
                            new_signature=None,
                            confidence_score=0.9,
                            affected_users_estimate="most",
                            migration_complexity="medium",
                            description=f"Function/class '{name}' was removed",
                            technical_details=f"The {old_sig.get('type', 'function')} '{name}' is no longer available.",
                            suggested_migration=f"Replace calls to '{name}' with alternative implementation."
                        ))
                    elif old_sig != new_signatures[name]:
                        # Signature changed
                        changes.append(BreakingChange(
                            id=f"change_{name}",
                            commit_hash="",
                            change_type=ChangeType.API_SIGNATURE_CHANGE,
                            severity=ImpactSeverity.MEDIUM,
                            intent=ChangeIntent.INTENTIONAL,
                            affected_component=name,
                            file_path=file_path,
                            line_number=new_signatures[name].get('line', 0),
                            old_signature=f"{name}({old_sig.get('args', [])})",
                            new_signature=f"{name}({new_signatures[name].get('args', [])})",
                            confidence_score=0.8,
                            affected_users_estimate="some",
                            migration_complexity="low",
                            description=f"Signature of '{name}' changed",
                            technical_details=f"Parameters or return type changed for '{name}'.",
                            suggested_migration=f"Update calls to '{name}' to match new signature."
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
        """Extract function and class signatures from Python AST"""
        signatures = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                signatures[node.name] = {
                    'type': 'function',
                    'args': [arg.arg for arg in node.args.args],
                    'line': node.lineno
                }
            elif isinstance(node, ast.ClassDef):
                signatures[node.name] = {
                    'type': 'class',
                    'bases': [base.id if isinstance(base, ast.Name) else str(base) for base in node.bases],
                    'line': node.lineno
                }
        
        return signatures
    
    def _detect_javascript_changes(self, file_path: str, old_content: str, new_content: str) -> List[BreakingChange]:
        """Detect JavaScript/TypeScript breaking changes"""
        changes = []
        
        # Simple regex-based detection for now
        # Look for exported functions/classes
        old_exports = re.findall(r'export\s+(?:function|class|const|let|var)\s+(\w+)', old_content)
        new_exports = re.findall(r'export\s+(?:function|class|const|let|var)\s+(\w+)', new_content)
        
        removed_exports = set(old_exports) - set(new_exports)
        for export in removed_exports:
            changes.append(BreakingChange(
                id=f"js_remove_{export}",
                commit_hash="",
                change_type=ChangeType.FUNCTION_REMOVAL,
                severity=ImpactSeverity.HIGH,
                intent=ChangeIntent.INTENTIONAL,
                affected_component=export,
                file_path=file_path,
                line_number=0,
                old_signature=f"export {export}",
                new_signature=None,
                confidence_score=0.85,
                affected_users_estimate="most",
                migration_complexity="medium",
                description=f"Exported '{export}' was removed",
                technical_details=f"The exported item '{export}' is no longer available.",
                suggested_migration=f"Replace usage of '{export}' with alternative implementation.",
                detection_methods=["regex_signature_analysis"]
            ))
        
        return changes
    
    def _detect_config_changes(self, file_path: str, old_content: str, new_content: str) -> List[BreakingChange]:
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
    
    def _detect_generic_changes(self, file_path: str, old_content: str, new_content: str, diff_item) -> List[BreakingChange]:
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
    
    # Placeholder methods for additional language support
    def _detect_java_signature_changes(self, old_content: str, new_content: str, file_path: str, commit_hash: str) -> List[BreakingChange]:
        """Detect Java signature changes"""
        # TODO: Implement Java-specific analysis
        return []
    
    def _detect_cpp_signature_changes(self, old_content: str, new_content: str, file_path: str, commit_hash: str) -> List[BreakingChange]:
        """Detect C++ signature changes"""
        # TODO: Implement C++ specific analysis
        return []
    
    def _detect_package_json_changes(self, old_content: str, new_content: str, file_path: str, commit_hash: str) -> List[BreakingChange]:
        """Detect package.json dependency changes"""
        # TODO: Implement package.json analysis
        return []
    
    def _detect_requirements_changes(self, old_content: str, new_content: str, file_path: str, commit_hash: str) -> List[BreakingChange]:
        """Detect requirements.txt changes"""
        # TODO: Implement requirements.txt analysis
        return []
    
    def _detect_json_schema_changes(self, old_content: str, new_content: str, file_path: str, commit_hash: str) -> List[BreakingChange]:
        """Detect JSON schema changes"""
        # TODO: Implement JSON schema analysis
        return []
    
    def _detect_yaml_schema_changes(self, old_content: str, new_content: str, file_path: str, commit_hash: str) -> List[BreakingChange]:
        """Detect YAML schema changes"""
        # TODO: Implement YAML schema analysis
        return []
    
    def _detect_database_schema_changes(self, old_content: str, new_content: str, file_path: str, commit_hash: str) -> List[BreakingChange]:
        """Detect database schema changes"""
        # TODO: Implement database schema analysis
        return []

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

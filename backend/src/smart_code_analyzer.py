"""
Smart Code Analyzer - Deep Understanding of Code Structure
Provides intelligent context for Claude by analyzing code architecture, patterns, and relationships
"""

import ast
import os
import re
import logging
from typing import Dict, Any, List, Set, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class CodeEntity:
    """Represents a code entity (class, function, variable)"""
    name: str
    type: str  # 'class', 'function', 'variable', 'import'
    file_path: str
    line_number: int
    complexity: int = 0
    dependencies: List[str] = None
    docstring: Optional[str] = None
    is_public: bool = True
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class FileAnalysis:
    """Analysis results for a single file"""
    file_path: str
    language: str
    entities: List[CodeEntity]
    imports: List[str]
    complexity_score: float
    test_coverage_indicators: List[str]
    security_patterns: List[str]
    architecture_patterns: List[str]
    lines_of_code: int
    technical_debt_indicators: List[str]

@dataclass
class ArchitectureAnalysis:
    """High-level architecture analysis"""
    patterns_detected: List[str]
    layer_structure: Dict[str, List[str]]
    dependency_graph: Dict[str, List[str]]
    entry_points: List[str]
    data_flow: List[Dict[str, Any]]
    potential_bottlenecks: List[str]

class SmartCodeAnalyzer:
    """Intelligent code analyzer that understands structure and patterns"""
    
    def __init__(self):
        self.supported_languages = {
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
            '.php': 'php'
        }
        
        # Security patterns to detect
        self.security_patterns = {
            'auth_issues': [
                r'password.*=.*["\']', r'api_key.*=.*["\']', r'secret.*=.*["\']',
                r'token.*=.*["\']', r'auth.*=.*["\']'
            ],
            'sql_injection': [
                r'execute.*\+.*', r'query.*\+.*', r'SELECT.*\+.*',
                r'INSERT.*\+.*', r'UPDATE.*\+.*', r'DELETE.*\+.*'
            ],
            'xss_patterns': [
                r'innerHTML.*\+.*', r'document\.write\(.*\+.*',
                r'eval\(.*\+.*', r'setTimeout\(.*\+.*'
            ],
            'file_inclusion': [
                r'include.*\$_', r'require.*\$_', r'file_get_contents.*\$_'
            ]
        }
        
        # Architecture patterns
        self.architecture_patterns = {
            'mvc': ['model', 'view', 'controller'],
            'microservices': ['service', 'api', 'gateway'],
            'layered': ['presentation', 'business', 'data', 'persistence'],
            'repository': ['repository', 'dao', 'entity'],
            'factory': ['factory', 'builder', 'creator'],
            'observer': ['observer', 'listener', 'subscriber'],
            'strategy': ['strategy', 'algorithm', 'policy']
        }
        
        # Technical debt indicators
        self.tech_debt_patterns = [
            'TODO', 'FIXME', 'HACK', 'XXX', 'BUG',
            'temporary', 'quick fix', 'workaround'
        ]
    
    def analyze_file(self, file_path: str, content: str) -> Optional[FileAnalysis]:
        """Analyze a single file and extract structure information"""
        try:
            ext = Path(file_path).suffix.lower()
            language = self.supported_languages.get(ext, 'text')
            
            if language == 'python':
                return self._analyze_python_file(file_path, content)
            elif language in ['javascript', 'typescript']:
                return self._analyze_js_file(file_path, content, language)
            else:
                return self._analyze_generic_file(file_path, content, language)
                
        except Exception as e:
            logger.debug(f"Error analyzing file {file_path}: {e}")
            return None
    
    def _analyze_python_file(self, file_path: str, content: str) -> FileAnalysis:
        """Analyze Python file using AST"""
        entities = []
        imports = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    entities.append(CodeEntity(
                        name=node.name,
                        type='class',
                        file_path=file_path,
                        line_number=node.lineno,
                        complexity=self._calculate_complexity(node),
                        docstring=ast.get_docstring(node),
                        is_public=not node.name.startswith('_')
                    ))
                
                elif isinstance(node, ast.FunctionDef):
                    entities.append(CodeEntity(
                        name=node.name,
                        type='function',
                        file_path=file_path,
                        line_number=node.lineno,
                        complexity=self._calculate_complexity(node),
                        docstring=ast.get_docstring(node),
                        is_public=not node.name.startswith('_')
                    ))
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    else:
                        imports.append(node.module or '')
                        
        except SyntaxError:
            logger.debug(f"Syntax error in Python file: {file_path}")
        
        return FileAnalysis(
            file_path=file_path,
            language='python',
            entities=entities,
            imports=imports,
            complexity_score=self._calculate_file_complexity(entities),
            test_coverage_indicators=self._find_test_indicators(content),
            security_patterns=self._find_security_patterns(content),
            architecture_patterns=self._find_architecture_patterns(file_path, content),
            lines_of_code=len([line for line in content.split('\n') if line.strip()]),
            technical_debt_indicators=self._find_tech_debt(content)
        )
    
    def _analyze_js_file(self, file_path: str, content: str, language: str) -> FileAnalysis:
        """Analyze JavaScript/TypeScript file"""
        entities = []
        imports = []
        
        # Extract functions using regex (simplified approach)
        function_pattern = r'(?:function\s+|const\s+|let\s+|var\s+)(\w+)\s*(?:=\s*(?:async\s+)?(?:\(\s*\)|function)|\()'
        for match in re.finditer(function_pattern, content):
            line_no = content[:match.start()].count('\n') + 1
            entities.append(CodeEntity(
                name=match.group(1),
                type='function',
                file_path=file_path,
                line_number=line_no,
                complexity=1,  # Simplified complexity
                is_public=not match.group(1).startswith('_')
            ))
        
        # Extract classes
        class_pattern = r'class\s+(\w+)'
        for match in re.finditer(class_pattern, content):
            line_no = content[:match.start()].count('\n') + 1
            entities.append(CodeEntity(
                name=match.group(1),
                type='class',
                file_path=file_path,
                line_number=line_no,
                complexity=1,
                is_public=not match.group(1).startswith('_')
            ))
        
        # Extract imports
        import_patterns = [
            r'import\s+.*\s+from\s+["\']([^"\']+)["\']',
            r'import\s+["\']([^"\']+)["\']',
            r'require\s*\(\s*["\']([^"\']+)["\']\s*\)'
        ]
        
        for pattern in import_patterns:
            for match in re.finditer(pattern, content):
                imports.append(match.group(1))
        
        return FileAnalysis(
            file_path=file_path,
            language=language,
            entities=entities,
            imports=imports,
            complexity_score=self._calculate_file_complexity(entities),
            test_coverage_indicators=self._find_test_indicators(content),
            security_patterns=self._find_security_patterns(content),
            architecture_patterns=self._find_architecture_patterns(file_path, content),
            lines_of_code=len([line for line in content.split('\n') if line.strip()]),
            technical_debt_indicators=self._find_tech_debt(content)
        )
    
    def _analyze_generic_file(self, file_path: str, content: str, language: str) -> FileAnalysis:
        """Analyze non-code files or unsupported languages"""
        return FileAnalysis(
            file_path=file_path,
            language=language,
            entities=[],
            imports=[],
            complexity_score=0,
            test_coverage_indicators=self._find_test_indicators(content),
            security_patterns=self._find_security_patterns(content),
            architecture_patterns=self._find_architecture_patterns(file_path, content),
            lines_of_code=len([line for line in content.split('\n') if line.strip()]),
            technical_debt_indicators=self._find_tech_debt(content)
        )
    
    def _calculate_complexity(self, node) -> int:
        """Calculate cyclomatic complexity for AST node"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
        
        return complexity
    
    def _calculate_file_complexity(self, entities: List[CodeEntity]) -> float:
        """Calculate overall file complexity score"""
        if not entities:
            return 0.0
        
        total_complexity = sum(entity.complexity for entity in entities)
        return min(total_complexity / len(entities), 10.0)  # Cap at 10
    
    def _find_test_indicators(self, content: str) -> List[str]:
        """Find indicators of test coverage"""
        indicators = []
        content_lower = content.lower()
        
        test_patterns = [
            'test_', 'def test', 'it(', 'describe(', 'expect(',
            'assert', 'mock', 'stub', 'spy', '@test'
        ]
        
        for pattern in test_patterns:
            if pattern in content_lower:
                indicators.append(pattern)
        
        return indicators
    
    def _find_security_patterns(self, content: str) -> List[str]:
        """Find potential security issues"""
        found_patterns = []
        content_lower = content.lower()
        
        for category, patterns in self.security_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    found_patterns.append(f"{category}: {pattern}")
        
        return found_patterns
    
    def _find_architecture_patterns(self, file_path: str, content: str) -> List[str]:
        """Detect architecture patterns"""
        patterns = []
        file_path_lower = file_path.lower()
        content_lower = content.lower()
        
        for pattern_name, keywords in self.architecture_patterns.items():
            if any(keyword in file_path_lower or keyword in content_lower for keyword in keywords):
                patterns.append(pattern_name)
        
        return patterns
    
    def _find_tech_debt(self, content: str) -> List[str]:
        """Find technical debt indicators"""
        debt_found = []
        
        for pattern in self.tech_debt_patterns:
            if pattern.lower() in content.lower():
                debt_found.append(pattern)
        
        return debt_found
    
    def analyze_architecture(self, file_analyses: List[FileAnalysis]) -> ArchitectureAnalysis:
        """Analyze overall architecture from file analyses"""
        patterns_detected = set()
        layer_structure = {}
        dependency_graph = {}
        entry_points = []
        
        # Aggregate patterns
        for analysis in file_analyses:
            patterns_detected.update(analysis.architecture_patterns)
            
            # Group files by layer/pattern
            for pattern in analysis.architecture_patterns:
                if pattern not in layer_structure:
                    layer_structure[pattern] = []
                layer_structure[pattern].append(analysis.file_path)
            
            # Build dependency graph
            file_name = Path(analysis.file_path).stem
            dependency_graph[file_name] = analysis.imports
            
            # Identify entry points (main, index, app files)
            if any(keyword in analysis.file_path.lower() 
                   for keyword in ['main', 'index', 'app', 'server', 'entry']):
                entry_points.append(analysis.file_path)
        
        # Identify potential bottlenecks
        bottlenecks = []
        complexity_scores = [a.complexity_score for a in file_analyses if a.complexity_score > 0]
        if complexity_scores:
            avg_complexity = sum(complexity_scores) / len(complexity_scores)
            for analysis in file_analyses:
                if analysis.complexity_score > avg_complexity * 1.5:
                    bottlenecks.append(f"{analysis.file_path} (complexity: {analysis.complexity_score:.1f})")
        
        return ArchitectureAnalysis(
            patterns_detected=list(patterns_detected),
            layer_structure=layer_structure,
            dependency_graph=dependency_graph,
            entry_points=entry_points,
            data_flow=[],  # TODO: Implement data flow analysis
            potential_bottlenecks=bottlenecks
        )
    
    def get_smart_context_for_query(self, query: str, file_analyses: List[FileAnalysis]) -> Dict[str, Any]:
        """Generate smart context based on query intent and code analysis"""
        query_lower = query.lower()
        context = {
            'relevant_files': [],
            'code_insights': [],
            'security_concerns': [],
            'architecture_insights': [],
            'recommendations': []
        }
        
        # Security-focused queries
        if any(word in query_lower for word in ['security', 'vulnerability', 'auth', 'password', 'token']):
            for analysis in file_analyses:
                if analysis.security_patterns:
                    context['relevant_files'].append(analysis)
                    context['security_concerns'].extend(analysis.security_patterns)
                    context['recommendations'].append(f"Review security patterns in {analysis.file_path}")
        
        # Architecture-focused queries  
        elif any(word in query_lower for word in ['architecture', 'design', 'structure', 'pattern']):
            architecture = self.analyze_architecture(file_analyses)
            context['architecture_insights'] = {
                'patterns': architecture.patterns_detected,
                'layers': architecture.layer_structure,
                'bottlenecks': architecture.potential_bottlenecks
            }
            context['recommendations'].extend([
                f"Consider refactoring high-complexity files: {', '.join(architecture.potential_bottlenecks)}",
                f"Architecture patterns detected: {', '.join(architecture.patterns_detected)}"
            ])
        
        # Code quality queries
        elif any(word in query_lower for word in ['quality', 'complexity', 'debt', 'refactor']):
            high_complexity_files = [a for a in file_analyses if a.complexity_score > 5]
            tech_debt_files = [a for a in file_analyses if a.technical_debt_indicators]
            
            context['code_insights'] = {
                'high_complexity_files': [(a.file_path, a.complexity_score) for a in high_complexity_files],
                'tech_debt_files': [(a.file_path, a.technical_debt_indicators) for a in tech_debt_files]
            }
            
            if high_complexity_files:
                context['recommendations'].append(f"Refactor complex files: {', '.join([a.file_path for a in high_complexity_files[:3]])}")
            if tech_debt_files:
                context['recommendations'].append(f"Address technical debt in: {', '.join([a.file_path for a in tech_debt_files[:3]])}")
        
        # Performance queries
        elif any(word in query_lower for word in ['performance', 'optimization', 'speed', 'bottleneck']):
            architecture = self.analyze_architecture(file_analyses)
            context['relevant_files'] = [a for a in file_analyses if a.complexity_score > 3]
            context['code_insights'].extend(architecture.potential_bottlenecks)
            context['recommendations'].append("Focus on optimizing high-complexity components")
        
        # General queries - return most relevant files based on entities
        else:
            # Score files by relevance to query
            scored_files = []
            for analysis in file_analyses:
                score = 0
                for entity in analysis.entities:
                    if any(word in entity.name.lower() for word in query_lower.split()):
                        score += 10
                    if entity.docstring and any(word in entity.docstring.lower() for word in query_lower.split()):
                        score += 5
                
                if any(word in analysis.file_path.lower() for word in query_lower.split()):
                    score += 15
                
                if score > 0:
                    scored_files.append((analysis, score))
            
            # Sort by relevance and take top files
            scored_files.sort(key=lambda x: x[1], reverse=True)
            context['relevant_files'] = [analysis for analysis, score in scored_files[:10]]
        
        return context

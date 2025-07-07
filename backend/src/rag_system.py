"""
Enhanced RAG System - Intelligent Context with Deep Code Understanding
Uses smart code analysis to provide highly relevant context to Claude
"""

import re
import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

from .code_analyzer import CodeAnalyzer, FileAnalysis, ArchitectureAnalysis
from .breaking_change_detector import BreakingChangeDetector

logger = logging.getLogger(__name__)

@dataclass 
class QueryResult:
    """Query result with deep code insights"""
    query: str
    response: str
    confidence: float
    sources: List[Dict[str, Any]]
    context_used: List[str]
    suggestions: List[str]
    code_insights: Dict[str, Any]
    architecture_analysis: Optional[Dict[str, Any]]
    security_analysis: Optional[Dict[str, Any]]
    quality_metrics: Optional[Dict[str, Any]]
    breaking_change_analysis: Optional[Dict[str, Any]]

class RAGSystem:
    """RAG with deep code understanding and intelligent context gathering"""
    
    def __init__(self, git_analyzer, claude_analyzer=None, embedding_engine=None):
        self.git_analyzer = git_analyzer
        self.claude_analyzer = claude_analyzer
        self.embedding_engine = embedding_engine
        self.code_analyzer = CodeAnalyzer()
        # Initialize breaking change detector only if git_analyzer is available
        self.breaking_change_detector = None
        if git_analyzer:
            self.breaking_change_detector = BreakingChangeDetector(git_analyzer, claude_analyzer)
        self.cached_analyses = {}  # Cache file analyses
        
    def query(self, repo_id: str, query: str, max_results: int = 10) -> QueryResult:
        """
        Main query method that handles both general code questions and breaking change analysis
        Uses semantic search, code analysis, and breaking change detection as needed
        """
        try:
            # Step 1: Analyze query intent to determine approach
            query_intent = self._analyze_query_intent(query)
            
            # Step 2: Check if this is a breaking change related query
            is_breaking_change_query = (
                query_intent.get('primary_domain') == 'breaking_changes' or
                any(keyword in query.lower() for keyword in [
                    'breaking', 'break', 'risk', 'danger', 'critical', 'migration', 
                    'compatibility', 'change', 'impact', 'semantic drift'
                ])
            )
            
            # Step 3: Gather relevant context using semantic search
            relevant_files = []
            relevant_commits = []
            
            if self.git_analyzer:
                # Use semantic search to find relevant files
                relevant_files = self._gather_intelligent_context(repo_id, query, query_intent)
                
                # Get relevant commits based on query
                relevant_commits = self._gather_intelligent_commits(repo_id, query, query_intent)
            
            # Step 4: Perform file analysis on relevant files - analyze ALL files for comprehensive context
            file_analyses = self._analyze_relevant_files(relevant_files)  # Analyze ALL loaded files, not just a subset
            
            # Step 5: Breaking change analysis if relevant
            breaking_change_analysis = None
            if is_breaking_change_query and self.breaking_change_detector:
                logger.info(f"🔍 Performing breaking change analysis for query: {query}")
                breaking_change_analysis = self._analyze_breaking_changes(relevant_commits, file_analyses, query_intent)
            
            # Step 6: Create specialized analysis based on query type
            specialized_analysis = self._create_specialized_analysis(query_intent, file_analyses, relevant_commits)
            
            # Step 7: Get smart context from code analyzer
            smart_context = {}
            if file_analyses:
                smart_context = self.code_analyzer.get_smart_context_for_query(query, file_analyses)
            
            # Step 8: Intelligent curation for LLM - select most relevant context
            curated_file_analyses = self._curate_files_for_llm(file_analyses, query_intent, query, max_llm_files=15)
            curated_commits = self._curate_commits_for_llm(relevant_commits, query_intent, query, max_llm_commits=8)
            
            logger.info(f"🎯 Curated context: {len(curated_file_analyses)}/{len(file_analyses)} files, {len(curated_commits)}/{len(relevant_commits)} commits for LLM")
            
            # Step 9: Generate response with curated context
            if self.claude_analyzer and self.claude_analyzer.available:
                claude_response = self._enhanced_claude_analysis(
                    query, query_intent, smart_context, specialized_analysis, 
                    curated_file_analyses, breaking_change_analysis
                )
                
                return QueryResult(
                    query=query,
                    response=claude_response['response'],
                    confidence=claude_response['confidence'],
                    sources=claude_response.get('sources', []),
                    context_used=claude_response.get('context_used', []),
                    suggestions=claude_response.get('suggestions', []),
                    code_insights=smart_context.get('code_insights', {}),
                    architecture_analysis=specialized_analysis.get('architecture'),
                    security_analysis=specialized_analysis.get('security'),
                    quality_metrics=specialized_analysis.get('quality'),
                    breaking_change_analysis=breaking_change_analysis
                )
            else:
                # Fallback response when Claude is not available
                return self._create_fallback_response(query, smart_context, specialized_analysis, breaking_change_analysis)
                
        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            return QueryResult(
                query=query,
                response=f"I encountered an error while analyzing your query: {str(e)}",
                confidence=0.0,
                sources=[],
                context_used=[],
                suggestions=["Please try rephrasing your question", "Check if the repository is properly loaded"],
                code_insights={},
                architecture_analysis=None,
                security_analysis=None,
                quality_metrics=None,
                breaking_change_analysis=None
            )
    
    def enhanced_query(self, repo_id: str, query: str, max_results: int = 10) -> QueryResult:
        """Enhanced query with deep code understanding"""
        
        # Step 1: Understand query intent with enhanced analysis
        query_intent = self._analyze_query_intent(query)
        
        # Step 2: Gather relevant files with intelligent filtering
        relevant_files = self._gather_intelligent_context(repo_id, query, query_intent)
        
        # Step 3: Perform deep code analysis on relevant files
        file_analyses = self._analyze_relevant_files(relevant_files)
        
        # Step 4: Get smart context specific to query intent
        smart_context = self.code_analyzer.get_smart_context_for_query(query, file_analyses)
        
        # Step 5: Analyze commits with enhanced intelligence
        relevant_commits = self._gather_intelligent_commits(repo_id, query, query_intent)
        
        # Step 6: Create specialized analysis based on query type
        specialized_analysis = self._create_specialized_analysis(query_intent, file_analyses, relevant_commits)
        
        # Step 7: Analyze breaking changes if relevant
        breaking_change_analysis = None
        if (query_intent['primary_domain'] == 'breaking_changes' or 
            'breaking' in query.lower() or 
            'risk' in query.lower() or 
            'compatibility' in query.lower() or 
            'migration' in query.lower()):
            breaking_change_analysis = self._analyze_breaking_changes(relevant_commits, file_analyses, query_intent)
        
        # Step 8: Generate response with Claude if available
        if self.claude_analyzer and self.claude_analyzer.available:
            claude_response = self._enhanced_claude_analysis(
                query, query_intent, smart_context, specialized_analysis, file_analyses, breaking_change_analysis
            )
            
            return QueryResult(
                query=query,
                response=claude_response['response'],
                confidence=claude_response['confidence'],
                sources=claude_response['sources'],
                context_used=claude_response['context_used'],
                suggestions=claude_response['suggestions'],
                code_insights=smart_context.get('code_insights', {}),
                architecture_analysis=specialized_analysis.get('architecture', None),
                security_analysis=specialized_analysis.get('security', None),
                quality_metrics=specialized_analysis.get('quality', None),
                breaking_change_analysis=breaking_change_analysis
            )
        else:
            return self._create_fallback_response(query, smart_context, specialized_analysis, breaking_change_analysis)
    
    def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Enhanced query intent analysis with domain detection and commit hash detection"""
        query_lower = query.lower()
        
        intent = {
            'primary_domain': 'general',
            'secondary_domains': [],
            'focus_areas': [],
            'complexity_level': 'medium',
            'keywords': re.findall(r'\b\w+\b', query_lower),
            'entities_mentioned': [],
            'technical_context': [],
            'specific_commit': None,
            'commit_query': False,
            'time_context': None,
            'specific_files_mentioned': [],  # NEW: Track specific files mentioned
            'file_focused_query': False      # NEW: Is this a file-specific query?
        }
        
        # Detect specific commit hash (7+ hex characters)
        commit_hash_pattern = r'\b([a-f0-9]{7,40})\b'
        commit_matches = re.findall(commit_hash_pattern, query_lower)
        if commit_matches:
            intent['specific_commit'] = commit_matches[0]
            intent['commit_query'] = True
            intent['primary_domain'] = 'commit_analysis'
        
        # Detect commit-related queries
        commit_keywords = ['commit', 'change', 'diff', 'modification', 'when', 'who changed', 'history']
        if any(keyword in query_lower for keyword in commit_keywords):
            intent['commit_query'] = True
            # Don't set primary_domain here - let breaking changes take priority
        
        # Detect time-based contexts
        time_patterns = {
            'recent': ['recent', 'latest', 'last', 'new'],
            'historical': ['old', 'previous', 'earlier', 'before'],
            'specific_time': ['yesterday', 'today', 'week', 'month', 'year']
        }
        
        for time_type, keywords in time_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                intent['time_context'] = time_type
                break
        
        # Detect primary domain (BREAKING CHANGES GETS PRIORITY)
        domain_keywords = {
            'breaking_changes': ['breaking', 'breaking change', 'compatibility', 'backward', 'risk', 'impact', 'migration', 'signature', 'deprecated'],
            'security': ['security', 'vulnerability', 'auth', 'password', 'token', 'encrypt', 'ssl', 'tls', 'xss', 'injection'],
            'performance': ['performance', 'speed', 'optimization', 'bottleneck', 'memory', 'cpu', 'cache', 'latency'],
            'architecture': ['architecture', 'design', 'pattern', 'structure', 'layer', 'component', 'module', 'service'],
            'quality': ['quality', 'complexity', 'debt', 'refactor', 'maintainability', 'readability', 'smell'],
            'testing': ['test', 'testing', 'coverage', 'unit', 'integration', 'mock', 'assert', 'spec'],
            'deployment': ['deploy', 'deployment', 'ci', 'cd', 'pipeline', 'docker', 'kubernetes', 'infrastructure'],
            'data': ['database', 'data', 'sql', 'nosql', 'schema', 'migration', 'query', 'orm'],
            'api': ['api', 'endpoint', 'rest', 'graphql', 'swagger', 'openapi', 'microservice']
        }
        
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                domain_scores[domain] = score
        
        # Set primary domain based on highest score (breaking_changes gets priority if tied)
        if domain_scores:
            intent['primary_domain'] = max(domain_scores, key=domain_scores.get)
            intent['secondary_domains'] = [domain for domain, score in domain_scores.items() 
                                         if score > 0 and domain != intent['primary_domain']]
        elif intent['commit_query'] and intent['primary_domain'] == 'general':
            # Only set commit_analysis if no specific domain was detected
            intent['primary_domain'] = 'commit_analysis'
        
        # Detect complexity level
        if any(word in query_lower for word in ['simple', 'basic', 'overview', 'intro']):
            intent['complexity_level'] = 'low'
        elif any(word in query_lower for word in ['detailed', 'deep', 'comprehensive', 'advanced', 'complex']):
            intent['complexity_level'] = 'high'
        
        # Extract technical entities (classes, functions, files mentioned)
        # Look for code-like patterns
        code_patterns = [
            r'(\w+\.\w+)',  # file.ext or module.function
            r'(\w+\(\))',   # function()
            r'class\s+(\w+)',  # class ClassName
            r'function\s+(\w+)',  # function functionName
            r'/(\w+/\w+)',  # path/to/file
        ]
        
        for pattern in code_patterns:
            matches = re.findall(pattern, query)
            intent['entities_mentioned'].extend(matches)
        
        # Detect specific files mentioned in the query
        file_patterns = [
            r'(\w+\.\w+)',          # filename.ext
            r'(\w+\.c|\.cpp|\.h|\.py|\.js|\.ts|\.java)',  # specific code file extensions
            r'(src/\w+\.\w+)',      # src/filename.ext
            r'(\w+/\w+\.\w+)',      # folder/filename.ext
        ]
        
        for pattern in file_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                # Clean up the match and add to specific files
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[1] if len(match) > 1 else ''
                if match and '.' in match:
                    intent['specific_files_mentioned'].append(match)
        
        # Check if this is a file-focused query
        file_focused_keywords = ['file', 'about', 'in', 'analyze', 'tell me about', 'what is', 'explain', 'describe']
        if (intent['specific_files_mentioned'] and 
            any(keyword in query_lower for keyword in file_focused_keywords)):
            intent['file_focused_query'] = True
            logger.info(f"🎯 File-focused query detected! Files: {intent['specific_files_mentioned']}")
        
        return intent
    
    def _gather_intelligent_context(self, repo_id: str, query: str, intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Intelligently gather relevant files based on query intent and code analysis"""
        repo = self.git_analyzer.repo
        relevant_files = []
        
        try:
            all_files = []
            for item in repo.head.commit.tree.traverse():
                if item.type == 'blob':
                    all_files.append(item.path)
            
            # Score files based on multiple intelligence factors
            scored_files = []
            for file_path in all_files:
                score = self._calculate_intelligent_file_score(file_path, query, intent)
                if score > 0:
                    scored_files.append((file_path, score))
            
            # Sort and take top files
            scored_files.sort(key=lambda x: x[1], reverse=True)
            
            # Load file contents for MANY more files - we'll analyze them all then curate for LLM
            logger.info(f"📊 Found {len(scored_files)} relevant files, analyzing top {min(200, len(scored_files))} for comprehensive context")
            
            for file_path, score in scored_files[:200]:  # Analyze up to 200 files for comprehensive coverage
                try:
                    content = repo.head.commit.tree[file_path].data_stream.read().decode('utf-8', errors='ignore')
                    
                    # Skip very large files (>500KB) for performance
                    if len(content) > 500000:
                        logger.debug(f"Skipping large file {file_path}: {len(content)} bytes")
                        continue
                    
                    # Skip binary-like files
                    if len(content) > 0 and sum(1 for c in content[:1000] if ord(c) < 32 and c not in '\n\r\t') / len(content[:1000]) > 0.3:
                        logger.debug(f"Skipping binary file {file_path}")
                        continue
                    
                    ext = os.path.splitext(file_path)[1].lower()
                    language = self.code_analyzer.supported_languages.get(ext, 'text')
                    
                    relevant_files.append({
                        'path': file_path,
                        'content': content,
                        'language': language,
                        'score': score,
                        'size': len(content),
                        'intelligence_factors': self._get_intelligence_factors(file_path, content, intent)
                    })
                    
                except Exception as e:
                    logger.debug(f"Could not read file {file_path}: {e}")
            
            logger.info(f"✅ Successfully loaded {len(relevant_files)} files for analysis")
            return relevant_files
            
        except Exception as e:
            logger.error(f"Error gathering intelligent context: {e}")
            return []
    
    def _calculate_intelligent_file_score(self, file_path: str, query: str, intent: Dict[str, Any]) -> float:
        """Calculate intelligent relevance score for a file"""
        score = 0.0
        file_path_lower = file_path.lower()
        query_lower = query.lower()
        
        # Base relevance - keyword matching
        for keyword in intent['keywords']:
            if keyword in file_path_lower:
                score += 15
        
        # Domain-specific scoring
        domain = intent['primary_domain']
        
        if domain == 'security':
            if any(pattern in file_path_lower for pattern in ['auth', 'security', 'login', 'token', 'crypto']):
                score += 25
        elif domain == 'performance':
            if any(pattern in file_path_lower for pattern in ['optimization', 'cache', 'performance', 'async']):
                score += 25
        elif domain == 'architecture':
            if any(pattern in file_path_lower for pattern in ['architecture', 'design', 'pattern', 'structure']):
                score += 25
        elif domain == 'testing':
            if any(pattern in file_path_lower for pattern in ['test', 'spec', 'mock']):
                score += 25
        elif domain == 'api':
            if any(pattern in file_path_lower for pattern in ['api', 'endpoint', 'route', 'controller']):
                score += 25
        
        # File importance factors
        file_name = os.path.basename(file_path_lower)
        
        # Critical files get high scores
        if any(name in file_name for name in ['main', 'index', 'app', 'server', 'config']):
            score += 20
        
        # Documentation and README files for overview queries
        if intent['complexity_level'] == 'low' and any(name in file_name for name in ['readme', 'doc', 'guide']):
            score += 15
        
        # Code files for technical queries
        if intent['complexity_level'] == 'high' and any(file_path.endswith(ext) for ext in ['.py', '.js', '.ts', '.java', '.cpp']):
            score += 10
        
        # Entity matching - mentioned classes, functions, files
        for entity in intent['entities_mentioned']:
            if entity.lower() in file_path_lower:
                score += 30
        
        # Language-specific boosts
        ext = os.path.splitext(file_path)[1].lower()
        if ext in self.code_analyzer.supported_languages:
            score += 5
        
        return score
    
    def _get_intelligence_factors(self, file_path: str, content: str, intent: Dict[str, Any]) -> List[str]:
        """Get intelligence factors that made this file relevant"""
        factors = []
        
        file_lower = file_path.lower()
        domain = intent['primary_domain']
        
        # Domain-specific factors
        if domain == 'security' and any(p in file_lower for p in ['auth', 'security', 'login']):
            factors.append('Security-related filename')
        
        if domain == 'performance' and any(p in file_lower for p in ['optimization', 'cache', 'async']):
            factors.append('Performance-related filename')
        
        # Content analysis factors
        if 'TODO' in content or 'FIXME' in content:
            factors.append('Contains technical debt markers')
        
        if len(re.findall(r'class\s+\w+', content)) > 0:
            factors.append('Contains class definitions')
        
        if len(re.findall(r'def\s+\w+|function\s+\w+', content)) > 5:
            factors.append('High function density')
        
        # Architecture factors
        if any(pattern in content.lower() for pattern in ['factory', 'builder', 'observer', 'strategy']):
            factors.append('Design pattern implementation')
        
        return factors
    
    def _analyze_relevant_files(self, relevant_files: List[Dict[str, Any]]) -> List[FileAnalysis]:
        """Perform deep code analysis on relevant files"""
        analyses = []
        
        for file_data in relevant_files:
            file_path = file_data['path']
            
            # Check cache first
            if file_path in self.cached_analyses:
                analyses.append(self.cached_analyses[file_path])
                continue
            
            # Perform analysis
            analysis = self.code_analyzer.analyze_file(file_path, file_data['content'])
            if analysis:
                self.cached_analyses[file_path] = analysis
                analyses.append(analysis)
        
        return analyses
    
    def _gather_intelligent_commits(self, repo_id: str, query: str, intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Gather commits with intelligent filtering based on query intent"""
        repo = self.git_analyzer.repo
        relevant_commits = []
        
        try:
            # Handle specific commit hash queries
            if intent.get('specific_commit'):
                try:
                    specific_commit = repo.commit(intent['specific_commit'])
                    commit_data = {
                        'hash': specific_commit.hexsha,
                        'message': specific_commit.message.strip(),
                        'author': specific_commit.author.name,
                        'author_email': specific_commit.author.email,
                        'date': specific_commit.committed_datetime.isoformat(),
                        'score': 100,  # Maximum score for specific commit
                        'intelligence_factors': ['specific_commit_requested'],
                        'files_changed': []
                    }
                    
                    # Get detailed diff information for specific commit
                    if specific_commit.parents:
                        try:
                            diffs = specific_commit.parents[0].diff(specific_commit, create_patch=True)
                            for diff_item in diffs[:10]:  # Limit to 10 files for performance
                                file_change = {
                                    'file': diff_item.a_path or diff_item.b_path,
                                    'change_type': diff_item.change_type,
                                    'insertions': 0,
                                    'deletions': 0
                                }
                                
                                # Get diff stats if available
                                if diff_item.diff:
                                    diff_text = diff_item.diff.decode('utf-8', errors='ignore')
                                    file_change['insertions'] = diff_text.count('\n+') - diff_text.count('\n+++')
                                    file_change['deletions'] = diff_text.count('\n-') - diff_text.count('\n---')
                                    file_change['diff_snippet'] = str(diff_item)[:1000]  # First 1000 chars
                                
                                commit_data['files_changed'].append(file_change)
                        except Exception as e:
                            logger.debug(f"Error getting diff for specific commit {specific_commit.hexsha}: {e}")
                    
                    return [commit_data]
                    
                except Exception as e:
                    logger.warning(f"Could not find specific commit {intent['specific_commit']}: {e}")
                    # Fall through to regular commit search
            
            # Regular commit gathering with enhanced intelligence
            time_window = 100  # default
            if intent.get('time_context') == 'recent' or any(word in query.lower() for word in ['recent', 'latest', 'last']):
                time_window = 200  # Increased from 30 to 200 to see more recent commits
            elif intent.get('time_context') == 'historical':
                time_window = 1000  # Look further back for historical queries
            
            commits = list(repo.iter_commits(max_count=time_window))
            
            for commit in commits:
                score = self._calculate_commit_intelligence_score(commit, query, intent)
                if score > 0:
                    commit_data = {
                        'hash': commit.hexsha,
                        'message': commit.message.strip(),
                        'author': commit.author.name,
                        'author_email': commit.author.email,
                        'date': commit.committed_datetime.isoformat(),
                        'score': score,
                        'intelligence_factors': self._get_commit_intelligence_factors(commit, intent),
                        'files_changed': []
                    }
                    
                    # Add diff information for high-scoring commits
                    if score > 5:
                        try:
                            if commit.parents:
                                diffs = commit.parents[0].diff(commit, create_patch=True)
                                for diff_item in diffs[:5]:  # Limit to 5 files
                                    if diff_item.a_path:
                                        commit_data['files_changed'].append({
                                            'file': diff_item.a_path,
                                            'change_type': diff_item.change_type,
                                            'diff': str(diff_item)[:3000]  # Limit diff size
                                        })
                        except Exception as e:
                            logger.debug(f"Error getting diff for commit {commit.hexsha}: {e}")
                    
                    relevant_commits.append(commit_data)
            
            # Sort by score and return top commits
            relevant_commits.sort(key=lambda x: x['score'], reverse=True)
            return relevant_commits[:10]
            
        except Exception as e:
            logger.error(f"Error gathering intelligent commits: {e}")
            return []
    
    def _calculate_commit_intelligence_score(self, commit, query: str, intent: Dict[str, Any]) -> float:
        """Calculate intelligent relevance score for a commit"""
        score = 0.0
        message_lower = commit.message.lower()
        query_lower = query.lower()
        
        # Keyword matching
        for keyword in intent['keywords']:
            if keyword in message_lower:
                score += 10
        
        # Domain-specific scoring
        domain = intent['primary_domain']
        domain_keywords = {
            'security': ['security', 'auth', 'vulnerability', 'fix', 'patch'],
            'performance': ['performance', 'optimization', 'speed', 'cache'],
            'architecture': ['refactor', 'architecture', 'design', 'structure'],
            'quality': ['cleanup', 'refactor', 'improve', 'quality'],
            'testing': ['test', 'testing', 'coverage', 'spec']
        }
        
        if domain in domain_keywords:
            for keyword in domain_keywords[domain]:
                if keyword in message_lower:
                    score += 15
        
        # Recent activity boost
        days_ago = (datetime.now() - commit.committed_datetime.replace(tzinfo=None)).days
        if days_ago < 7:
            score += 5
        elif days_ago < 30:
            score += 2
        
        # Bug fixes and important changes
        if any(word in message_lower for word in ['fix', 'bug', 'critical', 'important', 'breaking']):
            score += 10
        
        return score
    
    def _get_commit_intelligence_factors(self, commit, intent: Dict[str, Any]) -> List[str]:
        """Get factors that made this commit relevant"""
        factors = []
        message_lower = commit.message.lower()
        
        if intent['primary_domain'] == 'security' and any(word in message_lower for word in ['security', 'auth', 'fix']):
            factors.append('Security-related commit')
        
        if 'fix' in message_lower:
            factors.append('Bug fix')
        
        if 'refactor' in message_lower:
            factors.append('Refactoring change')
        
        if any(word in message_lower for word in ['breaking', 'major', 'important']):
            factors.append('Major change')
        
        return factors
    
    def _create_specialized_analysis(self, intent: Dict[str, Any], file_analyses: List[FileAnalysis], commits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create specialized analysis based on query domain"""
        analysis = {}
        
        domain = intent['primary_domain']
        
        if domain == 'security':
            analysis['security'] = self._create_security_analysis(file_analyses, commits)
        elif domain == 'architecture':
            analysis['architecture'] = self._create_architecture_analysis(file_analyses)
        elif domain == 'quality':
            analysis['quality'] = self._create_quality_analysis(file_analyses)
        elif domain == 'performance':
            analysis['performance'] = self._create_performance_analysis(file_analyses, commits)
        
        return analysis
    
    def _create_security_analysis(self, file_analyses: List[FileAnalysis], commits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create security-focused analysis"""
        security_issues = []
        affected_files = []
        risk_level = 'low'
        
        for analysis in file_analyses:
            if analysis.security_patterns:
                security_issues.extend(analysis.security_patterns)
                affected_files.append(analysis.file_path)
        
        # Assess risk level
        if len(security_issues) > 5:
            risk_level = 'high'
        elif len(security_issues) > 2:
            risk_level = 'medium'
        
        security_commits = [c for c in commits if any(word in c['message'].lower() for word in ['security', 'auth', 'vulnerability'])]
        
        return {
            'risk_level': risk_level,
            'issues_found': security_issues,
            'affected_files': affected_files,
            'recent_security_commits': security_commits[:5],
            'recommendations': self._generate_security_recommendations(security_issues, affected_files)
        }
    
    def _create_architecture_analysis(self, file_analyses: List[FileAnalysis]) -> Dict[str, Any]:
        """Create architecture-focused analysis"""
        if not file_analyses:
            return {}
        
        arch_analysis = self.code_analyzer.analyze_architecture(file_analyses)
        
        return {
            'patterns_detected': arch_analysis.patterns_detected,
            'layer_structure': arch_analysis.layer_structure,
            'complexity_distribution': [(a.file_path, a.complexity_score) for a in file_analyses],
            'potential_bottlenecks': arch_analysis.potential_bottlenecks,
            'entry_points': arch_analysis.entry_points,
            'recommendations': self._generate_architecture_recommendations(arch_analysis, file_analyses)
        }
    
    def _create_quality_analysis(self, file_analyses: List[FileAnalysis]) -> Dict[str, Any]:
        """Create code quality analysis"""
        high_complexity = [a for a in file_analyses if a.complexity_score > 5]
        tech_debt_files = [a for a in file_analyses if a.technical_debt_indicators]
        
        # Calculate quality score
        if file_analyses:
            avg_complexity = sum(a.complexity_score for a in file_analyses) / len(file_analyses)
            quality_score = max(0, 10 - avg_complexity)
        else:
            quality_score = 5
        
        return {
            'quality_score': quality_score,
            'high_complexity_files': [(a.file_path, a.complexity_score) for a in high_complexity],
            'technical_debt_files': [(a.file_path, a.technical_debt_indicators) for a in tech_debt_files],
            'total_lines_of_code': sum(a.lines_of_code for a in file_analyses),
            'recommendations': self._generate_quality_recommendations(high_complexity, tech_debt_files)
        }
    
    def _create_performance_analysis(self, file_analyses: List[FileAnalysis], commits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create performance-focused analysis"""
        performance_commits = [c for c in commits if any(word in c['message'].lower() for word in ['performance', 'optimization', 'speed'])]
        complex_files = [a for a in file_analyses if a.complexity_score > 3]
        
        return {
            'potential_bottlenecks': [(a.file_path, a.complexity_score) for a in complex_files],
            'recent_performance_commits': performance_commits[:5],
            'files_needing_optimization': [a.file_path for a in complex_files],
            'recommendations': self._generate_performance_recommendations(complex_files, performance_commits)
        }
    
    def _generate_security_recommendations(self, issues: List[str], files: List[str]) -> List[str]:
        """Generate security-specific recommendations"""
        recommendations = []
        
        if any('auth' in issue for issue in issues):
            recommendations.append("Review authentication implementation for security best practices")
        
        if any('password' in issue for issue in issues):
            recommendations.append("Ensure passwords are properly hashed and never stored in plain text")
        
        if any('sql' in issue for issue in issues):
            recommendations.append("Use parameterized queries to prevent SQL injection attacks")
        
        if files:
            recommendations.append(f"Conduct security review of: {', '.join(files[:3])}")
        
        return recommendations
    
    def _generate_architecture_recommendations(self, arch_analysis: ArchitectureAnalysis, file_analyses: List[FileAnalysis]) -> List[str]:
        """Generate architecture-specific recommendations"""
        recommendations = []
        
        if arch_analysis.potential_bottlenecks:
            recommendations.append(f"Consider refactoring high-complexity components: {', '.join(arch_analysis.potential_bottlenecks[:2])}")
        
        if not arch_analysis.patterns_detected:
            recommendations.append("Consider implementing architectural patterns for better organization")
        
        if len(arch_analysis.entry_points) == 0:
            recommendations.append("Consider defining clear entry points for your application")
        elif len(arch_analysis.entry_points) > 3:
            recommendations.append("Consider consolidating multiple entry points for simpler architecture")
        
        return recommendations
    
    def _generate_quality_recommendations(self, high_complexity: List[FileAnalysis], tech_debt: List[FileAnalysis]) -> List[str]:
        """Generate quality-specific recommendations"""
        recommendations = []
        
        if high_complexity:
            recommendations.append(f"Refactor complex files to improve maintainability: {', '.join([a.file_path for a in high_complexity[:3]])}")
        
        if tech_debt:
            recommendations.append(f"Address technical debt markers in: {', '.join([a.file_path for a in tech_debt[:3]])}")
        
        recommendations.append("Consider adding unit tests for better code coverage")
        recommendations.append("Regular code reviews can help maintain quality standards")
        
        return recommendations
    
    def _generate_performance_recommendations(self, complex_files: List[FileAnalysis], perf_commits: List[Dict[str, Any]]) -> List[str]:
        """Generate performance-specific recommendations"""
        recommendations = []
        
        if complex_files:
            recommendations.append(f"Profile and optimize high-complexity files: {', '.join([a.file_path for a in complex_files[:3]])}")
        
        recommendations.append("Consider implementing caching for frequently accessed data")
        recommendations.append("Monitor application performance metrics in production")
        
        if perf_commits:
            recommendations.append("Review recent performance-related changes for impact assessment")
        
        return recommendations
    
    def _analyze_breaking_changes(self, commits: List[Dict[str, Any]], file_analyses: List[FileAnalysis], intent: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze commits for breaking changes using the advanced detector"""
        try:
            breaking_changes_summary = {
                'total_breaking_changes': 0,
                'high_risk_changes': [],
                'medium_risk_changes': [],
                'low_risk_changes': [],
                'severity_breakdown': {'high': 0, 'medium': 0, 'low': 0},
                'intent_patterns': {},
                'affected_signatures': [],
                'migration_complexity': {'simple': 0, 'moderate': 0, 'complex': 0},
                'recommendations': []
            }
            
            # Check if breaking change detector is available
            if not self.breaking_change_detector:
                breaking_changes_summary['recommendations'] = [
                    'Breaking change detection requires git analyzer initialization',
                    'Initialize with git repository for advanced breaking change analysis'
                ]
                return breaking_changes_summary
            
            if not commits:
                return breaking_changes_summary
            
            # Analyze each commit for breaking changes
            for commit_data in commits[:10]:  # Limit analysis to recent commits
                if 'files_changed' not in commit_data:
                    continue
                    
                for file_change in commit_data['files_changed']:
                    if 'diff' not in file_change:
                        continue
                        
                    # Use the advanced breaking change detector
                    diff_text = file_change['diff']
                    file_path = file_change.get('file', 'unknown')
                    
                    breaking_changes = self.breaking_change_detector.analyze_breaking_changes(diff_text, file_path)
                    
                    if breaking_changes['changes']:
                        breaking_changes_summary['total_breaking_changes'] += len(breaking_changes['changes'])
                        
                        # Categorize by severity
                        for change in breaking_changes['changes']:
                            severity = change.get('severity', 'medium')
                            breaking_changes_summary['severity_breakdown'][severity] += 1
                            
                            change_info = {
                                'file': file_path,
                                'commit': commit_data['hash'][:8],
                                'commit_full': commit_data['hash'],
                                'author': commit_data.get('author', 'Unknown Author'),
                                'author_email': commit_data.get('author_email', ''),
                                'date': commit_data.get('date', ''),
                                'type': change.get('type', 'unknown'),
                                'description': change.get('description', ''),
                                'old_signature': change.get('old_signature', ''),
                                'new_signature': change.get('new_signature', ''),
                                'confidence': change.get('confidence', 0.0)
                            }
                            
                            if severity == 'high':
                                breaking_changes_summary['high_risk_changes'].append(change_info)
                            elif severity == 'medium':
                                breaking_changes_summary['medium_risk_changes'].append(change_info)
                            else:
                                breaking_changes_summary['low_risk_changes'].append(change_info)
                            
                            # Track signatures
                            if change.get('old_signature'):
                                breaking_changes_summary['affected_signatures'].append(change['old_signature'])
                        
                        # Track intent patterns
                        for intent_type, count in breaking_changes.get('intent_analysis', {}).items():
                            if intent_type not in breaking_changes_summary['intent_patterns']:
                                breaking_changes_summary['intent_patterns'][intent_type] = 0
                            breaking_changes_summary['intent_patterns'][intent_type] += count
                        
                        # Track migration complexity
                        complexity = breaking_changes.get('migration_complexity', 'moderate')
                        if complexity in breaking_changes_summary['migration_complexity']:
                            breaking_changes_summary['migration_complexity'][complexity] += 1
            
            # Generate recommendations based on analysis
            breaking_changes_summary['recommendations'] = self._generate_breaking_change_recommendations(breaking_changes_summary)
            
            return breaking_changes_summary
            
        except Exception as e:
            logger.error(f"Error analyzing breaking changes: {e}")
            return {
                'total_breaking_changes': 0,
                'error': str(e),
                'recommendations': ['Unable to analyze breaking changes - check system logs']
            }
    
    def _generate_breaking_change_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on breaking change analysis"""
        recommendations = []
        
        total_changes = analysis['total_breaking_changes']
        high_risk = len(analysis.get('high_risk_changes', []))
        medium_risk = len(analysis.get('medium_risk_changes', []))
        
        if total_changes == 0:
            recommendations.append("No breaking changes detected in recent commits")
            recommendations.append("Continue following semantic versioning for API changes")
        else:
            if high_risk > 0:
                recommendations.append(f"⚠️ {high_risk} high-risk breaking changes detected - immediate attention required")
                recommendations.append("Consider creating migration guides for affected APIs")
                recommendations.append("Plan a major version release to communicate breaking changes")
            
            if medium_risk > 3:
                recommendations.append(f"📊 {medium_risk} medium-risk changes may affect users")
                recommendations.append("Review and document all API changes in release notes")
            
            if total_changes > 10:
                recommendations.append("High volume of breaking changes detected")
                recommendations.append("Consider batching changes into fewer, well-planned releases")
            
            # Intent-based recommendations
            intent_patterns = analysis.get('intent_patterns', {})
            if intent_patterns.get('deprecation', 0) > 0:
                recommendations.append("Deprecation changes detected - ensure proper deprecation timeline")
            
            if intent_patterns.get('refactoring', 0) > 0:
                recommendations.append("Refactoring changes may have unintended breaking effects")
            
            # Migration complexity recommendations
            complexity = analysis.get('migration_complexity', {})
            if complexity.get('complex', 0) > 0:
                recommendations.append("Complex migrations detected - provide detailed upgrade guides")
                recommendations.append("Consider offering automated migration tools")
        
        return recommendations[:5]

    def _enhanced_claude_analysis(self, query: str, intent: Dict[str, Any], smart_context: Dict[str, Any], 
                                 specialized_analysis: Dict[str, Any], file_analyses: List[FileAnalysis], 
                                 breaking_change_analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate enhanced Claude analysis with domain expertise"""
        
        # Create domain-specific prompt
        domain_prompts = {
            'security': self._create_security_expert_prompt,
            'architecture': self._create_architecture_expert_prompt,
            'performance': self._create_performance_expert_prompt,
            'quality': self._create_quality_expert_prompt,
            'breaking_changes': self._create_breaking_changes_expert_prompt,
            'general': self._create_general_expert_prompt
        }
        
        domain = intent['primary_domain']
        prompt_creator = domain_prompts.get(domain, domain_prompts['general'])
        
        enhanced_prompt = prompt_creator(query, intent, smart_context, specialized_analysis, file_analyses, breaking_change_analysis)
        
        try:
            import anthropic
            response = self.claude_analyzer.client.messages.create(
                model=self.claude_analyzer.model,
                max_tokens=self.claude_analyzer.max_tokens,
                messages=[{"role": "user", "content": enhanced_prompt}]
            )
            
            claude_text = response.content[0].text
            
            # Parse response for structured information
            suggestions = self._extract_enhanced_suggestions(claude_text, domain, smart_context)
            sources = self._create_enhanced_sources(file_analyses, specialized_analysis)
            
            return {
                'response': claude_text,
                'confidence': min(0.9, 0.7 + len(file_analyses) * 0.05),
                'sources': sources,
                'context_used': [f"Enhanced {domain} analysis", f"Analyzed {len(file_analyses)} files"],
                'suggestions': suggestions
            }
            
        except Exception as e:
            logger.error(f"Enhanced Claude analysis failed: {e}")
            return self._create_fallback_response(query, smart_context, specialized_analysis)
    
    def _create_security_expert_prompt(self, query: str, intent: Dict[str, Any], smart_context: Dict[str, Any], 
                                     specialized_analysis: Dict[str, Any], file_analyses: List[FileAnalysis],
                                     breaking_change_analysis: Optional[Dict[str, Any]] = None) -> str:
        """Create security expert prompt"""
        security_data = specialized_analysis.get('security', {})
        
        prompt = f"""# 🛡️ Security Expert Analysis

You are a senior security engineer analyzing a codebase. Provide expert-level security insights.

## Query
{query}

## Security Analysis Results
- **Risk Level**: {security_data.get('risk_level', 'unknown')}
- **Issues Found**: {len(security_data.get('issues_found', []))} potential issues
- **Affected Files**: {len(security_data.get('affected_files', []))} files

## Security Issues Detected
{chr(10).join(f"- {issue}" for issue in security_data.get('issues_found', [])[:10])}

## Files with Security Concerns ({len([a for a in file_analyses if a.security_patterns])} out of {len(file_analyses)} total)
"""

        security_files = [a for a in file_analyses if a.security_patterns]
        other_files = [a for a in file_analyses if not a.security_patterns]
        
        # Show files with security issues first
        for analysis in security_files[:5]:
            prompt += f"""
### 🔍 {analysis.file_path} ⚠️ SECURITY CONCERNS
- **Security Patterns Found**: {', '.join(analysis.security_patterns[:5])}
- **Complexity Score**: {analysis.complexity_score:.1f}/10
- **Lines of Code**: {getattr(analysis, 'lines_of_code', 'unknown')}
- **Tech Debt Indicators**: {len(analysis.technical_debt_indicators)} found

**Key Security-Related Entities**:
"""
            for entity in analysis.entities[:3]:
                entity_type = getattr(entity, 'type', 'unknown')
                entity_name = getattr(entity, 'name', 'unnamed')
                # Highlight security-related entities
                if any(sec_keyword in entity_name.lower() for sec_keyword in ['auth', 'login', 'password', 'token', 'security', 'encrypt']):
                    prompt += f"  🔐 {entity_type}: `{entity_name}` (SECURITY RELATED)\n"
                else:
                    prompt += f"  - {entity_type}: `{entity_name}`\n"
            
            # ADD ACTUAL FILE CONTENT FOR CLAUDE TO ANALYZE
            if hasattr(analysis, 'content') and analysis.content:
                # Truncate very long files but include substantial content
                content = analysis.content
                if len(content) > 3000:
                    content = content[:3000] + f"\n... [File truncated - showing first 3000 chars of {len(analysis.content)} total chars] ..."
                
                prompt += f"""
**FULL FILE CONTENT**:
```{analysis.language}
{content}
```
"""
            elif hasattr(analysis, 'key_sections') and analysis.key_sections:
                prompt += f"""
**KEY CODE SECTIONS**:
```{analysis.language}
{analysis.key_sections[:2000]}
```
"""
        
        # Show other analyzed files as context
        if other_files:
            prompt += f"""
### 📁 Additional Files Analyzed ({len(other_files)} files)
"""
            for analysis in other_files[:8]:  # Show more files with content
                prompt += f"""
#### 📄 {analysis.file_path}
- **Language**: {analysis.language}, **Entities**: {len(analysis.entities)}, **Complexity**: {analysis.complexity_score:.1f}
"""
                # Include actual file content for better analysis
                if hasattr(analysis, 'content') and analysis.content:
                    content = analysis.content
                    if len(content) > 2000:
                        content = content[:2000] + f"\n... [File truncated - showing first 2000 chars of {len(analysis.content)} total] ..."
                    
                    prompt += f"""
**FILE CONTENT FROM: {analysis.file_path}**:
```{analysis.language}
// ===== FILE: {analysis.file_path} =====
{content}
// ===== END OF FILE: {analysis.file_path} =====
```
"""

        prompt += f"""

## 🎯 Expert Analysis Required

As a security expert, provide:

1. **🔍 Security Assessment**: Detailed evaluation of the security posture
2. **⚠️ Critical Vulnerabilities**: Prioritized list of issues requiring immediate attention  
3. **🛠️ Specific Remediation**: Concrete code fixes and security improvements
4. **📋 Security Checklist**: Actionable items for security hardening
5. **🚨 Risk Priority**: High/Medium/Low priority classification with reasoning

Focus on practical, implementable security improvements. Reference specific files and line numbers when possible.
"""
        
        return prompt
    def _create_architecture_expert_prompt(self, query: str, intent: Dict[str, Any], smart_context: Dict[str, Any], 
                                          specialized_analysis: Dict[str, Any], file_analyses: List[FileAnalysis],
                                          breaking_change_analysis: Optional[Dict[str, Any]] = None) -> str:
        """Create architecture expert prompt"""
        arch_data = specialized_analysis.get('architecture', {})
        
        prompt = f"""# 🏗️ Software Architecture Expert Analysis

You are a senior software architect analyzing a codebase. Provide expert architectural insights.

## Query
{query}

## Architecture Analysis
- **Patterns Detected**: {', '.join(arch_data.get('patterns_detected', []))}
- **Entry Points**: {len(arch_data.get('entry_points', []))} identified
- **Potential Bottlenecks**: {len(arch_data.get('potential_bottlenecks', []))} found

## Code Structure
"""

        # Add complexity analysis
        complexity_dist = arch_data.get('complexity_distribution', [])
        if complexity_dist:
            prompt += "### Complexity Distribution\n"
            for file_path, complexity in complexity_dist[:5]:
                prompt += f"- **{file_path}**: {complexity:.1f}\n"

        # Add layer structure if available
        layer_structure = arch_data.get('layer_structure', {})
        if layer_structure:
            prompt += "\n### Layer Structure\n"
            for pattern, files in layer_structure.items():
                prompt += f"- **{pattern.title()}**: {len(files)} files\n"

        prompt += f"""

## 🎯 Architecture Analysis Required

As a software architect, provide:

1. **🏗️ Architecture Assessment**: Overall design quality and structure analysis
2. **📐 Design Patterns**: Evaluation of current patterns and recommendations for improvement
3. **🔄 Dependency Analysis**: Assessment of coupling and cohesion
4. **⚡ Scalability Concerns**: Potential bottlenecks and scaling limitations
5. **🎯 Improvement Roadmap**: Prioritized architectural improvements

Focus on maintainable, scalable architecture recommendations with specific implementation guidance.
"""
        
        return prompt
    def _create_performance_expert_prompt(self, query: str, intent: Dict[str, Any], smart_context: Dict[str, Any], 
                                         specialized_analysis: Dict[str, Any], file_analyses: List[FileAnalysis],
                                         breaking_change_analysis: Optional[Dict[str, Any]] = None) -> str:
        """Create performance expert prompt"""
        perf_data = specialized_analysis.get('performance', {})
        
        prompt = f"""# ⚡ Performance Engineering Expert Analysis

You are a senior performance engineer analyzing a codebase for optimization opportunities.

## Query
{query}

## Performance Analysis
- **Potential Bottlenecks**: {len(perf_data.get('potential_bottlenecks', []))} identified
- **Files Needing Optimization**: {len(perf_data.get('files_needing_optimization', []))}
- **Recent Performance Commits**: {len(perf_data.get('recent_performance_commits', []))}

## Performance Hotspots
"""

        for file_path, complexity in perf_data.get('potential_bottlenecks', [])[:5]:
            prompt += f"- **{file_path}**: Complexity {complexity:.1f}\n"

        prompt += f"""

## 🎯 Performance Analysis Required

As a performance expert, provide:

1. **⚡ Performance Assessment**: Current performance characteristics and bottlenecks
2. **🔍 Optimization Opportunities**: Specific areas for performance improvement
3. **📊 Metrics Recommendations**: Key performance indicators to monitor
4. **🛠️ Implementation Strategy**: Prioritized optimization plan with estimated impact
5. **⚠️ Risk Assessment**: Performance risks and mitigation strategies

Focus on measurable, high-impact performance improvements with implementation details.
"""
        
        return prompt
    
    def _create_quality_expert_prompt(self, query: str, intent: Dict[str, Any], smart_context: Dict[str, Any], 
                                     specialized_analysis: Dict[str, Any], file_analyses: List[FileAnalysis],
                                     breaking_change_analysis: Optional[Dict[str, Any]] = None) -> str:
        """Create code quality expert prompt"""
        quality_data = specialized_analysis.get('quality', {})
        
        prompt = f"""# 📊 Code Quality Expert Analysis

You are a senior code quality engineer analyzing a codebase for maintainability and best practices.

## Query
{query}

## Quality Metrics
- **Quality Score**: {quality_data.get('quality_score', 0):.1f}/10
- **High Complexity Files**: {len(quality_data.get('high_complexity_files', []))}
- **Technical Debt Files**: {len(quality_data.get('technical_debt_files', []))}
- **Total Lines of Code**: {quality_data.get('total_lines_of_code', 0):,}

## Quality Issues
"""

        # Add complexity issues
        for file_path, complexity in quality_data.get('high_complexity_files', [])[:5]:
            prompt += f"- **{file_path}**: Complexity {complexity:.1f} (needs refactoring)\n"

        # Add technical debt
        for file_path, debt_indicators in quality_data.get('technical_debt_files', [])[:5]:
            prompt += f"- **{file_path}**: {', '.join(debt_indicators[:3])}\n"

        prompt += f"""

## 🎯 Quality Analysis Required

As a code quality expert, provide:

1. **📊 Quality Assessment**: Overall code health and maintainability evaluation
2. **🔧 Refactoring Priorities**: Specific files and methods needing improvement
3. **📝 Best Practices**: Code quality standards and guidelines
4. **🧪 Testing Strategy**: Recommendations for improving test coverage and quality
5. **📈 Quality Metrics**: KPIs to track and improve code quality over time

Focus on practical, incremental improvements that enhance maintainability and reduce technical debt.
"""
        
        return prompt
    
    def _create_breaking_changes_expert_prompt(self, query: str, intent: Dict[str, Any], smart_context: Dict[str, Any], 
                                             specialized_analysis: Dict[str, Any], file_analyses: List[FileAnalysis],
                                             breaking_change_analysis: Optional[Dict[str, Any]] = None) -> str:
        """Create breaking changes expert prompt"""
        bc_data = breaking_change_analysis or {}
        
        prompt = f"""# ⚠️ Breaking Changes Expert Analysis

You are a senior software architect specializing in API compatibility and breaking change analysis.

## Query
{query}

## 🎯 **IMPORTANT**: When answering about breaking changes, always include:
- **📁 File Name**: The exact file where the change occurred
- **🔧 Function/Method Name**: The specific function or method affected  
- **💾 Commit Hash**: The commit that introduced the change
- **👤 Author**: Who made the change
- **📅 Date**: When the change was made

## ⚠️ CRITICAL: SPECIFIC BREAKING CHANGES DETECTED

⚠️ **ATTENTION: {bc_data.get('total_breaking_changes', 0)} BREAKING CHANGES FOUND** ⚠️

The automated analysis has detected specific breaking changes that need your expert evaluation:
"""

        # Show ALL detected breaking changes prominently at the top
        all_changes = []
        all_changes.extend(bc_data.get('high_risk_changes', []))
        all_changes.extend(bc_data.get('medium_risk_changes', []))
        all_changes.extend(bc_data.get('low_risk_changes', []))
        
        if all_changes:
            prompt += f"""
### 🚨 SPECIFIC BREAKING CHANGES DETECTED ({len(all_changes)} Total):

**📋 BREAKING CHANGES SUMMARY TABLE:**
"""
            # Create a detailed table of breaking changes
            for i, change in enumerate(all_changes, 1):
                risk_level = "🔴 HIGH RISK" if change in bc_data.get('high_risk_changes', []) else \
                           "🟡 MEDIUM RISK" if change in bc_data.get('medium_risk_changes', []) else "🟢 LOW RISK"
                
                # Extract function name from description if available
                function_name = "Unknown Function"
                description = change.get('description', '')
                if "function '" in description and "' was" in description:
                    function_name = description.split("function '")[1].split("'")[0]
                elif "method '" in description and "' was" in description:
                    function_name = description.split("method '")[1].split("'")[0]
                
                prompt += f"""
**CHANGE #{i} [{risk_level}]**
┌─ 📁 **File**: `{change.get('file', 'unknown')}`
├─ 🔧 **Function/Method**: `{function_name}`
├─ 💾 **Commit**: `{change.get('commit', 'unknown')}` (Full: `{change.get('commit_full', 'unknown')}`)
├─ 👤 **Author**: {change.get('author', 'Unknown Author')} <{change.get('author_email', 'no-email')}>
├─ 📅 **Date**: {change.get('date', 'Unknown date')}
├─ 📊 **Change Type**: {change.get('type', 'unknown_change').replace('_', ' ').title()}
├─ 📝 **Description**: {change.get('description', 'No description available')}
├─ 📈 **Confidence**: {change.get('confidence', 0.0):.0%}
├─ 🔹 **Old Signature**: `{change.get('old_signature', 'Not captured')}`
└─ 🔸 **New Signature**: `{change.get('new_signature', 'Not captured')}`

"""
        else:
            prompt += """
✅ **No breaking changes detected** - This is good news for API stability.
"""

        prompt += f"""

## Analysis Scope & Context
- **Total Files Analyzed**: {len(file_analyses)} files
- **Repository Analysis Depth**: Comprehensive codebase scan
- **Analysis Confidence**: High (based on {len(file_analyses)} file analyses)

## Breaking Changes Analysis Results
- **Total Breaking Changes**: {bc_data.get('total_breaking_changes', 0)}
- **High Risk Changes**: {len(bc_data.get('high_risk_changes', []))}
- **Medium Risk Changes**: {len(bc_data.get('medium_risk_changes', []))}
- **Low Risk Changes**: {len(bc_data.get('low_risk_changes', []))}

## Severity Breakdown
"""
        
        severity_breakdown = bc_data.get('severity_breakdown', {})
        for severity, count in severity_breakdown.items():
            prompt += f"- **{severity.capitalize()}**: {count} changes\n"
        
        prompt += f"""

## High Risk Breaking Changes (Immediate Attention Required)
"""
        high_risk_changes = bc_data.get('high_risk_changes', [])
        if high_risk_changes:
            for i, change in enumerate(high_risk_changes[:5], 1):
                prompt += f"""
### {i}. {change.get('type', 'Unknown Change')} in {change.get('file', 'unknown')}
- **Commit**: {change.get('commit', 'unknown')}
- **Description**: {change.get('description', 'No description')}
- **Old Signature**: `{change.get('old_signature', 'Not available')}`
- **New Signature**: `{change.get('new_signature', 'Not available')}`
- **Confidence**: {change.get('confidence', 0.0):.0%}
"""
        else:
            prompt += "✅ No high-risk breaking changes detected.\n"

        # Add Medium and Low Risk Changes too
        medium_risk_changes = bc_data.get('medium_risk_changes', [])
        if medium_risk_changes:
            prompt += f"""

## Medium Risk Breaking Changes ({len(medium_risk_changes)} found)
"""
            for i, change in enumerate(medium_risk_changes[:5], 1):
                prompt += f"""
### {i}. {change.get('type', 'Unknown Change')} in {change.get('file', 'unknown')}
- **Commit**: {change.get('commit', 'unknown')}
- **Description**: {change.get('description', 'No description')}
- **Impact**: {change.get('impact', 'Medium impact on users')}
- **Confidence**: {change.get('confidence', 0.0):.0%}
"""

        low_risk_changes = bc_data.get('low_risk_changes', [])
        if low_risk_changes:
            prompt += f"""

## Low Risk Breaking Changes ({len(low_risk_changes)} found)
"""
            for i, change in enumerate(low_risk_changes[:3], 1):
                prompt += f"""
### {i}. {change.get('type', 'Unknown Change')} in {change.get('file', 'unknown')}
- **Commit**: {change.get('commit', 'unknown')}
- **Description**: {change.get('description', 'No description')}
"""
        
        prompt += f"""

## All Detected Breaking Changes Summary
"""
        all_changes = []
        all_changes.extend(bc_data.get('high_risk_changes', []))
        all_changes.extend(bc_data.get('medium_risk_changes', []))
        all_changes.extend(bc_data.get('low_risk_changes', []))
        
        if all_changes:
            prompt += f"""
**Total Breaking Changes Found**: {len(all_changes)}

### Complete Breaking Changes List:
"""
            for i, change in enumerate(all_changes[:15], 1):  # Show up to 15 total changes
                risk_level = "HIGH" if change in bc_data.get('high_risk_changes', []) else \
                           "MEDIUM" if change in bc_data.get('medium_risk_changes', []) else "LOW"
                prompt += f"""
{i}. **[{risk_level} RISK]** {change.get('type', 'Unknown Change')} in `{change.get('file', 'unknown')}`
   - Commit: {change.get('commit', 'unknown')[:8]}...
   - Change: {change.get('description', 'No description')[:100]}
   - Confidence: {change.get('confidence', 0.0):.0%}
"""
        else:
            prompt += "✅ **No breaking changes detected** in the analyzed commits.\n"

        prompt += f"""

## Intent Patterns Analysis
"""
        intent_patterns = bc_data.get('intent_patterns', {})
        if intent_patterns:
            for intent_type, count in intent_patterns.items():
                prompt += f"- **{intent_type.capitalize()}**: {count} occurrences\n"
        else:
            prompt += "No specific intent patterns detected.\n"
        
        prompt += f"""

## Migration Complexity Assessment
"""
        migration_complexity = bc_data.get('migration_complexity', {})
        for complexity, count in migration_complexity.items():
            prompt += f"- **{complexity.capitalize()}**: {count} changes\n"
        
        prompt += f"""

## Affected Signatures
"""
        affected_signatures = bc_data.get('affected_signatures', [])
        if affected_signatures:
            for i, signature in enumerate(affected_signatures[:10], 1):
                prompt += f"{i}. `{signature}`\n"
        else:
            prompt += "No specific signatures identified.\n"
        
        prompt += f"""

## Files Analyzed ({len(file_analyses)} total files)
"""
        for analysis in file_analyses[:10]:  # Show detailed analysis for top files WITH CONTENT
            prompt += f"""
### 📁 {analysis.file_path}
- **Language**: {analysis.language}
- **Complexity Score**: {analysis.complexity_score:.1f}/10
- **Lines of Code**: {getattr(analysis, 'lines_of_code', 'unknown')}
- **Functions/Classes**: {len(analysis.entities)} entities
- **Architecture Patterns**: {', '.join(analysis.architecture_patterns) if analysis.architecture_patterns else 'Standard structure'}
- **Security Concerns**: {len(analysis.security_patterns)} issues found
- **Technical Debt**: {len(analysis.technical_debt_indicators)} indicators

**Key Entities**:
"""
            # Show key entities (functions, classes) found in the file
            for entity in analysis.entities[:5]:  # Top 5 entities
                entity_type = getattr(entity, 'type', 'unknown')
                entity_name = getattr(entity, 'name', 'unnamed')
                prompt += f"  - {entity_type}: `{entity_name}`\n"
            
            # ADD FULL FILE CONTENT FOR CLAUDE'S BREAKING CHANGE ANALYSIS WITH CLEAR HEADERS
            if hasattr(analysis, 'content') and analysis.content:
                content = analysis.content
                if len(content) > 4000:  # Allow more content for breaking change analysis
                    content = content[:4000] + f"\n... [File truncated - showing first 4000 chars of {len(analysis.content)} total] ..."
                
                prompt += f"""
**COMPLETE FILE CONTENT FROM: {analysis.file_path}**:
```{analysis.language}
// ===== FILE: {analysis.file_path} =====
{content}
// ===== END OF FILE: {analysis.file_path} =====
```
"""
            elif hasattr(analysis, 'content_summary') and analysis.content_summary:
                prompt += f"""
**Content Summary**: {analysis.content_summary[:200]}...
"""
            elif len(analysis.entities) > 0:
                prompt += f"**Primary Focus**: Contains {analysis.entities[0].get('type', 'code')} definitions and related functionality\n"
        
        # Show summary of remaining files
        if len(file_analyses) > 10:
            remaining_count = len(file_analyses) - 10
            prompt += f"""
### 📋 Additional Files Analyzed ({remaining_count} more)
"""
            for analysis in file_analyses[10:]:
                prompt += f"""
#### 📄 {analysis.file_path}
- **Language**: {analysis.language}, **Complexity**: {analysis.complexity_score:.1f}
"""
                # Include shorter content excerpts for remaining files
                if hasattr(analysis, 'content') and analysis.content:
                    content = analysis.content
                    if len(content) > 1000:
                        content = content[:1000] + f"\n... [Truncated] ..."
                    
                    prompt += f"""
**FILE CONTENT FROM: {analysis.file_path}**:
```{analysis.language}
// ===== FILE: {analysis.file_path} =====
{content}
// ===== END OF FILE: {analysis.file_path} =====
```
"""

        prompt += f"""

## 🎯 Breaking Changes Expert Analysis Required

As a breaking changes expert, provide:

1. **🔍 Risk Assessment**: Evaluate the overall impact and risk level of detected changes
2. **📊 Impact Analysis**: Analyze which changes will affect users most significantly  
3. **🛠️ Migration Strategy**: Recommend migration approaches for affected users
4. **📋 Version Planning**: Suggest versioning strategy (major/minor/patch)
5. **📖 Documentation Needs**: Identify what documentation and guides are needed
6. **⏰ Rollout Timeline**: Recommend phased rollout strategy if needed
7. **🔄 Backward Compatibility**: Suggest strategies to maintain compatibility where possible

Focus on minimizing user disruption while enabling necessary evolution of the codebase.
Provide specific, actionable recommendations for managing these breaking changes.
"""
        
        return prompt
    
    def _create_general_expert_prompt(self, query: str, intent: Dict[str, Any], smart_context: Dict[str, Any], 
                                    specialized_analysis: Dict[str, Any], file_analyses: List[FileAnalysis],
                                    breaking_change_analysis: Optional[Dict[str, Any]] = None) -> str:
        """Create general expert prompt"""
        
        # Check if this is a file-focused query
        file_focused = intent.get('file_focused_query', False)
        specific_files = intent.get('specific_files_mentioned', [])
        
        prompt = f"""# 💻 Senior Software Engineer Analysis

You are a senior software engineer providing comprehensive code analysis and recommendations.

## Query
{query}

## 🎯 **IMPORTANT INSTRUCTIONS**: 
- **Direct Questions**: If the query asks about a specific file (e.g., "tell me about executor.c"), focus your analysis PRIMARILY on that file
- **Specific Focus**: Answer the specific question asked, using other files only for supporting context
- **File Content**: Use the actual file content provided below to give detailed, accurate answers

## Analysis Context
- **Primary Domain**: {intent['primary_domain']}
- **Files Analyzed**: {len(file_analyses)}
- **Analysis Confidence**: {smart_context.get('confidence', 0.5):.0%}"""

        # Add file-focused context if applicable
        if file_focused and specific_files:
            prompt += f"""
- **🎯 FILE-FOCUSED QUERY**: Yes - Focus analysis on the requested files
- **📁 Specific Files Requested**: {', '.join(specific_files)}
- **📋 Analysis Type**: Detailed analysis of specifically requested files with supporting context"""
        
        prompt += """

## Code Insights
"""

        # Add relevant insights from smart context
        code_insights = smart_context.get('code_insights', {})
        if code_insights:
            prompt += f"- **Code Quality**: {json.dumps(code_insights, indent=2)}\n"

        # Handle file-focused queries differently
        if file_focused and specific_files:
            # For file-focused queries, prioritize mentioned files
            mentioned_analyses = []
            other_analyses = []
            
            for analysis in file_analyses:
                filename = os.path.basename(analysis.file_path)
                if any(sf.lower() in filename.lower() or filename.lower() in sf.lower() for sf in specific_files):
                    mentioned_analyses.append(analysis)
                else:
                    other_analyses.append(analysis)
            
            prompt += f"""

## 🎯 PRIMARY FOCUS: Requested Files Analysis
**Note**: The user specifically asked about these files. Provide detailed analysis focusing primarily on these files:
"""
            
            # Show mentioned files first with full detail
            for analysis in mentioned_analyses:
                prompt += f"""
### 🔍 PRIMARY: {analysis.file_path}
- **Language**: {analysis.language}
- **Complexity Score**: {analysis.complexity_score:.1f}/10
- **Lines of Code**: {getattr(analysis, 'lines_of_code', 'unknown')}
- **Architecture Patterns**: {', '.join(analysis.architecture_patterns) if analysis.architecture_patterns else 'Standard structure'}
- **Security Patterns**: {len(analysis.security_patterns)} found
- **Technical Debt Indicators**: {len(analysis.technical_debt_indicators)} found

**Key Entities Found**:
"""
                for entity in analysis.entities[:5]:  # Show more entities for requested files
                    entity_type = getattr(entity, 'type', 'unknown')
                    entity_name = getattr(entity, 'name', 'unnamed')
                    entity_line = getattr(entity, 'line_number', 'unknown')
                    prompt += f"  - {entity_type}: `{entity_name}` (line {entity_line})\n"
                
                # ADD FULL FILE CONTENT FOR REQUESTED FILES
                if hasattr(analysis, 'content') and analysis.content:
                    content = analysis.content
                    if len(content) > 4000:  # More content for requested files
                        content = content[:4000] + f"\n... [File truncated - showing first 4000 chars of {len(analysis.content)} total] ..."
                    
                    prompt += f"""
**COMPLETE FILE CONTENT FROM: {analysis.file_path}**:
```{analysis.language}
// ===== REQUESTED FILE: {analysis.file_path} =====
{content}
// ===== END OF REQUESTED FILE: {analysis.file_path} =====
```
"""
            
            # Show other files for context if any
            if other_analyses:
                prompt += f"""

## 📁 Additional Context Files ({len(other_analyses)} files for reference)
"""
                for analysis in other_analyses[:4]:  # Fewer context files
                    prompt += f"""
### 📄 {analysis.file_path}
- **Language**: {analysis.language}, **Entities**: {len(analysis.entities)} entities, **Complexity**: {analysis.complexity_score:.1f}/10
"""
                    # Shorter content for context files
                    if hasattr(analysis, 'content') and analysis.content:
                        content = analysis.content
                        if len(content) > 1000:
                            content = content[:1000] + f"\n... [File truncated for context] ..."
                        
                        prompt += f"""
**CONTEXT FILE CONTENT**:
```{analysis.language}
// ===== CONTEXT FILE: {analysis.file_path} =====
{content}
// ===== END CONTEXT FILE =====
```
"""
        else:
            # Standard analysis for non-file-focused queries
            prompt += f"""

## 📁 Detailed File Analysis ({len(file_analyses)} files analyzed)
"""
            for analysis in file_analyses[:8]:  # Show top 8 files in detail with FULL CONTENT
                prompt += f"""
### {analysis.file_path}
- **Language**: {analysis.language}
- **Complexity Score**: {analysis.complexity_score:.1f}/10
- **Lines of Code**: {getattr(analysis, 'lines_of_code', 'unknown')}
- **Architecture Patterns**: {', '.join(analysis.architecture_patterns) if analysis.architecture_patterns else 'Standard structure'}
- **Security Patterns**: {len(analysis.security_patterns)} found
- **Technical Debt Indicators**: {len(analysis.technical_debt_indicators)} found

**Key Entities Found**:
"""
            for entity in analysis.entities[:3]:  # Show top 3 entities
                entity_type = getattr(entity, 'type', 'unknown')
                entity_name = getattr(entity, 'name', 'unnamed')
                entity_line = getattr(entity, 'line_number', 'unknown')
                prompt += f"  - {entity_type}: `{entity_name}` (line {entity_line})\n"
            
            # ADD FULL FILE CONTENT FOR CLAUDE TO ANALYZE WITH CLEAR HEADERS
            if hasattr(analysis, 'content') and analysis.content:
                content = analysis.content
                if len(content) > 3000:
                    content = content[:3000] + f"\n... [File truncated - showing first 3000 chars of {len(analysis.content)} total] ..."
                
                prompt += f"""
**COMPLETE FILE CONTENT FROM: {analysis.file_path}**:
```{analysis.language}
// ===== FILE: {analysis.file_path} =====
{content}
// ===== END OF FILE: {analysis.file_path} =====
```
"""
        
        # Summary of additional files
        if len(file_analyses) > 8:
            remaining_files = file_analyses[8:]
            prompt += f"""
### Additional Files ({len(remaining_files)} more)
"""
            for analysis in remaining_files:
                prompt += f"""
#### 📄 {analysis.file_path}
- **Language**: {analysis.language}, **Entities**: {len(analysis.entities)} entities
"""
                # Include content for remaining files too (shorter excerpts)
                if hasattr(analysis, 'content') and analysis.content:
                    content = analysis.content
                    if len(content) > 1500:
                        content = content[:1500] + f"\n... [File truncated - showing first 1500 chars] ..."
                    
                    prompt += f"""
**FILE CONTENT FROM: {analysis.file_path}**:
```{analysis.language}
// ===== FILE: {analysis.file_path} =====
{content}
// ===== END OF FILE: {analysis.file_path} =====
```
"""

        prompt += f"""

## 🎯 Expert Analysis Required

As a senior engineer, provide:

1. **🔍 Technical Assessment**: Comprehensive evaluation of the codebase
2. **🎯 Key Insights**: Most important findings and observations
3. **⚠️ Issues & Risks**: Problems that need attention with priority levels
4. **💡 Recommendations**: Specific, actionable improvements
5. **📈 Next Steps**: Prioritized roadmap for improvements

Focus on practical, business-value driven recommendations that improve the codebase.
"""
        
        return prompt
    
    def _extract_enhanced_suggestions(self, claude_text: str, domain: str, smart_context: Dict[str, Any]) -> List[str]:
        """Extract enhanced, domain-specific suggestions from Claude response"""
        suggestions = []
        
        # Base suggestion extraction
        lines = claude_text.split('\n')
        in_recommendations = False
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['recommendation', 'action', 'next step', 'should']):
                in_recommendations = True
                continue
            
            if in_recommendations and line:
                if any(line.startswith(prefix) for prefix in ['1.', '2.', '3.', '4.', '5.', '-', '*', '•']):
                    clean_suggestion = line
                    for prefix in ['1. ', '2. ', '3. ', '4. ', '5. ', '- ', '* ', '• ']:
                        if clean_suggestion.startswith(prefix):
                            clean_suggestion = clean_suggestion[len(prefix):].strip()
                            break
                    
                    if len(clean_suggestion) > 15:
                        suggestions.append(clean_suggestion)
            
            if line.startswith('#') and in_recommendations:
                break
        
        # Add domain-specific smart suggestions
        recommendations = smart_context.get('recommendations', [])
        for rec in recommendations[:3]:  # Add top 3 smart recommendations
            if rec not in suggestions:
                suggestions.append(rec)
        
        return suggestions[:6]  # Return top 6 suggestions
    
    def _create_enhanced_sources(self, file_analyses: List[FileAnalysis], specialized_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create enhanced source information"""
        sources = []
        
        for analysis in file_analyses:
            source = {
                'type': 'file',
                'path': analysis.file_path,
                'language': analysis.language,
                'complexity': analysis.complexity_score,
                'entities': len(analysis.entities),
                'size_kb': round(analysis.lines_of_code * 50 / 1024, 1),  # Rough estimate
                'intelligence_factors': []
            }
            
            # Add intelligence factors
            if analysis.security_patterns:
                source['intelligence_factors'].append('Security patterns detected')
            if analysis.architecture_patterns:
                source['intelligence_factors'].append(f"Architecture: {', '.join(analysis.architecture_patterns)}")
            if analysis.complexity_score > 5:
                source['intelligence_factors'].append('High complexity')
            if analysis.technical_debt_indicators:
                source['intelligence_factors'].append('Technical debt markers')
            
            sources.append(source)
        
        return sources
    
    def _create_fallback_response(self, query: str, smart_context: Dict[str, Any], specialized_analysis: Dict[str, Any]) -> QueryResult:
        """Create fallback response when Claude is not available"""
        
        # Create a comprehensive fallback response using smart context
        response_parts = [
            f"# 📊 Smart Analysis Results",
            f"",
            f"**Query**: {query}",
            f"**Analysis**: Enhanced context gathering completed",
            f""
        ]
        
        # Add insights from smart context
        if smart_context.get('code_insights'):
            response_parts.extend([
                "## 🔍 Code Insights",
                ""
            ])
            for insight in smart_context['code_insights']:
                response_parts.append(f"- {insight}")
            response_parts.append("")
        
        # Add specialized analysis results
        for domain, analysis in specialized_analysis.items():
            response_parts.extend([
                f"## {domain.title()} Analysis",
                ""
            ])
            if isinstance(analysis, dict):
                for key, value in analysis.items():
                    if isinstance(value, list) and value:
                        response_parts.append(f"- **{key}**: {', '.join(map(str, value[:3]))}")
                    elif not isinstance(value, (list, dict)):
                        response_parts.append(f"- **{key}**: {value}")
            response_parts.append("")
        
        # Add recommendations
        recommendations = smart_context.get('recommendations', [])
        if recommendations:
            response_parts.extend([
                "## 💡 Recommendations",
                ""
            ])
            for rec in recommendations[:5]:
                response_parts.append(f"- {rec}")
        
        response_parts.append("\n*Configure Claude API for AI-enhanced analysis.*")
        
        return QueryResult(
            query=query,
            response="\n".join(response_parts),
            confidence=0.6,
            sources=[],
            context_used=["Smart context analysis", "Code structure analysis"],
            suggestions=recommendations[:4],
            code_insights=smart_context.get('code_insights', {}),
            architecture_analysis=specialized_analysis.get('architecture'),
            security_analysis=specialized_analysis.get('security'),
            quality_metrics=specialized_analysis.get('quality')
        )
    
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Public wrapper for query intent analysis"""
        return self._analyze_query_intent(query)
    
    def gather_intelligent_context(self, git_analyzer, query: str, query_analysis: Dict[str, Any], file_analyses: List = None) -> Dict[str, Any]:
        """Gather intelligent context using the enhanced RAG approach"""
        try:
            repo_id = getattr(git_analyzer, 'repo_id', None) or getattr(git_analyzer, 'get_repo_id', lambda: None)()
            if not repo_id:
                raise ValueError("Could not determine repo_id from git_analyzer")
            
            # Use the internal method to gather context
            context_files = self._gather_intelligent_context(repo_id, query, query_analysis)
            context_commits = self._gather_intelligent_commits(repo_id, query, query_analysis)
            
            # If file_analyses were provided, use them; otherwise analyze the context files
            if file_analyses is None:
                file_analyses = self._analyze_relevant_files(context_files)
            
            # Create specialized analysis
            specialized_analysis = self._create_specialized_analysis(query_analysis, file_analyses, context_commits)
            
            return {
                'files': context_files,
                'commits': context_commits,
                'file_analyses': file_analyses,
                'specialized_analysis': specialized_analysis,
                'query_analysis': query_analysis,
                'confidence': self._calculate_context_confidence(context_files, context_commits, query_analysis)
            }
        except Exception as e:
            logger.error(f"Error gathering intelligent context: {e}")
            return {
                'files': [],
                'commits': [],
                'file_analyses': file_analyses or [],
                'specialized_analysis': {},
                'query_analysis': query_analysis,
                'confidence': 0.3
            }
    
    def determine_expert_domains(self, query: str, query_analysis: Dict[str, Any]) -> List[str]:
        """Determine which expert domains are relevant for the query"""
        domains = []
        
        # Check primary domain from query analysis
        primary_domain = query_analysis.get('domain', 'general')
        if primary_domain != 'general':
            domains.append(primary_domain)
        
        # Check for domain keywords in query
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in ['security', 'vulnerability', 'auth', 'encrypt', 'permission', 'access']):
            domains.append('security')
        
        if any(keyword in query_lower for keyword in ['architecture', 'design', 'pattern', 'structure', 'component']):
            domains.append('architecture')
        
        if any(keyword in query_lower for keyword in ['performance', 'speed', 'optimization', 'memory', 'cpu', 'slow']):
            domains.append('performance')
        
        if any(keyword in query_lower for keyword in ['quality', 'test', 'bug', 'error', 'code review', 'maintainability']):
            domains.append('quality')
        
        if any(keyword in query_lower for keyword in ['deploy', 'ci/cd', 'pipeline', 'build', 'docker', 'kubernetes']):
            domains.append('devops')
        
        # If no specific domains identified, use general analysis
        if not domains:
            domains = ['quality', 'architecture']  # Default domains for general queries
        
        return list(set(domains))  # Remove duplicates
    
    def generate_intelligent_response(self, context: Dict[str, Any], query: str) -> str:
        """Generate an intelligent response based on the gathered context"""
        try:
            query_analysis = context.get('query_analysis', {})
            specialized_analysis = context.get('specialized_analysis', {})
            file_analyses = context.get('file_analyses', [])
            commits = context.get('commits', [])
            
            # Build response based on query type
            response_parts = []
            
            # Add domain-specific insights
            domain = query_analysis.get('domain', 'general')
            if domain in specialized_analysis:
                domain_data = specialized_analysis[domain]
                if domain == 'security':
                    if domain_data.get('security_issues'):
                        response_parts.append(f"Security Analysis: Found {len(domain_data['security_issues'])} potential security issues.")
                        response_parts.extend(domain_data['security_issues'][:3])  # Top 3 issues
                elif domain == 'architecture':
                    patterns = domain_data.get('patterns_detected', [])
                    if patterns:
                        response_parts.append(f"Architecture Analysis: Detected patterns: {', '.join(patterns)}")
                elif domain == 'performance':
                    complex_files = domain_data.get('complex_files', [])
                    if complex_files:
                        response_parts.append(f"Performance Analysis: {len(complex_files)} files with high complexity detected.")
            
            # Add file-specific insights
            high_relevance_files = [f for f in file_analyses if f.relevance_score > 0.7]
            if high_relevance_files:
                response_parts.append(f"\nRelevant Files ({len(high_relevance_files)} found):")
                for file_analysis in high_relevance_files[:5]:  # Top 5 files
                    response_parts.append(f"- {file_analysis.file_path} (relevance: {file_analysis.relevance_score:.2f})")
            
            # Add commit insights if relevant
            if commits and query_analysis.get('query_type') in ['commit_specific', 'recent_changes']:
                response_parts.append(f"\nRecent Changes: Analyzed {len(commits)} relevant commits.")
                for commit in commits[:3]:  # Top 3 commits
                    response_parts.append(f"- {commit.get('hash', '')[:8]}: {commit.get('message', '')}")
            
            # Add recommendations
            recommendations = self._generate_domain_recommendations(domain, specialized_analysis.get(domain, {}))
            if recommendations:
                response_parts.append(f"\nRecommendations:")
                response_parts.extend([f"- {rec}" for rec in recommendations[:3]])
            
            return "\n".join(response_parts) if response_parts else "Analysis complete. No specific issues detected."
            
        except Exception as e:
            logger.error(f"Error generating intelligent response: {e}")
            return "An error occurred while generating the response."
    
    def _calculate_context_confidence(self, files: List[Dict], commits: List[Dict], query_analysis: Dict[str, Any]) -> float:
        """Calculate confidence score for the gathered context"""
        try:
            confidence = 0.5  # Base confidence
            
            # Boost for relevant files
            if files:
                avg_file_relevance = sum(f.get('relevance_score', 0) for f in files) / len(files)
                confidence += avg_file_relevance * 0.3
            
            # Boost for relevant commits
            if commits:
                avg_commit_relevance = sum(c.get('relevance_score', 0) for c in commits) / len(commits)
                confidence += avg_commit_relevance * 0.2
            
            # Boost for specific query types
            query_type = query_analysis.get('query_type', 'general')
            if query_type in ['commit_specific', 'file_specific']:
                confidence += 0.2
            
            return min(confidence, 1.0)  # Cap at 1.0
        except Exception:
            return 0.5
    
    def _generate_domain_recommendations(self, domain: str, analysis: Dict[str, Any]) -> List[str]:
        """Generate domain-specific recommendations based on analysis"""
        recommendations = []
        
        if domain == 'security' and isinstance(analysis, dict):
            issues = analysis.get('issues_found', [])
            if issues:
                recommendations.append(f"Address {len(issues)} security issues found in the codebase")
                for issue in issues[:2]:  # Top 2 issues
                    recommendations.append(f"Security: {issue}")
            else:
                recommendations.append("No major security issues detected")
        
        elif domain == 'architecture' and isinstance(analysis, dict):
            patterns = analysis.get('patterns_detected', [])
            if patterns:
                recommendations.append(f"Architectural patterns detected: {', '.join(patterns[:3])}")
            else:
                recommendations.append("Consider implementing architectural patterns for better structure")
            
            bottlenecks = analysis.get('potential_bottlenecks', [])
            if bottlenecks:
                recommendations.append(f"Address potential bottlenecks in: {', '.join([b[0] for b in bottlenecks[:2]])}")
        
        elif domain == 'performance' and isinstance(analysis, dict):
            bottlenecks = analysis.get('potential_bottlenecks', [])
            if bottlenecks:
                recommendations.append(f"Optimize {len(bottlenecks)} files with performance bottlenecks")
                recommendations.append(f"Focus on: {bottlenecks[0][0]} (complexity: {bottlenecks[0][1]:.1f})")
            else:
                recommendations.append("No major performance issues detected")
        
        elif domain == 'quality' and isinstance(analysis, dict):
            high_complexity = analysis.get('high_complexity_files', [])
            if high_complexity:
                recommendations.append(f"Refactor {len(high_complexity)} high-complexity files")
                recommendations.append(f"Priority: {high_complexity[0][0]} (complexity: {high_complexity[0][1]:.1f})")
            
            tech_debt = analysis.get('technical_debt_files', [])
            if tech_debt:
                recommendations.append(f"Address technical debt in {len(tech_debt)} files")
            
            if not high_complexity and not tech_debt:
                recommendations.append("Code quality looks good overall")
        
        else:
            # General recommendations for unknown domains
            recommendations.append("Continue following software engineering best practices")
            recommendations.append("Regular code reviews and testing are recommended")
        
        return recommendations[:5]  # Limit to 5 recommendations

    def _curate_files_for_llm(self, file_analyses: List[FileAnalysis], query_intent: Dict[str, Any], 
                              query: str, max_llm_files: int = 15) -> List[FileAnalysis]:
        """Intelligently curate the most relevant files for LLM analysis"""
        if not file_analyses:
            return []
        
        # If this is a file-focused query, prioritize mentioned files
        if query_intent.get('file_focused_query') and query_intent.get('specific_files_mentioned'):
            prioritized_files = []
            remaining_files = []
            
            mentioned_files = query_intent['specific_files_mentioned']
            logger.info(f"🎯 Prioritizing specifically mentioned files: {mentioned_files}")
            
            # First, find files that match the mentioned files
            for file_analysis in file_analyses:
                file_name = os.path.basename(file_analysis.file_path)
                file_path_lower = file_analysis.file_path.lower()
                
                # Check if this file matches any mentioned file
                is_mentioned = False
                for mentioned_file in mentioned_files:
                    mentioned_lower = mentioned_file.lower()
                    if (mentioned_lower in file_name.lower() or 
                        mentioned_lower in file_path_lower or
                        file_name.lower() in mentioned_lower):
                        is_mentioned = True
                        break
                
                if is_mentioned:
                    prioritized_files.append(file_analysis)
                    logger.info(f"✅ Prioritized file: {file_analysis.file_path}")
                else:
                    remaining_files.append(file_analysis)
            
            # If we found the specific files, use them first and fill remaining slots
            if prioritized_files:
                # Start with prioritized files
                curated_files = prioritized_files[:max_llm_files]
                remaining_slots = max_llm_files - len(curated_files)
                
                # Add some context files if we have remaining slots
                if remaining_slots > 0:
                    # Score and sort remaining files
                    scored_remaining = []
                    for file_analysis in remaining_files:
                        score = self._calculate_file_llm_score(file_analysis, query_intent, query)
                        scored_remaining.append((file_analysis, score))
                    
                    scored_remaining.sort(key=lambda x: x[1], reverse=True)
                    curated_files.extend([f[0] for f in scored_remaining[:remaining_slots]])
                
                logger.info(f"🎯 File curation: Selected {len(prioritized_files)} prioritized + {len(curated_files) - len(prioritized_files)} context files")
                return curated_files
        
        # Regular curation logic for non-file-focused queries
        if len(file_analyses) <= max_llm_files:
            return file_analyses
        
        # Score files for LLM relevance
        scored_files = []
        
        for file_analysis in file_analyses:
            score = self._calculate_file_llm_score(file_analysis, query_intent, query)
            scored_files.append((file_analysis, score))
        
        # Sort by score and take top files
        scored_files.sort(key=lambda x: x[1], reverse=True)
        
        # Ensure we include diverse file types
        curated_files = []
        file_types_included = set()
        
        # First pass: include highest scoring files with type diversity
        for file_analysis, score in scored_files:
            if len(curated_files) >= max_llm_files:
                break
                
            file_ext = os.path.splitext(file_analysis.file_path)[1].lower()
            
            # Always include top 5 regardless of type
            if len(curated_files) < 5:
                curated_files.append(file_analysis)
                file_types_included.add(file_ext)
            # Then include diverse types
            elif file_ext not in file_types_included or len(file_types_included) < 3:
                curated_files.append(file_analysis)
                file_types_included.add(file_ext)
            # Fill remaining slots with highest scores
            elif score > 15:  # Only include if significantly relevant
                curated_files.append(file_analysis)
        
        logger.info(f"🎯 File curation: Selected {len(curated_files)}/{len(file_analyses)} files for LLM analysis")
        logger.debug(f"Selected files: {[f.file_path for f in curated_files]}")
        
        return curated_files
    
    def _calculate_file_llm_score(self, file_analysis: FileAnalysis, query_intent: Dict[str, Any], query: str) -> float:
        """Calculate LLM relevance score for a file"""
        score = 0.0
        
        # Base relevance score (already calculated during analysis)
        score += getattr(file_analysis, 'relevance_score', 0.0) * 30
        
        # Query-specific scoring
        query_lower = query.lower()
        file_path_lower = file_analysis.file_path.lower()
        
        # Direct mentions in query
        if any(keyword in file_path_lower for keyword in query_intent.get('keywords', [])):
            score += 25
        
        # Domain-specific relevance
        domain = query_intent.get('primary_domain', 'general')
        if domain == 'security' and file_analysis.security_patterns:
            score += 30
        elif domain == 'architecture' and file_analysis.architecture_patterns:
            score += 25
        elif domain == 'quality' and file_analysis.technical_debt_indicators:
            score += 20
        elif domain == 'performance' and file_analysis.complexity_score > 3:
            score += 20
        
        # File importance factors
        file_name = os.path.basename(file_path_lower)
        if any(name in file_name for name in ['main', 'index', 'app', 'server', 'config']):
            score += 15
        
        # Complexity and content richness
        score += min(file_analysis.complexity_score * 2, 10)  # Cap at 10
        score += min(len(file_analysis.entities) * 0.5, 8)     # Cap at 8
        
        # File size factor (prefer moderately sized files)
        lines_of_code = getattr(file_analysis, 'lines_of_code', 0)
        if 50 <= lines_of_code <= 500:
            score += 5
        elif lines_of_code > 1000:
            score -= 5  # Penalize very large files
        
        return score
    
    def _curate_commits_for_llm(self, commits: List[Dict[str, Any]], query_intent: Dict[str, Any], 
                                query: str, max_llm_commits: int = 8) -> List[Dict[str, Any]]:
        """Intelligently curate the most relevant commits for LLM analysis"""
        if not commits:
            return []
        
        if len(commits) <= max_llm_commits:
            return commits
        
        # Score commits for LLM relevance
        scored_commits = []
        
        for commit in commits:
            score = commit.get('score', 0.0)  # Base score from commit analysis
            
            # Query-specific relevance
            message_lower = commit.get('message', '').lower()
            query_lower = query.lower()
            
            # Direct keyword matches
            for keyword in query_intent.get('keywords', []):
                if keyword in message_lower:
                    score += 10
            
            # Domain-specific scoring
            domain = query_intent.get('primary_domain', 'general')
            domain_keywords = {
                'security': ['security', 'auth', 'vulnerability', 'fix', 'patch'],
                'performance': ['performance', 'optimization', 'speed', 'cache'],
                'architecture': ['refactor', 'architecture', 'design', 'structure'],
                'quality': ['cleanup', 'refactor', 'improve', 'quality'],
                'breaking_changes': ['breaking', 'change', 'api', 'signature', 'deprecated']
            }
            
            
            if domain in domain_keywords:
                for keyword in domain_keywords[domain]:
                    if keyword in message_lower:
                        score += 8
            
            # Recency boost
            try:
                commit_date = commit.get('date', '')
                if commit_date:
                    from datetime import datetime, timezone
                    commit_datetime = datetime.fromisoformat(commit_date.replace('Z', '+00:00'))
                    days_ago = (datetime.now(timezone.utc) - commit_datetime).days
                    if days_ago < 7:
                        score += 10
                    elif days_ago < 30:
                        score += 5
            except:
                pass
            
            # Files changed factor
            files_changed = len(commit.get('files_changed', []))
            if 1 <= files_changed <= 5:
                score += 5  # Focused changes
            elif files_changed > 20:
                score -= 3  # Very large changes may be less focused
            
            # Important change indicators
            if any(word in message_lower for word in ['fix', 'bug', 'critical', 'important', 'breaking']):
                score += 8
            
            scored_commits.append((commit, score))
        
        # Sort by score and take top commits
        scored_commits.sort(key=lambda x: x[1], reverse=True)
        
        # Select diverse commits (not all from same author/time period)
        curated_commits = []
        authors_included = set()
        
        for commit, score in scored_commits:
            if len(curated_commits) >= max_llm_commits:
                break
                
            author = commit.get('author', 'unknown')
            
            # Always include top 3 regardless of author
            if len(curated_commits) < 3:
                curated_commits.append(commit)
                authors_included.add(author)
            # Then ensure author diversity
            elif author not in authors_included or len(authors_included) < 3:
                curated_commits.append(commit)
                authors_included.add(author)
            # Fill remaining with highest scores
            elif score > 10:  # Only include if significantly relevant
                curated_commits.append(commit)
        
        logger.info(f"🎯 Commit curation: Selected {len(curated_commits)}/{len(commits)} commits for LLM analysis")
        
        return curated_commits

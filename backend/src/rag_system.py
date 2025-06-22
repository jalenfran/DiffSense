"""
Smart RAG System - Intelligent Context Gathering
Finds exactly the right files, commits, and diffs for Claude analysis
"""

import re
import os
import git
import logging
import tempfile
import shutil
import subprocess
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Beautiful response from Smart Claude"""
    query: str
    response: str
    confidence: float
    sources: List[Dict[str, Any]]
    context_used: List[str]
    suggestions: List[str]

class RepositoryKnowledgeBase:
    """Smart context gatherer that feeds Claude with perfect information"""
    
    def __init__(self, git_analyzer, embedding_engine=None, claude_analyzer=None):
        self.git_analyzer = git_analyzer
        self.embedding_engine = embedding_engine
        self.claude_analyzer = claude_analyzer
    
    def query(self, repo_id: str, query: str, max_results: int = 5) -> QueryResult:
        """Main entry: analyze query and gather perfect context"""
        
        # Analyze what user wants
        query_type, keywords, filters = self._analyze_query(query)
        
        # Get repo context
        repo_context = self._get_repo_context(repo_id)
        
        # Gather context based on query type
        if query_type == 'file':
            context = self._gather_file_context(repo_id, keywords, filters)
        elif query_type == 'commit':
            context = self._gather_commit_context(repo_id, keywords, filters)
        elif query_type == 'author':
            context = self._gather_author_context(repo_id, keywords, filters)
        elif query_type == 'security':
            context = self._gather_security_context(repo_id, keywords, filters)
        elif query_type == 'diff':
            context = self._gather_diff_context(repo_id, keywords, filters)
        else:
            context = self._gather_general_context(repo_id, keywords, filters)
        
        # Calculate confidence based on context quality
        confidence = self._calculate_confidence(context, query)
        
        # Actually use Claude for analysis if available
        if self.claude_analyzer and self.claude_analyzer.available:
            try:
                # Create SmartContext for Claude
                from src.claude_analyzer import SmartContext
                
                claude_context = SmartContext(
                    query=query,
                    query_type=query_type,
                    files=context.get('files', []),
                    commits=context.get('commits', []),
                    repo_context=self._get_repo_context(repo_id),
                    confidence=confidence,
                    reasoning=context.get('reasoning', f'{query_type} analysis')
                )
                
                # Get Claude's enhanced analysis
                claude_response = self.claude_analyzer.analyze(claude_context)
                
                # Use Claude's response instead of the basic template
                return QueryResult(
                    query=claude_response.query,
                    response=claude_response.response,
                    confidence=claude_response.confidence,
                    sources=claude_response.sources,
                    context_used=claude_response.context_used,
                    suggestions=claude_response.suggestions
                )
                
            except Exception as e:
                logger.warning(f"Claude analysis failed, falling back to basic response: {e}")
                # Fall through to basic response
        
        # Create basic response from context (fallback when Claude not available or fails)
        response_parts = [
            f"# 📊 Smart Analysis Results",
            f"",
            f"**Query**: {query}",
            f"**Analysis Type**: {query_type}",
            f"**Context Quality**: {confidence:.0%}",
            f"",
        ]
        
        # Add findings
        if context.get('files'):
            response_parts.extend([
                f"## 📁 Files Found ({len(context['files'])})",
                ""
            ])
            for file_data in context['files'][:8]:  # Increased from 3 to 8
                size_kb = file_data.get('size', 0) / 1024
                response_parts.append(f"- **{file_data['path']}** ({file_data.get('language', 'text')}, {size_kb:.1f}KB)")
            if len(context['files']) > 8:
                response_parts.append(f"- *...and {len(context['files']) - 8} more files*")
            response_parts.append("")
        
        if context.get('commits'):
            response_parts.extend([
                f"## 📝 Commits Found ({len(context['commits'])})",
                ""
            ])
            for commit in context['commits'][:6]:  # Increased from 3 to 6
                response_parts.append(f"- **{commit.get('hash', '')[:8]}**: {commit.get('message', '')[:60]}...")
            if len(context['commits']) > 6:
                response_parts.append(f"- *...and {len(context['commits']) - 6} more commits*")
            response_parts.append("")
        
        response_parts.extend([
            f"## 💡 Summary",
            f"",
            f"Context gathered successfully with {confidence:.0%} confidence.",
            f"Reasoning: {context.get('reasoning', 'General analysis')}",
            f"",
        ])
        
        # Only add Claude configuration message if Claude is not available
        if not self.claude_analyzer or not self.claude_analyzer.available:
            response_parts.append(f"*Configure Claude API for enhanced AI analysis.*")
        else:
            response_parts.append(f"*Analysis enhanced with Claude AI.*")
        
        # Create sources
        sources = []
        for file_data in context.get('files', []):
            sources.append({
                'type': 'file',
                'path': file_data['path'],
                'language': file_data.get('language', 'text'),
                'size_kb': round(file_data.get('size', 0) / 1024, 1)
            })
        
        for commit in context.get('commits', []):
            sources.append({
                'type': 'commit',
                'hash': commit.get('hash', '')[:8],
                'message': commit.get('message', '')[:50],
                'author': commit.get('author', 'Unknown')
            })
        
        # Create suggestions
        suggestions = []
        if self.claude_analyzer and not self.claude_analyzer.available:
            suggestions.append("Configure Claude API for AI-powered analysis")
        elif not self.claude_analyzer:
            suggestions.append("Configure Claude API for AI-powered analysis")
        
        if context.get('files'):
            suggestions.append(f"Review the {len(context['files'])} relevant files")
        if context.get('commits'):
            suggestions.append(f"Examine the {len(context['commits'])} related commits")
        
        return QueryResult(
            query=query,
            response="\n".join(response_parts),
            confidence=confidence,
            sources=sources,
            context_used=[f"Smart context gathered: {context.get('reasoning', 'General analysis')}"],
            suggestions=suggestions
        )
    
    def _analyze_query(self, query: str) -> Tuple[str, List[str], Dict[str, Any]]:
        """Understand what user wants"""
        query_lower = query.lower()
        
        # Extract keywords
        keywords = re.findall(r'\b\w+\b', query_lower)
        
        # Detect query type
        if any(word in query_lower for word in ['file', 'function', 'class', 'method', 'code']):
            query_type = 'file'
        elif any(word in query_lower for word in ['commit', 'change', 'history', 'when', 'who changed']):
            query_type = 'commit'  
        elif any(word in query_lower for word in ['author', 'developer', 'contributor', 'who wrote']):
            query_type = 'author'
        elif any(word in query_lower for word in ['security', 'vulnerability', 'risk', 'exploit', 'auth']):
            query_type = 'security'
        elif any(word in query_lower for word in ['diff', 'difference', 'compare', 'between']):
            query_type = 'diff'
        else:
            query_type = 'general'
        
        # Extract filters
        filters = {}
        
        # Time filters
        if 'recent' in query_lower or 'latest' in query_lower:
            filters['time_limit'] = 30  # days
        elif 'last week' in query_lower:
            filters['time_limit'] = 7
        elif 'last month' in query_lower:
            filters['time_limit'] = 30
        
        # File type filters
        file_extensions = re.findall(r'\.(\w+)', query)
        if file_extensions:
            filters['file_types'] = file_extensions
        
        # Author filters
        author_match = re.search(r'by\s+(\w+)', query_lower)
        if author_match:
            filters['author'] = author_match.group(1)
        
        return query_type, keywords, filters
    
    def _get_repo_context(self, repo_id: str) -> Dict[str, Any]:
        """Get repository overview"""
        try:
            # Use the git analyzer's repo directly
            repo = self.git_analyzer.repo
            
            # Get basic stats
            commits = list(repo.iter_commits(max_count=1000))
            all_files = []
            
            try:
                for item in repo.head.commit.tree.traverse():
                    if item.type == 'blob':
                        all_files.append(item.path)
            except:
                pass
            
            # Get language distribution
            languages = {}
            for file_path in all_files:
                ext = os.path.splitext(file_path)[1].lower()
                if ext:
                    languages[ext] = languages.get(ext, 0) + 1
            
            return {
                'repo_id': repo_id,
                'total_commits': len(commits),
                'total_files': len(all_files),
                'languages': dict(sorted(languages.items(), key=lambda x: x[1], reverse=True)[:5]),
                'recent_activity': len([c for c in commits if (datetime.now() - c.committed_datetime.replace(tzinfo=None)) < timedelta(days=30)])
            }
        except Exception as e:
            logger.error(f"Error getting repo context: {e}")
            return {'repo_id': repo_id, 'error': str(e)}
    
    def _gather_file_context(self, repo_id: str, keywords: List[str], filters: Dict[str, Any]) -> Dict[str, Any]:
        """Gather file-specific context"""
        try:
            # Use the git analyzer's repo directly
            repo = self.git_analyzer.repo
            
            files = []
            reasoning_parts = []
            
            # Get all files
            all_files = []
            try:
                for item in repo.head.commit.tree.traverse():
                    if item.type == 'blob':
                        all_files.append(item.path)
            except:
                pass
            
            # DEBUG: Log all files found in repo
            logger.info(f"🔍 RAG DEBUG - File Analysis for query: '{' '.join(keywords)}'")
            logger.info(f"  📁 Total files in repository: {len(all_files)}")
            
            # Filter by file type if specified
            if 'file_types' in filters:
                all_files = [f for f in all_files if any(f.endswith(f'.{ext}') for ext in filters['file_types'])]
                reasoning_parts.append(f"Filtered to {filters['file_types']} files")
                logger.info(f"  🔧 After file type filter: {len(all_files)} files")
            
            # Score files by keyword relevance
            scored_files = []
            for file_path in all_files:
                score = 1  # Give every file a base score so it gets included
                file_lower = file_path.lower()
                
                # High score for filename match
                for keyword in keywords:
                    if keyword in file_lower:
                        score += 20
                
                # Score by file type relevance (expanded list)
                ext = os.path.splitext(file_path)[1].lower()
                if ext in ['.py', '.js', '.ts', '.jsx', '.tsx', '.vue', '.java', '.cpp', '.c', '.h', 
                          '.cs', '.rb', '.php', '.go', '.rs', '.swift', '.kt', '.scala', '.r', 
                          '.sql', '.html', '.css', '.scss', '.less', '.json', '.xml', '.yaml', '.yml',
                          '.md', '.txt', '.sh', '.bat', '.ps1', '.dockerfile']:
                    score += 10
                
                # Boost for important files
                filename = os.path.basename(file_path).lower()
                if any(name in filename for name in ['main', 'index', 'app', 'server', 'client', 
                                                   'config', 'setup', 'init', 'readme', 'api',
                                                   'model', 'view', 'controller', 'service', 'utils']):
                    score += 15
                
                # Only slightly reduce test files, don't exclude them
                if 'test' not in ' '.join(keywords) and ('test' in file_lower or 'spec' in file_lower):
                    score -= 3  # Much smaller penalty
                
                # Include all files, even with score 0
                scored_files.append((file_path, score))
            
            # Sort by score and take more files
            scored_files.sort(key=lambda x: x[1], reverse=True)
            top_files = scored_files[:20]  # Increased from 5 to 20 files
            
            # DEBUG: Log file scoring results
            logger.info(f"  📊 File scoring completed:")
            logger.info(f"    🥇 Top 20 scored files (showing top 10):")
            for i, (file_path, score) in enumerate(top_files[:10]):
                logger.info(f"      {i+1}. {file_path} (score: {score})")
            if len(top_files) > 10:
                logger.info(f"      ... and {len(top_files)-10} more files")
            
            # Get file contents
            for file_path, score in top_files:
                try:
                    file_content = repo.head.commit.tree[file_path].data_stream.read().decode('utf-8', errors='ignore')
                    
                    # Determine language
                    ext = os.path.splitext(file_path)[1].lower()
                    language_map = {
                        '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
                        '.jsx': 'javascript', '.tsx': 'typescript', '.vue': 'vue',
                        '.java': 'java', '.cpp': 'cpp', '.c': 'c', '.h': 'c',
                        '.rs': 'rust', '.go': 'go', '.rb': 'ruby', '.php': 'php',
                        '.cs': 'csharp', '.swift': 'swift', '.kt': 'kotlin',
                        '.md': 'markdown', '.txt': 'text', '.json': 'json',
                        '.xml': 'xml', '.html': 'html', '.css': 'css', '.scss': 'scss',
                        '.sql': 'sql', '.sh': 'bash', '.bat': 'batch'
                    }
                    language = language_map.get(ext, 'text')
                    
                    files.append({
                        'path': file_path,
                        'content': file_content,
                        'language': language,
                        'score': score,
                        'size': len(file_content)
                    })
                    
                    # DEBUG: Log file content details
                    logger.info(f"    📄 Loaded: {file_path} ({language}, {len(file_content):,} chars, score: {score})")
                    
                except Exception as e:
                    logger.debug(f"Could not read file {file_path}: {e}")
            
            reasoning_parts.append(f"Found {len(files)} relevant files")
            logger.info(f"  ✅ Final result: {len(files)} files loaded with content")
            
            return {
                'files': files,
                'reasoning': '; '.join(reasoning_parts) if reasoning_parts else 'File analysis'
            }
            
        except Exception as e:
            logger.error(f"Error gathering file context: {e}")
            return {'files': [], 'reasoning': f'Error: {e}'}
    
    def _gather_commit_context(self, repo_id: str, keywords: List[str], filters: Dict[str, Any]) -> Dict[str, Any]:
        """Gather commit-specific context"""
        try:
            # Use the git analyzer's repo directly
            repo = self.git_analyzer.repo
            
            commits = []
            reasoning_parts = []
            
            # Get commits with time filter
            max_count = 500  # Increased from 100 to 500 for better coverage
            if 'time_limit' in filters:
                since_date = datetime.now() - timedelta(days=filters['time_limit'])
                commit_iter = repo.iter_commits(since=since_date, max_count=max_count)
                reasoning_parts.append(f"Last {filters['time_limit']} days")
            else:
                commit_iter = repo.iter_commits(max_count=max_count)
            
            # Score commits by relevance
            scored_commits = []
            for commit in commit_iter:
                score = 0
                commit_text = f"{commit.message} {commit.author.name}".lower()
                
                # Score by keyword match
                for keyword in keywords:
                    if keyword in commit_text:
                        score += 10
                
                # Score by files changed
                try:
                    if commit.parents:
                        files_changed = len(commit.stats.files)
                        if files_changed > 1:
                            score += min(files_changed, 5)
                except:
                    pass
                
                # Give every commit a base score to include more commits
                if score == 0:
                    score = 1
                
                scored_commits.append((commit, score))
            
            # Sort and take top commits
            scored_commits.sort(key=lambda x: x[1], reverse=True)
            top_commits = scored_commits[:25]  # Increased from 10 to 25 commits
            
            # Get commit details with diffs
            for commit, score in top_commits:
                try:
                    commit_data = {
                        'hash': commit.hexsha,
                        'message': commit.message.strip(),
                        'author': commit.author.name,
                        'email': commit.author.email,
                        'date': commit.committed_datetime.isoformat(),
                        'score': score,
                        'files_changed': []
                    }
                    
                    # Get diff for this commit
                    if commit.parents:
                        diffs = commit.parents[0].diff(commit, create_patch=True)
                        for diff_item in diffs:
                            if diff_item.a_path:
                                commit_data['files_changed'].append({
                                    'file': diff_item.a_path,
                                    'change_type': diff_item.change_type,
                                    'diff': str(diff_item)[:2000]  # Limit diff size
                                })
                    
                    commits.append(commit_data)
                    
                except Exception as e:
                    logger.debug(f"Error processing commit {commit.hexsha}: {e}")
            
            reasoning_parts.append(f"Found {len(commits)} relevant commits")
            
            return {
                'commits': commits,
                'reasoning': '; '.join(reasoning_parts) if reasoning_parts else 'Commit analysis'
            }
            
        except Exception as e:
            logger.error(f"Error gathering commit context: {e}")
            return {'commits': [], 'reasoning': f'Error: {e}'}
    
    def _gather_author_context(self, repo_id: str, keywords: List[str], filters: Dict[str, Any]) -> Dict[str, Any]:
        """Gather author-specific context"""
        # Reuse commit gathering but filter by author
        if 'author' not in filters and keywords:
            # Try to extract author from keywords
            filters['author'] = keywords[0]
        
        context = self._gather_commit_context(repo_id, keywords, filters)
        
        if 'author' in filters:
            # Filter commits by author
            author_filter = filters['author'].lower()
            filtered_commits = []
            for commit in context.get('commits', []):
                if author_filter in commit.get('author', '').lower() or author_filter in commit.get('email', '').lower():
                    filtered_commits.append(commit)
            
            context['commits'] = filtered_commits
            context['reasoning'] = f"Author analysis for '{filters['author']}'"
        
        return context
    
    def _gather_security_context(self, repo_id: str, keywords: List[str], filters: Dict[str, Any]) -> Dict[str, Any]:
        """Gather security-specific context"""
        # Get both files and commits, focus on security patterns
        file_context = self._gather_file_context(repo_id, keywords + ['auth', 'password', 'token', 'secret'], filters)
        commit_context = self._gather_commit_context(repo_id, keywords + ['security', 'fix', 'vulnerability'], filters)
        
        # Filter for security-relevant content
        security_files = []
        for file_data in file_context.get('files', []):
            content = file_data['content'].lower()
            if any(pattern in content for pattern in ['password', 'secret', 'token', 'auth', 'login', 'security']):
                file_data['security_score'] = 10
                security_files.append(file_data)
        
        return {
            'files': security_files,
            'commits': commit_context.get('commits', []),
            'reasoning': 'Security analysis'
        }
    
    def _gather_diff_context(self, repo_id: str, keywords: List[str], filters: Dict[str, Any]) -> Dict[str, Any]:
        """Gather diff-specific context"""
        # Focus on commits with substantial changes
        commit_context = self._gather_commit_context(repo_id, keywords, filters)
        
        # Enhance with more detailed diffs
        enhanced_commits = []
        for commit in commit_context.get('commits', []):
            if len(commit.get('files_changed', [])) > 0:
                enhanced_commits.append(commit)
        
        return {
            'commits': enhanced_commits[:5],  # Limit to 5 for detailed diff analysis
            'reasoning': 'Diff analysis'
        }
    
    def _gather_general_context(self, repo_id: str, keywords: List[str], filters: Dict[str, Any]) -> Dict[str, Any]:
        """Gather general context when query type is unclear"""
        try:
            # Get both files and commits with lower limits
            file_context = self._gather_file_context(repo_id, keywords, filters)
            commit_context = self._gather_commit_context(repo_id, keywords, filters)
            
            # If no specific files found, try to get some general interesting files
            files = file_context.get('files', [])
            if not files:
                # Get some main files like README, main files, etc.
                repo = self.git_analyzer.repo
                interesting_files = []
                try:
                    for item in repo.head.commit.tree.traverse():
                        if item.type == 'blob':
                            file_path = item.path
                            if any(name in file_path.lower() for name in ['readme', 'main', 'index', 'app', 'server', 
                                                                         'config', 'setup', 'api', 'model', 'view', 
                                                                         'controller', 'service', 'utils', 'component',
                                                                         'router', 'middleware', 'database', 'db']):
                                try:
                                    file_content = item.data_stream.read().decode('utf-8', errors='ignore')
                                    ext = os.path.splitext(file_path)[1].lower()
                                    language_map = {
                                        '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
                                        '.jsx': 'javascript', '.tsx': 'typescript', '.vue': 'vue',
                                        '.java': 'java', '.cpp': 'cpp', '.c': 'c', '.h': 'c',
                                        '.rs': 'rust', '.go': 'go', '.rb': 'ruby', '.php': 'php',
                                        '.cs': 'csharp', '.swift': 'swift', '.kt': 'kotlin',
                                        '.md': 'markdown', '.txt': 'text', '.json': 'json',
                                        '.xml': 'xml', '.html': 'html', '.css': 'css', '.scss': 'scss'
                                    }
                                    language = language_map.get(ext, 'text')
                                    
                                    interesting_files.append({
                                        'path': file_path,
                                        'content': file_content,
                                        'language': language,
                                        'score': 5,
                                        'size': len(file_content)
                                    })
                                    
                                    if len(interesting_files) >= 15:  # Increased from 3 to 15
                                        break
                                except:
                                    pass
                except:
                    pass
                
                files = interesting_files
                
                # If still not enough files, add some random files to ensure broad coverage
                if len(files) < 10:
                    try:
                        additional_files = []
                        file_paths_already_included = {f['path'] for f in files}
                        
                        for item in repo.head.commit.tree.traverse():
                            if item.type == 'blob' and item.path not in file_paths_already_included:
                                # Skip binary files and very large files
                                if item.size > 50000:  # Skip files larger than 50KB
                                    continue
                                    
                                ext = os.path.splitext(item.path)[1].lower()
                                # Focus on code and config files
                                if ext in ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h',
                                          '.cs', '.rb', '.php', '.go', '.rs', '.swift', '.kt', '.scala',
                                          '.json', '.xml', '.yaml', '.yml', '.md', '.txt', '.sql',
                                          '.html', '.css', '.scss', '.less', '.sh', '.bat']:
                                    try:
                                        file_content = item.data_stream.read().decode('utf-8', errors='ignore')
                                        language_map = {
                                            '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
                                            '.jsx': 'javascript', '.tsx': 'typescript', '.vue': 'vue',
                                            '.java': 'java', '.cpp': 'cpp', '.c': 'c', '.h': 'c',
                                            '.rs': 'rust', '.go': 'go', '.rb': 'ruby', '.php': 'php',
                                            '.cs': 'csharp', '.swift': 'swift', '.kt': 'kotlin',
                                            '.md': 'markdown', '.txt': 'text', '.json': 'json',
                                            '.xml': 'xml', '.html': 'html', '.css': 'css', '.scss': 'scss',
                                            '.sql': 'sql', '.sh': 'bash', '.bat': 'batch'
                                        }
                                        language = language_map.get(ext, 'text')
                                        
                                        additional_files.append({
                                            'path': item.path,
                                            'content': file_content,
                                            'language': language,
                                            'score': 2,  # Lower score for random files
                                            'size': len(file_content)
                                        })
                                        
                                        if len(files) + len(additional_files) >= 20:
                                            break
                                    except:
                                        pass
                        
                        files.extend(additional_files)
                    except:
                        pass
            
            # If no specific commits found, get recent commits
            commits = commit_context.get('commits', [])
            if not commits:
                try:
                    repo = self.git_analyzer.repo
                    recent_commits = []
                    for commit in repo.iter_commits(max_count=5):
                        try:
                            commit_data = {
                                'hash': commit.hexsha,
                                'message': commit.message.strip(),
                                'author': commit.author.name,
                                'email': commit.author.email,
                                'date': commit.committed_datetime.isoformat(),
                                'score': 3,
                                'files_changed': []
                            }
                            
                            # Get basic diff info
                            if commit.parents:
                                try:
                                    files_changed = len(commit.stats.files)
                                    commit_data['files_changed'] = [{'file': f'~{files_changed} files changed'}]
                                except:
                                    pass
                            
                            recent_commits.append(commit_data)
                        except:
                            pass
                    
                    commits = recent_commits
                except:
                    pass
            
            return {
                'files': files[:10],  # Increased from 3 to 10 files
                'commits': commits[:8],  # Increased from 5 to 8 commits
                'reasoning': 'General analysis with repository overview'
            }
            
        except Exception as e:
            logger.error(f"Error gathering general context: {e}")
            return {
                'files': [],
                'commits': [],
                'reasoning': f'General analysis (with errors: {e})'
            }
    
    def _calculate_confidence(self, context: Dict[str, Any], query: str) -> float:
        """Calculate confidence score based on context quality"""
        score = 0.5  # Base score
        
        # Boost for relevant files found
        files = context.get('files', [])
        if files:
            score += min(len(files) * 0.1, 0.3)
        
        # Boost for relevant commits found
        commits = context.get('commits', [])
        if commits:
            score += min(len(commits) * 0.05, 0.2)
        
        # Boost for high-scoring content
        if files:
            avg_file_score = sum(f.get('score', 0) for f in files) / len(files)
            score += min(avg_file_score / 100, 0.2)
        
        return min(score, 0.95)  # Cap at 95%
    
    def index_repository(self, git_analyzer, repo_id: str, repo_url: str):
        """Index repository for smart queries"""
        try:
            logger.info(f"Indexing repository {repo_id}")
            # Store git analyzer reference
            self.git_analyzer = git_analyzer
            
            # Get repo context for future queries
            self.repo_context = self._get_repo_context(repo_id)
            logger.info(f"Successfully indexed repository {repo_id}")
        except Exception as e:
            logger.error(f"Error indexing repository {repo_id}: {e}")
    
    def get_repository_summary(self, repo_id: str) -> Dict[str, Any]:
        """Get repository summary"""
        if not hasattr(self, 'git_analyzer') or not self.git_analyzer:
            return {"error": "Repository not indexed"}
        
        try:
            repo_context = self._get_repo_context(repo_id)
            
            # Get recent commits
            repo = self.git_analyzer.repo
            commits = list(repo.iter_commits(max_count=50))
            
            recent_activity = []
            for commit in commits[:10]:
                try:
                    files_changed = len(commit.stats.files) if commit.parents else 0
                    recent_activity.append({
                        "commit_hash": commit.hexsha,
                        "message": commit.message.strip(),
                        "author": commit.author.name,
                        "risk_score": 0.5,  # Default risk score
                        "files_changed": files_changed
                    })
                except:
                    pass
            
            return {
                "repository": repo_context,
                "statistics": {
                    "total_commits_indexed": len(commits),
                    "total_files_indexed": repo_context.get('total_files', 0),
                    "high_risk_commits": 0,
                    "average_risk_score": 0.5
                },
                "recent_activity": recent_activity
            }
            
        except Exception as e:
            logger.error(f"Error getting repository summary: {e}")
            return {"error": str(e)}
    
    def search_commits(self, repo_id: str, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search commits by query"""
        try:
            # Use the same smart context gathering for commits
            context = self._gather_commit_context(repo_id, [query], {})
            commits = context.get('commits', [])
            return commits[:max_results]
        except Exception as e:
            logger.error(f"Error searching commits: {e}")
            return []
    
    def search_files(self, repo_id: str, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search files by query"""
        try:
            # Use the same smart context gathering for files
            context = self._gather_file_context(repo_id, [query], {})
            files = context.get('files', [])
            
            # Convert to expected format
            file_results = []
            for file_data in files[:max_results]:
                file_results.append({
                    'file_path': file_data['path'],
                    'language': file_data.get('language', 'text'),
                    'size': file_data.get('size', 0),
                    'score': file_data.get('score', 0)
                })
            
            return file_results
        except Exception as e:
            logger.error(f"Error searching files: {e}")
            return []

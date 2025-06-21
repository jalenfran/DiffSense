"""
Repository Knowledge Base and RAG System
Provides intelligent querying capabilities over repository data
"""

import json
import pickle
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime
import logging
import os

logger = logging.getLogger(__name__)

@dataclass
class RepositoryContext:
    """Repository context for RAG system"""
    repo_id: str
    repo_url: str
    total_commits: int
    contributors: List[str]
    primary_languages: List[str]
    file_structure: Dict[str, Any]
    commit_frequency: Dict[str, int]  # commits per day/week/month
    risk_patterns: Dict[str, Any]

@dataclass
class CommitContext:
    """Context information for a commit"""
    commit_hash: str
    message: str
    author: str
    timestamp: str
    files_changed: List[str]
    additions: int
    deletions: int
    breaking_changes: List[Dict[str, Any]]
    risk_score: float
    semantic_embedding: Optional[np.ndarray] = None

@dataclass
class FileContext:
    """Context information for a file"""
    file_path: str
    language: str
    size: int
    last_modified: str
    modification_frequency: int
    main_contributors: List[str]
    functions: List[str]
    classes: List[str]
    dependencies: List[str]
    semantic_embedding: Optional[np.ndarray] = None

@dataclass
class QueryResult:
    """Result from RAG query"""
    query: str
    response: str
    confidence: float
    sources: List[Dict[str, Any]]
    context_used: List[str]
    suggestions: List[str]

class RepositoryKnowledgeBase:
    """Repository knowledge base for RAG system"""
    
    def __init__(self, embedding_engine, storage_path: str = "./knowledge_base"):
        self.embedding_engine = embedding_engine
        self.storage_path = storage_path
        self.repositories: Dict[str, RepositoryContext] = {}
        self.commits: Dict[str, List[CommitContext]] = {}  # repo_id -> commits
        self.files: Dict[str, List[FileContext]] = {}  # repo_id -> files
        self.embeddings: Dict[str, np.ndarray] = {}  # content_hash -> embedding
        
        # Ensure storage directory exists
        os.makedirs(storage_path, exist_ok=True)
        
        # Load existing data
        self._load_knowledge_base()
    
    def index_repository(self, git_analyzer, repo_id: str, repo_url: str) -> None:
        """Index a repository for RAG queries"""
        logger.info(f"Indexing repository {repo_id}")
        
        try:
            # Build repository context
            repo_context = self._build_repository_context(git_analyzer, repo_id, repo_url)
            self.repositories[repo_id] = repo_context
            
            # Index commits
            commit_contexts = self._index_commits(git_analyzer, repo_id)
            self.commits[repo_id] = commit_contexts
            
            # Index files
            file_contexts = self._index_files(git_analyzer, repo_id)
            self.files[repo_id] = file_contexts
            
            # Save to persistent storage
            self._save_knowledge_base()
            
            logger.info(f"Successfully indexed repository {repo_id}")
            
        except Exception as e:
            logger.error(f"Error indexing repository {repo_id}: {str(e)}")
            raise
    
    def query(self, repo_id: str, query: str, max_results: int = 5) -> QueryResult:
        """Query the knowledge base for information"""
        if repo_id not in self.repositories:
            raise ValueError(f"Repository {repo_id} not found in knowledge base")
        
        # Get query embedding
        query_embedding = self.embedding_engine.get_embedding(query)
        if query_embedding is None:
            return QueryResult(
                query=query,
                response="Unable to process query - embedding generation failed",
                confidence=0.0,
                sources=[],
                context_used=[],
                suggestions=[]
            )
        
        # Find relevant context
        relevant_contexts = self._find_relevant_context(repo_id, query_embedding, max_results)
        
        # Generate response
        response = self._generate_response(query, relevant_contexts, self.repositories[repo_id])
        
        return response
    
    def get_repository_summary(self, repo_id: str) -> Dict[str, Any]:
        """Get comprehensive repository summary"""
        if repo_id not in self.repositories:
            raise ValueError(f"Repository {repo_id} not found")
        
        repo_context = self.repositories[repo_id]
        commits = self.commits.get(repo_id, [])
        files = self.files.get(repo_id, [])
        
        # Calculate risk metrics
        high_risk_commits = [c for c in commits if c.risk_score > 0.7]
        avg_risk_score = np.mean([c.risk_score for c in commits]) if commits else 0.0
        
        # Most active files
        file_activity = {}
        for commit in commits:
            for file_path in commit.files_changed:
                file_activity[file_path] = file_activity.get(file_path, 0) + 1
        
        most_active_files = sorted(file_activity.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "repository": asdict(repo_context),
            "statistics": {
                "total_commits_indexed": len(commits),
                "total_files_indexed": len(files),
                "high_risk_commits": len(high_risk_commits),
                "average_risk_score": avg_risk_score,
                "most_active_files": most_active_files
            },
            "recent_activity": [
                {
                    "commit_hash": c.commit_hash[:8],
                    "message": c.message[:100] + "..." if len(c.message) > 100 else c.message,
                    "author": c.author,
                    "risk_score": c.risk_score,
                    "files_changed": len(c.files_changed)
                }
                for c in sorted(commits, key=lambda x: x.timestamp, reverse=True)[:10]
            ]
        }
    
    def search_commits(self, repo_id: str, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search commits by query"""
        if repo_id not in self.commits:
            return []
        
        commits = self.commits[repo_id]
        query_embedding = self.embedding_engine.get_embedding(query)
        
        if query_embedding is None:
            # Fallback to text search
            results = []
            query_lower = query.lower()
            for commit in commits:
                if (query_lower in commit.message.lower() or 
                    query_lower in commit.author.lower() or
                    any(query_lower in f.lower() for f in commit.files_changed)):
                    results.append(commit)
            return [asdict(c) for c in results[:max_results]]
        
        # Semantic search
        similarities = []
        for commit in commits:
            if commit.semantic_embedding is not None:
                similarity = np.dot(query_embedding, commit.semantic_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(commit.semantic_embedding)
                )
                similarities.append((similarity, commit))
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [asdict(c[1]) for c in similarities[:max_results]]
    
    def search_files(self, repo_id: str, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search files by query"""
        if repo_id not in self.files:
            return []
        
        files = self.files[repo_id]
        query_embedding = self.embedding_engine.get_embedding(query)
        
        if query_embedding is None:
            # Fallback to text search
            results = []
            query_lower = query.lower()
            for file_ctx in files:
                if (query_lower in file_ctx.file_path.lower() or
                    any(query_lower in f.lower() for f in file_ctx.functions) or
                    any(query_lower in c.lower() for c in file_ctx.classes)):
                    results.append(file_ctx)
            return [asdict(f) for f in results[:max_results]]
        
        # Semantic search
        similarities = []
        for file_ctx in files:
            if file_ctx.semantic_embedding is not None:
                similarity = np.dot(query_embedding, file_ctx.semantic_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(file_ctx.semantic_embedding)
                )
                similarities.append((similarity, file_ctx))
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [asdict(f[1]) for f in similarities[:max_results]]
    
    def _build_repository_context(self, git_analyzer, repo_id: str, repo_url: str) -> RepositoryContext:
        """Build repository context"""
        repo = git_analyzer.repo
        
        # Get basic stats
        commits = list(repo.iter_commits())
        contributors = list(set(commit.author.name for commit in commits))
        
        # Analyze file structure and languages
        file_structure = {}
        language_counts = {}
        
        for item in repo.head.commit.tree.traverse():
            if item.type == 'blob':
                path_parts = item.path.split('/')
                current = file_structure
                for part in path_parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[path_parts[-1]] = {'size': item.size, 'type': 'file'}
                
                # Count languages
                ext = os.path.splitext(item.path)[1].lower()
                language_counts[ext] = language_counts.get(ext, 0) + 1
        
        # Get primary languages
        primary_languages = sorted(language_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        primary_languages = [lang[0] for lang in primary_languages if lang[0]]
        
        # Analyze commit frequency
        commit_frequency = self._analyze_commit_frequency(commits)
        
        return RepositoryContext(
            repo_id=repo_id,
            repo_url=repo_url,
            total_commits=len(commits),
            contributors=contributors,
            primary_languages=primary_languages,
            file_structure=file_structure,
            commit_frequency=commit_frequency,
            risk_patterns={}
        )
    
    def _index_commits(self, git_analyzer, repo_id: str) -> List[CommitContext]:
        """Index commits for the repository"""
        commits = []
        repo = git_analyzer.repo
        
        for commit in repo.iter_commits(max_count=1000):  # Limit for performance
            try:
                # Get commit diff info
                files_changed = []
                additions = 0
                deletions = 0
                
                if commit.parents:
                    parent = commit.parents[0]
                    diff = parent.diff(commit)
                    
                    for diff_item in diff:
                        if diff_item.a_path:
                            files_changed.append(diff_item.a_path)
                    
                    # Count line changes
                    for diff_item in diff:
                        if diff_item.diff:
                            diff_text = diff_item.diff.decode('utf-8', errors='ignore')
                            additions += len([line for line in diff_text.split('\n') if line.startswith('+')])
                            deletions += len([line for line in diff_text.split('\n') if line.startswith('-')])
                
                # Create commit context
                commit_text = f"{commit.message}\n{' '.join(files_changed)}"
                embedding = self.embedding_engine.get_embedding(commit_text)
                
                commit_context = CommitContext(
                    commit_hash=commit.hexsha,
                    message=commit.message.strip(),
                    author=commit.author.name,
                    timestamp=commit.committed_datetime.isoformat(),
                    files_changed=files_changed,
                    additions=additions,
                    deletions=deletions,
                    breaking_changes=[],  # Will be populated by breaking change analysis
                    risk_score=0.0,  # Will be calculated
                    semantic_embedding=embedding
                )
                
                commits.append(commit_context)
                
            except Exception as e:
                logger.warning(f"Error indexing commit {commit.hexsha}: {str(e)}")
                continue
        
        return commits
    
    def _index_files(self, git_analyzer, repo_id: str) -> List[FileContext]:
        """Index files for the repository"""
        files = []
        repo = git_analyzer.repo
        
        for item in repo.head.commit.tree.traverse():
            if item.type == 'blob':
                try:
                    # Get file content
                    content = item.data_stream.read().decode('utf-8', errors='ignore')
                    
                    # Extract metadata
                    language = self._detect_language(item.path)
                    functions, classes = self._extract_code_elements(content, language)
                    dependencies = self._extract_dependencies(content, language)
                    
                    # Get file history
                    commits_for_file = list(repo.iter_commits(paths=item.path, max_count=100))
                    modification_frequency = len(commits_for_file)
                    contributors = list(set(commit.author.name for commit in commits_for_file))
                    
                    # Generate embedding
                    file_summary = f"{item.path} {language} {' '.join(functions)} {' '.join(classes)}"
                    embedding = self.embedding_engine.get_embedding(file_summary)
                    
                    file_context = FileContext(
                        file_path=item.path,
                        language=language,
                        size=item.size,
                        last_modified=commits_for_file[0].committed_datetime.isoformat() if commits_for_file else "",
                        modification_frequency=modification_frequency,
                        main_contributors=contributors[:3],  # Top 3 contributors
                        functions=functions,
                        classes=classes,
                        dependencies=dependencies,
                        semantic_embedding=embedding
                    )
                    
                    files.append(file_context)
                    
                except Exception as e:
                    logger.warning(f"Error indexing file {item.path}: {str(e)}")
                    continue
        
        return files
    
    def _analyze_commit_frequency(self, commits) -> Dict[str, int]:
        """Analyze commit frequency patterns"""
        from collections import defaultdict
        
        daily_commits = defaultdict(int)
        weekly_commits = defaultdict(int)
        monthly_commits = defaultdict(int)
        
        for commit in commits:
            date = commit.committed_datetime.date()
            daily_commits[str(date)] += 1
            
            # Get week number
            week = date.isocalendar()[1]
            weekly_commits[f"{date.year}-W{week}"] += 1
            
            # Get month
            monthly_commits[f"{date.year}-{date.month:02d}"] += 1
        
        return {
            "daily_average": len(commits) / max(len(daily_commits), 1),
            "weekly_average": len(commits) / max(len(weekly_commits), 1),
            "monthly_average": len(commits) / max(len(monthly_commits), 1),
            "total_active_days": len(daily_commits),
            "most_active_day": max(daily_commits.items(), key=lambda x: x[1])[0] if daily_commits else None
        }
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        ext_to_lang = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.go': 'go',
            '.rs': 'rust',
            '.cpp': 'cpp',
            '.c': 'c',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.clj': 'clojure',
            '.hs': 'haskell',
            '.ml': 'ocaml',
            '.fs': 'fsharp',
            '.vb': 'vbnet',
            '.cs': 'csharp',
            '.pl': 'perl',
            '.sh': 'bash',
            '.ps1': 'powershell',
            '.r': 'r',
            '.m': 'matlab',
            '.lua': 'lua',
            '.dart': 'dart',
            '.elm': 'elm',
            '.ex': 'elixir',
            '.erl': 'erlang',
        }
        
        ext = os.path.splitext(file_path)[1].lower()
        return ext_to_lang.get(ext, 'unknown')
    
    def _extract_code_elements(self, content: str, language: str) -> Tuple[List[str], List[str]]:
        """Extract functions and classes from code"""
        functions = []
        classes = []
        
        if language == 'python':
            import re
            functions = re.findall(r'def\s+(\w+)\s*\(', content)
            classes = re.findall(r'class\s+(\w+)\s*[:\(]', content)
        elif language in ['javascript', 'typescript']:
            import re
            functions.extend(re.findall(r'function\s+(\w+)\s*\(', content))
            functions.extend(re.findall(r'(\w+)\s*:\s*function', content))
            functions.extend(re.findall(r'(\w+)\s*=\s*\([^)]*\)\s*=>', content))
            classes = re.findall(r'class\s+(\w+)', content)
        elif language == 'java':
            import re
            functions = re.findall(r'(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\(', content)
            classes = re.findall(r'class\s+(\w+)', content)
        
        return functions[:50], classes[:20]  # Limit results
    
    def _extract_dependencies(self, content: str, language: str) -> List[str]:
        """Extract dependencies from code"""
        dependencies = []
        
        if language == 'python':
            import re
            imports = re.findall(r'from\s+(\S+)\s+import', content)
            imports.extend(re.findall(r'import\s+(\S+)', content))
            dependencies.extend(imports)
        elif language in ['javascript', 'typescript']:
            import re
            requires = re.findall(r'require\s*\(\s*[\'"]([^\'"]+)[\'"]', content)
            imports = re.findall(r'from\s+[\'"]([^\'"]+)[\'"]', content)
            dependencies.extend(requires + imports)
        elif language == 'java':
            import re
            imports = re.findall(r'import\s+([^;]+);', content)
            dependencies.extend(imports)
        
        return dependencies[:20]  # Limit results
    
    def _find_relevant_context(self, repo_id: str, query_embedding: np.ndarray, max_results: int) -> List[Dict[str, Any]]:
        """Find relevant context for query"""
        relevant_contexts = []
        
        # Search commits
        commits = self.commits.get(repo_id, [])
        for commit in commits:
            if commit.semantic_embedding is not None:
                similarity = np.dot(query_embedding, commit.semantic_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(commit.semantic_embedding)
                )
                relevant_contexts.append({
                    'type': 'commit',
                    'similarity': similarity,
                    'data': asdict(commit)
                })
        
        # Search files
        files = self.files.get(repo_id, [])
        for file_ctx in files:
            if file_ctx.semantic_embedding is not None:
                similarity = np.dot(query_embedding, file_ctx.semantic_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(file_ctx.semantic_embedding)
                )
                relevant_contexts.append({
                    'type': 'file',
                    'similarity': similarity,
                    'data': asdict(file_ctx)
                })
        
        # Sort by similarity and return top results
        relevant_contexts.sort(key=lambda x: x['similarity'], reverse=True)
        return relevant_contexts[:max_results]
    
    def _generate_response(self, query: str, contexts: List[Dict[str, Any]], repo_context: RepositoryContext) -> QueryResult:
        """Generate response based on query and context"""
        # This is a simplified response generation
        # In a full implementation, you would use Claude API here
        
        if not contexts:
            return QueryResult(
                query=query,
                response="No relevant information found for your query.",
                confidence=0.0,
                sources=[],
                context_used=[],
                suggestions=["Try rephrasing your question", "Check if the repository has been properly indexed"]
            )
        
        # Build response based on top contexts
        response_parts = []
        sources = []
        context_used = []
        
        for ctx in contexts[:3]:  # Use top 3 contexts
            if ctx['type'] == 'commit':
                commit_data = ctx['data']
                response_parts.append(f"Commit {commit_data['commit_hash'][:8]}: {commit_data['message']}")
                sources.append({
                    'type': 'commit',
                    'hash': commit_data['commit_hash'],
                    'message': commit_data['message']
                })
                context_used.append(f"commit:{commit_data['commit_hash'][:8]}")
            elif ctx['type'] == 'file':
                file_data = ctx['data']
                response_parts.append(f"File {file_data['file_path']}: {file_data['language']} file with {len(file_data['functions'])} functions")
                sources.append({
                    'type': 'file',
                    'path': file_data['file_path'],
                    'language': file_data['language']
                })
                context_used.append(f"file:{file_data['file_path']}")
        
        response = "Based on the repository analysis:\n\n" + "\n".join(response_parts)
        
        # Calculate confidence based on similarity scores
        confidence = np.mean([ctx['similarity'] for ctx in contexts[:3]]) if contexts else 0.0
        
        suggestions = [
            "Ask about specific files or functions",
            "Query about recent changes or commits",
            "Request breaking change analysis"
        ]
        
        return QueryResult(
            query=query,
            response=response,
            confidence=confidence,
            sources=sources,
            context_used=context_used,
            suggestions=suggestions
        )
    
    def _save_knowledge_base(self):
        """Save knowledge base to persistent storage"""
        try:
            data = {
                'repositories': {k: asdict(v) for k, v in self.repositories.items()},
                'commits': {k: [asdict(c) for c in v] for k, v in self.commits.items()},
                'files': {k: [asdict(f) for f in v] for k, v in self.files.items()},
            }
            
            with open(os.path.join(self.storage_path, 'knowledge_base.json'), 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
            # Save embeddings separately
            with open(os.path.join(self.storage_path, 'embeddings.pkl'), 'wb') as f:
                pickle.dump(self.embeddings, f)
                
        except Exception as e:
            logger.error(f"Error saving knowledge base: {str(e)}")
    
    def _load_knowledge_base(self):
        """Load knowledge base from persistent storage"""
        try:
            kb_path = os.path.join(self.storage_path, 'knowledge_base.json')
            if os.path.exists(kb_path):
                with open(kb_path, 'r') as f:
                    data = json.load(f)
                
                # Load repositories
                for repo_id, repo_data in data.get('repositories', {}).items():
                    self.repositories[repo_id] = RepositoryContext(**repo_data)
                
                # Load commits
                for repo_id, commits_data in data.get('commits', {}).items():
                    self.commits[repo_id] = [CommitContext(**c) for c in commits_data]
                
                # Load files
                for repo_id, files_data in data.get('files', {}).items():
                    self.files[repo_id] = [FileContext(**f) for f in files_data]
            
            # Load embeddings
            emb_path = os.path.join(self.storage_path, 'embeddings.pkl')
            if os.path.exists(emb_path):
                with open(emb_path, 'rb') as f:
                    self.embeddings = pickle.load(f)
                    
        except Exception as e:
            logger.error(f"Error loading knowledge base: {str(e)}")
            # Initialize empty if loading fails
            self.repositories = {}
            self.commits = {}
            self.files = {}
            self.embeddings = {}
    
    def index_repository(self, git_analyzer, repo_id: str, repo_url: str):
        """Index repository for RAG (stub implementation)"""
        self.repositories[repo_id] = {
            "url": repo_url,
            "indexed": True,
            "git_analyzer": git_analyzer
        }
        logger.info(f"Indexed repository {repo_id} (stub)")
    
    def query(self, repo_id: str, query: str, max_results: int = 10) -> QueryResult:
        """Query repository knowledge (stub implementation)"""
        return QueryResult(
            query=query,
            response="RAG system not fully implemented yet.",
            confidence=0.0,
            sources=[],
            context_used=[],
            suggestions=["Implement full RAG system"]
        )
    
    def get_repository_summary(self, repo_id: str) -> Dict[str, Any]:
        """Get repository summary (stub implementation)"""
        return {
            "repo_id": repo_id,
            "status": "stub",
            "recent_activity": []
        }
    
    def search_commits(self, repo_id: str, query: str, max_results: int) -> List[Dict]:
        """Search commits (stub implementation)"""
        return []
    
    def search_files(self, repo_id: str, query: str, max_results: int) -> List[Dict]:
        """Search files (stub implementation)"""
        return []

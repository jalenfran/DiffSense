import git
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib

@dataclass
class DiffInfo:
    """Structured representation of a git diff"""
    commit_hash: str
    commit_message: str
    commit_date: datetime
    author: str
    file_path: str
    added_lines: List[str]
    removed_lines: List[str]
    change_type: str
    lines_added: int
    lines_removed: int

class GitAnalyzer:
    """Extract and analyze git repository history"""
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.repo = git.Repo(repo_path)
    
    @classmethod
    def clone_repository(cls, repo_url: str, local_path: str) -> 'GitAnalyzer':
        """Clone a repository and return GitAnalyzer instance"""
        try:
            print(f"Cloning {repo_url} to {local_path}")
            git.Repo.clone_from(repo_url, local_path)
            print(f"Successfully cloned repository")
            return cls(local_path)
        except Exception as e:
            print(f"Failed to clone repository: {e}")
            raise
    
    def get_commits_for_file(self, file_path: str, max_commits: int = 100) -> List[git.Commit]:
        """Get commits that modified a specific file"""
        try:
            commits = list(self.repo.iter_commits(paths=file_path, max_count=max_commits))
            return commits
        except Exception as e:
            print(f"Error getting commits for {file_path}: {e}")
            return []
    
    def get_commits_for_function(self, file_path: str, function_name: str, max_commits: int = 50) -> List[git.Commit]:
        """Get commits that modified a specific function (simplified approach)"""
        # This is a simplified version - in practice, you'd want AST parsing
        commits = []
        for commit in self.repo.iter_commits(paths=file_path, max_count=max_commits):
            try:
                # Check if function name appears in the diff
                diffs = commit.diff(commit.parents[0] if commit.parents else None)
                for diff in diffs:
                    if diff.b_path == file_path and diff.diff:
                        diff_text = diff.diff.decode('utf-8', errors='ignore')
                        if function_name in diff_text:
                            commits.append(commit)
                            break
            except Exception as e:
                continue
        return commits
    
    def extract_diff_info(self, commit: git.Commit, file_path: Optional[str] = None) -> List[DiffInfo]:
        """Extract structured diff information from a commit"""
        diff_infos = []
        
        try:
            # Compare with parent commit
            parent = commit.parents[0] if commit.parents else None
            diffs = commit.diff(parent)
            
            for diff in diffs:
                # Filter by file if specified
                if file_path and diff.b_path != file_path:
                    continue
                
                # Skip binary files
                if diff.diff is None:
                    continue
                
                try:
                    # Handle both bytes and string diff content
                    if isinstance(diff.diff, bytes):
                        diff_text = diff.diff.decode('utf-8', errors='ignore')
                    else:
                        diff_text = str(diff.diff)
                    added_lines, removed_lines = self._parse_diff_lines(diff_text)
                    
                    diff_info = DiffInfo(
                        commit_hash=commit.hexsha,
                        commit_message=commit.message.strip(),
                        commit_date=datetime.fromtimestamp(commit.committed_date),
                        author=commit.author.name,
                        file_path=diff.b_path or diff.a_path,
                        added_lines=added_lines,
                        removed_lines=removed_lines,
                        change_type=diff.change_type,
                        lines_added=len(added_lines),
                        lines_removed=len(removed_lines)
                    )
                    diff_infos.append(diff_info)
                    
                except Exception as e:
                    print(f"Error processing diff: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error extracting diff info from commit {commit.hexsha}: {e}")
        
        return diff_infos
    
    def _parse_diff_lines(self, diff_text: str) -> tuple[List[str], List[str]]:
        """Parse diff text to extract added and removed lines"""
        added_lines = []
        removed_lines = []
        
        for line in diff_text.split('\n'):
            if line.startswith('+') and not line.startswith('+++'):
                added_lines.append(line[1:])  # Remove the + prefix
            elif line.startswith('-') and not line.startswith('---'):
                removed_lines.append(line[1:])  # Remove the - prefix
        
        return added_lines, removed_lines
    
    def get_file_content_at_commit(self, commit_hash: str, file_path: str) -> Optional[str]:
        """Get the content of a file at a specific commit"""
        try:
            commit = self.repo.commit(commit_hash)
            blob = commit.tree / file_path
            return blob.data_stream.read().decode('utf-8', errors='ignore')
        except Exception as e:
            print(f"Error getting file content: {e}")
            return None
    
    def get_function_content_at_commit(self, commit_hash: str, file_path: str, function_name: str) -> Optional[str]:
        """Extract function content at a specific commit (simplified)"""
        content = self.get_file_content_at_commit(commit_hash, file_path)
        if not content:
            return None
        
        # Simple function extraction - in practice, use AST parsing
        lines = content.split('\n')
        function_lines = []
        in_function = False
        brace_count = 0
        
        for line in lines:
            if function_name in line and ('def ' in line or 'function ' in line or 'func ' in line):
                in_function = True
                function_lines.append(line)
                brace_count += line.count('{') - line.count('}')
            elif in_function:
                function_lines.append(line)
                brace_count += line.count('{') - line.count('}')
                if brace_count <= 0 and line.strip():
                    break
        
        return '\n'.join(function_lines) if function_lines else None
    
    def analyze_repository_stats(self) -> Dict[str, Any]:
        """Get basic repository statistics"""
        commits = list(self.repo.iter_commits())
        
        return {
            'total_commits': len(commits),
            'contributors': len(set(commit.author.name for commit in commits)),
            'date_range': {
                'start': datetime.fromtimestamp(commits[-1].committed_date).isoformat() if commits else None,
                'end': datetime.fromtimestamp(commits[0].committed_date).isoformat() if commits else None
            },
            'active_branches': len(list(self.repo.branches)),
            'file_count': len(list(self.repo.tree().traverse()))
        }

def example_usage():
    """Example of how to use GitAnalyzer"""
    # Clone and analyze a repository
    analyzer = GitAnalyzer.clone_repository(
        "https://github.com/microsoft/vscode.git", 
        "/tmp/vscode_repo"
    )
    
    # Get repository stats
    stats = analyzer.analyze_repository_stats()
    print(f"Repository stats: {stats}")
    
    # Analyze a specific file
    commits = analyzer.get_commits_for_file("src/vs/editor/editor.api.ts")
    print(f"Found {len(commits)} commits for editor API")
    
    # Extract diff information
    if commits:
        diff_infos = analyzer.extract_diff_info(commits[0])
        for diff_info in diff_infos:
            print(f"Commit: {diff_info.commit_hash[:8]} - {diff_info.commit_message[:50]}...")

if __name__ == "__main__":
    example_usage()

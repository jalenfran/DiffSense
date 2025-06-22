"""
GitHub Service - Handle OAuth and private repository access
"""

import os
import requests
import secrets
import base64
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class GitHubUser:
    """GitHub user information"""
    id: str
    username: str
    email: Optional[str]
    name: str
    avatar_url: str
    
@dataclass
class GitHubRepository:
    """GitHub repository information"""
    id: str
    name: str
    full_name: str
    clone_url: str
    ssh_url: str
    private: bool
    description: Optional[str]
    language: Optional[str]
    default_branch: str

class GitHubService:
    """GitHub OAuth and API service"""
    
    def __init__(self):
        self.client_id = os.getenv('GITHUB_CLIENT_ID')
        self.client_secret = os.getenv('GITHUB_CLIENT_SECRET')
        self.redirect_uri = os.getenv('GITHUB_REDIRECT_URI', 'http://76.125.217.28:8080/api/auth/github/callback')
        self.frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:5173')
        
        if not self.client_id or not self.client_secret:
            logger.warning("GitHub OAuth credentials not configured")
            self.available = False
        else:
            self.available = True
            logger.info("GitHub service initialized successfully")
    
    def get_oauth_url(self, state: Optional[str] = None) -> str:
        """Generate GitHub OAuth authorization URL"""
        if not state:
            state = secrets.token_urlsafe(32)
        
        scopes = 'read:user,user:email,repo'  # repo scope for private repositories
        
        oauth_url = (
            f"https://github.com/login/oauth/authorize?"
            f"client_id={self.client_id}&"
            f"redirect_uri={self.redirect_uri}&"
            f"scope={scopes}&"
            f"state={state}"
        )
        
        return oauth_url, state
    
    def exchange_code_for_token(self, code: str, state: str) -> Optional[str]:
        """Exchange OAuth code for access token"""
        try:
            data = {
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'code': code,
                'redirect_uri': self.redirect_uri,
                'state': state
            }
            
            headers = {'Accept': 'application/json'}
            
            response = requests.post(
                'https://github.com/login/oauth/access_token',
                data=data,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                access_token = result.get('access_token')
                
                if access_token:
                    logger.info("Successfully exchanged OAuth code for token")
                    return access_token
                else:
                    logger.error(f"No access token in response: {result}")
                    return None
            else:
                logger.error(f"Failed to exchange code: {response.status_code} {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error exchanging OAuth code: {e}")
            return None
    
    def get_user_info(self, access_token: str) -> Optional[GitHubUser]:
        """Get GitHub user information"""
        try:
            headers = {
                'Authorization': f'token {access_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            # Get user info
            user_response = requests.get(
                'https://api.github.com/user',
                headers=headers,
                timeout=30
            )
            
            if user_response.status_code != 200:
                logger.error(f"Failed to get user info: {user_response.status_code}")
                return None
            
            user_data = user_response.json()
            
            # Get user emails
            email_response = requests.get(
                'https://api.github.com/user/emails',
                headers=headers,
                timeout=30
            )
            
            primary_email = None
            if email_response.status_code == 200:
                emails = email_response.json()
                for email in emails:
                    if email.get('primary'):
                        primary_email = email.get('email')
                        break
            
            return GitHubUser(
                id=str(user_data['id']),
                username=user_data['login'],
                email=primary_email,
                name=user_data.get('name') or user_data['login'],
                avatar_url=user_data['avatar_url']
            )
            
        except Exception as e:
            logger.error(f"Error getting user info: {e}")
            return None
    
    def get_user_repositories(self, access_token: str, per_page: int = 50) -> List[GitHubRepository]:
        """Get user's repositories (including private ones)"""
        try:
            headers = {
                'Authorization': f'token {access_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            repositories = []
            page = 1
            
            while True:
                response = requests.get(
                    'https://api.github.com/user/repos',
                    headers=headers,
                    params={
                        'per_page': per_page,
                        'page': page,
                        'sort': 'updated',
                        'type': 'all'  # Include all repositories (owner, collaborator, etc.)
                    },
                    timeout=30
                )
                
                if response.status_code != 200:
                    logger.error(f"Failed to get repositories: {response.status_code}")
                    break
                
                repos_data = response.json()
                if not repos_data:
                    break
                
                for repo_data in repos_data:
                    repositories.append(GitHubRepository(
                        id=str(repo_data['id']),
                        name=repo_data['name'],
                        full_name=repo_data['full_name'],
                        clone_url=repo_data['clone_url'],
                        ssh_url=repo_data['ssh_url'],
                        private=repo_data['private'],
                        description=repo_data.get('description'),
                        language=repo_data.get('language'),
                        default_branch=repo_data.get('default_branch', 'main')
                    ))
                
                page += 1
                
                # GitHub API returns up to 100 pages
                if page > 100:
                    break
            
            logger.info(f"Retrieved {len(repositories)} repositories for user")
            return repositories
            
        except Exception as e:
            logger.error(f"Error getting user repositories: {e}")
            return []
    
    def get_repository_info(self, access_token: str, owner: str, repo: str) -> Optional[GitHubRepository]:
        """Get specific repository information"""
        try:
            headers = {
                'Authorization': f'token {access_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            response = requests.get(
                f'https://api.github.com/repos/{owner}/{repo}',
                headers=headers,
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to get repository info: {response.status_code}")
                return None
            
            repo_data = response.json()
            
            return GitHubRepository(
                id=str(repo_data['id']),
                name=repo_data['name'],
                full_name=repo_data['full_name'],
                clone_url=repo_data['clone_url'],
                ssh_url=repo_data['ssh_url'],
                private=repo_data['private'],
                description=repo_data.get('description'),
                language=repo_data.get('language'),
                default_branch=repo_data.get('default_branch', 'main')
            )
            
        except Exception as e:
            logger.error(f"Error getting repository info: {e}")
            return None
    
    def create_authenticated_clone_url(self, access_token: str, clone_url: str) -> str:
        """Create an authenticated clone URL for private repositories"""
        try:
            # For HTTPS clone URLs, inject the token
            if clone_url.startswith('https://github.com/'):
                # Convert https://github.com/owner/repo.git to https://token@github.com/owner/repo.git
                authenticated_url = clone_url.replace('https://github.com/', f'https://{access_token}@github.com/')
                return authenticated_url
            
            # For other URLs, return as-is
            return clone_url
            
        except Exception as e:
            logger.error(f"Error creating authenticated clone URL: {e}")
            return clone_url
    
    def validate_repository_access(self, access_token: str, repo_url: str) -> bool:
        """Validate that the user has access to a repository"""
        try:
            # Extract owner and repo from URL
            if 'github.com/' in repo_url:
                path_part = repo_url.split('github.com/')[-1]
                if path_part.endswith('.git'):
                    path_part = path_part[:-4]
                
                if '/' in path_part:
                    owner, repo = path_part.split('/', 1)
                    
                    # Try to get repository info
                    repo_info = self.get_repository_info(access_token, owner, repo)
                    return repo_info is not None
            
            return False
            
        except Exception as e:
            logger.error(f"Error validating repository access: {e}")
            return False
    
    def encrypt_token(self, token: str) -> str:
        """Simple base64 encoding for token storage (use proper encryption in production)"""
        try:
            encoded = base64.b64encode(token.encode()).decode()
            return encoded
        except Exception as e:
            logger.error(f"Error encoding token: {e}")
            return token
    
    def decrypt_token(self, encoded_token: str) -> str:
        """Simple base64 decoding for token retrieval"""
        try:
            decoded = base64.b64decode(encoded_token.encode()).decode()
            return decoded
        except Exception as e:
            logger.error(f"Error decoding token: {e}")
            return encoded_token
    
    def get_status(self) -> Dict[str, Any]:
        """Get GitHub service status"""
        return {
            "available": self.available,
            "client_id_configured": bool(self.client_id),
            "client_secret_configured": bool(self.client_secret),
            "redirect_uri": self.redirect_uri
        }

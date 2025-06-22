
import requests
import json
import time

API_URL = "http://localhost:8080"
REPO_URL = "https://github.com/SadeekFarhan21/QuantifyAI"

def test_clone_repository():
    """Test cloning a repository"""
    print("Testing POST /api/clone-repository")
    payload = {"repo_url": REPO_URL, "max_commits": 10}
    response = requests.post(f"{API_URL}/api/clone-repository", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Clone successful:", data)
        return data.get("repo_id")
    else:
        print("❌ Clone failed:", response.status_code, response.text)
        return None

def test_analyze_commit(repo_id):
    """Test analyzing a commit"""
    if not repo_id:
        return
        
    print(f"\nTesting POST /api/analyze-commit/{repo_id}")
    
    # First, get a commit hash to analyze from the commits endpoint
    commits_response = requests.get(f"{API_URL}/api/repository/{repo_id}/commits?limit=10")
    if commits_response.status_code != 200:
        print("❌ Could not get repo commits to find a commit hash")
        return
        
    commits_data = commits_response.json()
    commits = commits_data.get("commits", [])
    if not commits:
        print("❌ No commits found in repository")
        return
        
    commit_hash = commits[0].get("hash")
    print(f"Testing with commit: {commit_hash[:8]}...")
    
    payload = {"commit_hash": commit_hash, "include_claude_analysis": False}
    response = requests.post(f"{API_URL}/api/analyze-commit/{repo_id}", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Commit analysis successful:")
        print(f"   Risk Score: {result.get('overall_risk_score', 'N/A')}")
        print(f"   Breaking Changes: {len(result.get('breaking_changes', []))}")
        print(f"   Files Changed: {len(result.get('files_changed', []))}")
    else:
        print("❌ Commit analysis failed:", response.status_code, response.text)

def main():
    """Run API tests"""
    repo_id = test_clone_repository()
    time.sleep(2) # Give server time to process
    test_analyze_commit(repo_id)

if __name__ == "__main__":
    main()

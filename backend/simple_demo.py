#!/usr/bin/env python3
"""
Simplified Demo script for DiffSense - Feature Drift Detector
This version demonstrates core git analysis without ML dependencies.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from git_analyzer import GitAnalyzer
import tempfile
import shutil
import hashlib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SimpleDriftDetector:
    """Simplified drift detector without ML models"""
    
    def __init__(self, git_analyzer):
        self.git_analyzer = git_analyzer
        self.drift_threshold = 0.3
    
    def simple_text_embedding(self, text):
        """Create a simple text embedding using character frequency"""
        # Simple character frequency vector (simplified embedding)
        chars = {}
        for char in text.lower():
            if char.isalnum():
                chars[char] = chars.get(char, 0) + 1
        
        # Create fixed-size vector (26 letters + 10 digits)
        vector = []
        for char in 'abcdefghijklmnopqrstuvwxyz0123456789':
            vector.append(chars.get(char, 0))
        
        # Normalize
        vector = np.array(vector, dtype=float)
        if np.sum(vector) > 0:
            vector = vector / np.sum(vector)
        
        return vector
    
    def analyze_file_drift(self, file_path, max_commits=10):
        """Simplified file drift analysis"""
        # Get commits
        commits = self.git_analyzer.get_commits_for_file(file_path, max_commits)
        
        if not commits:
            raise ValueError(f"No commits found for file: {file_path}")
        
        # Extract diff information
        all_diffs = []
        for commit in commits:
            diffs = self.git_analyzer.extract_diff_info(commit, file_path)
            all_diffs.extend(diffs)
        
        # Sort by date
        all_diffs.sort(key=lambda x: x.commit_date)
        
        # Create simple embeddings
        embeddings = []
        for diff in all_diffs:
            # Combine added and removed lines
            text = ' '.join(diff.added_lines + diff.removed_lines + [diff.commit_message])
            embedding = self.simple_text_embedding(text)
            embeddings.append(embedding)
        
        # Calculate drift scores
        drift_scores = []
        if len(embeddings) > 1:
            for i in range(1, len(embeddings)):
                similarity = cosine_similarity([embeddings[0]], [embeddings[i]])[0][0]
                drift_score = 1 - similarity
                drift_scores.append(drift_score)
        
        # Find significant changes
        significant_changes = []
        for i, (diff, score) in enumerate(zip(all_diffs[1:], drift_scores)):
            if score > self.drift_threshold:
                significant_changes.append({
                    'commit_hash': diff.commit_hash,
                    'drift_score': score,
                    'commit_message': diff.commit_message,
                    'added_lines': diff.lines_added,
                    'removed_lines': diff.lines_removed,
                    'timestamp': diff.commit_date
                })
        
        overall_drift = max(drift_scores) if drift_scores else 0
        
        return {
            'file_path': file_path,
            'total_commits': len(all_diffs),
            'overall_drift': overall_drift,
            'drift_scores': drift_scores,
            'significant_changes': significant_changes,
            'all_diffs': all_diffs
        }

def create_sample_repo():
    """Create a sample git repository with some code changes for demo purposes"""
    print("🔧 Creating sample repository for demo...")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="diffsense_demo_")
    original_dir = os.getcwd()
    os.chdir(temp_dir)
    
    try:
        # Initialize git repo
        os.system("git init")
        os.system("git config user.email 'demo@diffsense.com'")
        os.system("git config user.name 'DiffSense Demo'")
        
        # Create initial file
        with open("calculator.py", "w") as f:
            f.write("""def add(a, b):
    return a + b

def multiply(a, b):
    return a * b
""")
        
        os.system("git add calculator.py")
        os.system("git commit -m 'Initial calculator implementation'")
        
        # Modify file - add validation
        with open("calculator.py", "w") as f:
            f.write("""def add(a, b):
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Arguments must be numbers")
    return a + b

def multiply(a, b):
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Arguments must be numbers")
    return a * b
""")
        
        os.system("git add calculator.py")
        os.system("git commit -m 'Add input validation to calculator functions'")
        
        # Major refactor - change API
        with open("calculator.py", "w") as f:
            f.write("""class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise TypeError("Arguments must be numbers")
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def multiply(self, a, b):
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise TypeError("Arguments must be numbers")
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result
    
    def get_history(self):
        return self.history.copy()
""")
        
        os.system("git add calculator.py")
        os.system("git commit -m 'Refactor to class-based calculator with history tracking'")
        
        # Add new feature
        with open("calculator.py", "w") as f:
            f.write("""class Calculator:
    def __init__(self):
        self.history = []
        self.precision = 2
    
    def add(self, a, b):
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise TypeError("Arguments must be numbers")
        result = round(a + b, self.precision)
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def multiply(self, a, b):
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise TypeError("Arguments must be numbers")
        result = round(a * b, self.precision)
        self.history.append(f"{a} * {b} = {result}")
        return result
    
    def divide(self, a, b):
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise TypeError("Arguments must be numbers")
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = round(a / b, self.precision)
        self.history.append(f"{a} / {b} = {result}")
        return result
    
    def get_history(self):
        return self.history.copy()
    
    def clear_history(self):
        self.history.clear()
        
    def set_precision(self, precision):
        self.precision = max(0, int(precision))
""")
        
        os.system("git add calculator.py")
        os.system("git commit -m 'Add division, precision control, and history management'")
        
        print(f"✅ Sample repository created at: {temp_dir}")
        return temp_dir
        
    except Exception as e:
        print(f"Error creating sample repo: {e}")
        os.chdir(original_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    finally:
        os.chdir(original_dir)

def demo_drift_analysis():
    """Demonstrate drift analysis functionality"""
    print("\n🚀 Starting DiffSense Demo (Simplified Version)")
    print("=" * 60)
    
    # Create sample repository
    repo_path = create_sample_repo()
    
    try:
        # Initialize analyzer
        print("\n📊 Initializing Git Analyzer...")
        git_analyzer = GitAnalyzer(repo_path)
        detector = SimpleDriftDetector(git_analyzer)
        
        # Get repository stats
        stats = git_analyzer.analyze_repository_stats()
        print(f"\n📈 Repository Statistics:")
        print(f"   Total Commits: {stats['total_commits']}")
        print(f"   Contributors: {stats['contributors']}")
        print(f"   Files: {stats['file_count']}")
        
        # Analyze file drift
        print("\n🔍 Analyzing semantic drift for calculator.py...")
        analysis = detector.analyze_file_drift("calculator.py")
        
        print(f"\n📈 Analysis Results:")
        print(f"   Overall Drift Score: {analysis['overall_drift']:.3f}")
        print(f"   Total Commits: {analysis['total_commits']}")
        print(f"   Significant Changes: {len(analysis['significant_changes'])}")
        
        # Show drift progression
        print(f"\n📊 Drift Progression:")
        for i, (diff, score) in enumerate(zip(analysis['all_diffs'][1:], analysis['drift_scores'])):
            status = "🔴" if score > 0.3 else "🟡" if score > 0.1 else "🟢"
            print(f"   {status} Commit {i+2}: {score:.3f} - {diff.commit_message}")
        
        # Show significant events
        if analysis['significant_changes']:
            print(f"\n⚠️  Significant Drift Events:")
            for i, event in enumerate(analysis['significant_changes'], 1):
                print(f"   {i}. {event['commit_hash'][:8]} - Drift: {event['drift_score']:.3f}")
                print(f"      Message: {event['commit_message']}")
                print(f"      Changes: +{event['added_lines']} -{event['removed_lines']} lines")
                print()
        
        # Risk assessment (simplified)
        max_drift = analysis['overall_drift']
        if max_drift > 0.7:
            risk_level = "HIGH"
            risk_emoji = "🔴"
        elif max_drift > 0.3:
            risk_level = "MEDIUM"
            risk_emoji = "🟡"
        else:
            risk_level = "LOW"
            risk_emoji = "🟢"
        
        print(f"\n🎯 Breaking Change Risk Assessment:")
        print(f"   {risk_emoji} Risk Level: {risk_level}")
        print(f"   📊 Max Drift Score: {max_drift:.3f}")
        print(f"   💭 Analysis: {'High semantic drift detected - review recommended' if max_drift > 0.5 else 'Moderate changes detected' if max_drift > 0.2 else 'Low risk - stable evolution'}")
        
        print(f"\n✅ Demo completed successfully!")
        print(f"💡 This simplified demo shows how DiffSense can track semantic")
        print(f"   evolution and identify potentially breaking changes.")
        print(f"🔬 The full version uses advanced ML models (CodeBERT, etc.)")
        print(f"   for more accurate semantic understanding.")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        print(f"\n🧹 Cleaning up temporary files...")
        shutil.rmtree(repo_path, ignore_errors=True)

if __name__ == "__main__":
    demo_drift_analysis()

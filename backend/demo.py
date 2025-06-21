#!/usr/bin/env python3
"""
Demo script for DiffSense - Feature Drift Detector
This script demonstrates the core functionality without requiring a full repository clone.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from git_analyzer import GitAnalyzer
from drift_detector import DriftDetector
import tempfile
import shutil

def create_sample_repo():
    """Create a sample git repository with some code changes for demo purposes"""
    print("🔧 Creating sample repository for demo...")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="diffsense_demo_")
    os.chdir(temp_dir)
    
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

def demo_drift_analysis():
    """Demonstrate drift analysis functionality"""
    print("\n🚀 Starting DiffSense Demo")
    print("=" * 50)
    
    # Create sample repository
    repo_path = create_sample_repo()
    
    try:
        # Initialize analyzer
        print("\n📊 Initializing Git Analyzer...")
        git_analyzer = GitAnalyzer(repo_path)
        detector = DriftDetector(git_analyzer)
        
        # Analyze file drift
        print("\n🔍 Analyzing semantic drift for calculator.py...")
        feature_history = detector.analyze_file_drift("calculator.py")
        
        print(f"\n📈 Analysis Results:")
        print(f"   Overall Drift Score: {feature_history.overall_drift:.3f}")
        print(f"   Total Commits: {len(feature_history.commits)}")
        print(f"   Significant Drift Events: {len(feature_history.drift_events)}")
        
        # Show drift events
        if feature_history.drift_events:
            print(f"\n⚠️  Significant Drift Events:")
            for i, event in enumerate(feature_history.drift_events, 1):
                print(f"   {i}. {event.commit_hash[:8]} - {event.drift_score:.3f}")
                print(f"      {event.commit_message}")
                print(f"      {event.description}")
                print()
        
        # Generate timeline
        print("\n📅 Drift Timeline:")
        timeline = detector.generate_drift_timeline(feature_history)
        for point in timeline:
            status = "🔴" if point['is_significant'] else "🟢"
            print(f"   {status} {point['commit_hash'][:8]} - Drift: {point['drift_score']:.3f}")
            print(f"      {point['commit_message']}")
        
        # Risk assessment
        print(f"\n🎯 Breaking Change Risk Assessment:")
        risk = detector.predict_breaking_change_risk(feature_history)
        risk_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        print(f"   {risk_emoji[risk['risk_level']]} Risk Level: {risk['risk_level'].upper()}")
        print(f"   📊 Risk Score: {risk['risk_score']:.1%}")
        print(f"   💭 Reasoning: {risk['reasoning']}")
        
        # Change patterns
        if feature_history.change_summary:
            print(f"\n📊 Change Pattern Analysis:")
            summary = feature_history.change_summary
            print(f"   Average Similarity: {summary.get('average_similarity', 0):.3f}")
            print(f"   Maximum Drift: {summary.get('max_drift', 0):.3f}")
            print(f"   Drift Trend: {summary.get('drift_trend', 'unknown').title()}")
        
        print(f"\n✅ Demo completed successfully!")
        print(f"💡 This demonstrates how DiffSense can track semantic evolution")
        print(f"   and identify potentially breaking changes in your codebase.")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        
    finally:
        # Cleanup
        print(f"\n🧹 Cleaning up temporary files...")
        shutil.rmtree(repo_path, ignore_errors=True)

if __name__ == "__main__":
    demo_drift_analysis()

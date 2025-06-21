#!/bin/bash

# DiffSense Validation Script
# Quick test to ensure everything is working

echo "🔍 DiffSense - Validation Test"
echo "=============================="

# Test 1: Check files exist
echo "📁 Checking project structure..."
REQUIRED_FILES=(
    "README.md"
    "backend/main.py"
    "backend/simple_demo.py"
    "backend/src/git_analyzer.py"
    "frontend/package.json"
    "frontend/src/App.jsx"
    "setup.sh"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ Missing: $file"
        exit 1
    fi
done

# Test 2: Check Python dependencies
echo ""
echo "🐍 Testing Python environment..."
cd backend

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Install minimal deps
pip install -q gitpython numpy scikit-learn 2>/dev/null || {
    echo "❌ Failed to install Python dependencies"
    exit 1
}

# Test import
python3 -c "
import sys
sys.path.append('src')
try:
    from git_analyzer import GitAnalyzer
    print('✅ Git analyzer import successful')
except Exception as e:
    print(f'❌ Import failed: {e}')
    exit(1)
" || exit 1

# Test 3: Quick functionality test
echo ""
echo "⚡ Testing core functionality..."
python3 -c "
import sys, tempfile, os
sys.path.append('src')
from git_analyzer import GitAnalyzer

# Test git analyzer basic functionality
try:
    temp_dir = tempfile.mkdtemp()
    os.chdir(temp_dir)
    os.system('git init >/dev/null 2>&1')
    os.system('git config user.email test@test.com >/dev/null 2>&1')
    os.system('git config user.name Test >/dev/null 2>&1')
    
    with open('test.py', 'w') as f:
        f.write('print(\"hello\")')
    
    os.system('git add test.py >/dev/null 2>&1')
    os.system('git commit -m \"test\" >/dev/null 2>&1')
    
    analyzer = GitAnalyzer('.')
    stats = analyzer.analyze_repository_stats()
    
    if stats['total_commits'] > 0:
        print('✅ Core functionality working')
    else:
        print('❌ Core functionality failed')
        exit(1)
        
except Exception as e:
    print(f'❌ Functionality test failed: {e}')
    exit(1)
finally:
    os.chdir('/')
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
" || exit 1

# Test 4: Check Node.js if available
echo ""
echo "📦 Checking frontend setup..."
if command -v node >/dev/null 2>&1; then
    echo "✅ Node.js available: $(node --version)"
    cd ../frontend
    if [ -f "package.json" ]; then
        echo "✅ Frontend package.json found"
    else
        echo "❌ Frontend package.json missing"
        exit 1
    fi
else
    echo "⚠️  Node.js not available (demo will still work)"
fi

echo ""
echo "🎉 Validation Complete!"
echo ""
echo "✅ All systems ready for demo"
echo "🚀 Run: ./setup.sh demo"
echo "🌐 Or:  ./setup.sh full"

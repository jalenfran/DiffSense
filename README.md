# DiffSense: Feature Drift Detector

> 🏆 **AI Berkeley Hackathon Project** - Semantic drift detection using embedding-powered analysis of git history

[![Demo](https://img.shields.io/badge/Demo-Ready-brightgreen)](./setup.sh)
[![Documentation](https://img.shields.io/badge/Docs-Complete-blue)](./TECHNICAL_ROADMAP.md)
[![Presentation](https://img.shields.io/badge/Presentation-Guide-purple)](./DEMO_GUIDE.md)

## 🚀 Quick Start for Judges

### **Instant Demo** (2 minutes)
```bash
./setup.sh demo
```
*Shows semantic drift analysis on a generated repository*

### **Full Web Application** (5 minutes)
```bash
./setup.sh full
# Open http://localhost:3000
```
*Complete interface for analyzing any GitHub repository*

### **Help & Options**
```bash
./setup.sh help
```

## 🎯 The Problem

Modern software development moves fast. Features change, APIs evolve, and sometimes small commits create huge unexpected impacts—whether breaking downstream functionality, causing subtle bugs, or violating intended product behavior. Teams lose track of why things were changed or when something started behaving differently.

## 💡 Solution: DiffSense

**DiffSense** solves this by using embedding-powered semantic drift detection over git diffs, commit messages, issue tickets, and changelogs.

→ You input a function, file, or API you want to audit  
→ The system retrieves its historical versions, compares the semantic meaning of changes over time via embeddings  
→ Generates clear, human-readable explanations of how and why that feature changed  

## 🚀 Quick Start

### Option 1: One-Command Start (Recommended)
```bash
./start.sh
```

### Option 2: Manual Setup

**Backend Setup:**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

**Frontend Setup:**
```bash
cd frontend
npm install
npm run dev
```

**Demo Script:**
```bash
cd backend
python demo.py
```

## 🏗️ Architecture

### Technology Stack
- **Backend**: Python FastAPI with ML pipeline
- **Frontend**: React with Recharts visualization
- **ML Models**: 
  - CodeBERT for code embeddings
  - sentence-transformers for text embeddings
  - Hybrid embedding approach
- **Git Analysis**: GitPython for repository parsing
- **API**: RESTful endpoints with real-time analysis

### Core Components
1. **Git Analyzer** (`git_analyzer.py`) - Extract and parse git history
2. **Embedding Engine** (`embedding_engine.py`) - Generate semantic embeddings
3. **Drift Detector** (`drift_detector.py`) - Analyze semantic changes over time
4. **FastAPI Backend** (`main.py`) - REST API for frontend integration
5. **React Frontend** - Interactive visualization and analysis interface

## ✨ Core Features

### 1. **Semantic Drift Detection**
- Track how code meaning changes over time using AI embeddings
- Identify gradual vs sudden semantic shifts
- Measure cumulative drift from original implementation

### 2. **Breaking Change Prediction**
- ML-powered risk scoring for commits
- Predict potentially risky changes before they impact users
- Historical pattern analysis for risk assessment

### 3. **Interactive Timeline Visualization**
- Visual drift timeline with commit details
- Identify significant change events
- Hover details with commit messages and metrics

### 4. **Multi-Level Analysis**
- **File-level**: Analyze entire file evolution
- **Function-level**: Track specific function changes
- **Repository-level**: Overall project health metrics

## 🎮 Demo Flow

1. **Repository Input**: Enter GitHub repository URL
2. **File Selection**: Choose file or function to analyze  
3. **Semantic Analysis**: AI processes git history and generates embeddings
4. **Drift Visualization**: Interactive timeline showing semantic changes
5. **Risk Assessment**: Breaking change prediction with explanations
6. **Export Results**: Summary reports and recommendations

## 📊 Use Cases

### For Development Teams
- **Detect undocumented breaking changes** before a release
- **Help new developers** quickly catch up on why parts of the codebase evolved
- **Trace regressions** back to their origins, even in noisy or badly documented projects

### For Project Managers
- **Risk assessment** for releases
- **Technical debt tracking** over time
- **API stability monitoring**

### For Open Source Maintainers
- **Contributor onboarding** with feature evolution stories
- **Impact analysis** for proposed changes
- **Documentation gap identification**

## 🛠️ Technical Implementation

### Embedding Strategy
```python
# Hybrid approach combining code and text semantics
code_embedding = CodeBERT.encode(code_diff)
text_embedding = SentenceTransformer.encode(commit_message)
hybrid_embedding = 0.7 * code_embedding + 0.3 * text_embedding
```

### Drift Calculation
```python
# Semantic similarity tracking over time
def calculate_drift(embeddings_timeline):
    drift_scores = []
    for i in range(1, len(embeddings_timeline)):
        similarity = cosine_similarity(embeddings_timeline[0], embeddings_timeline[i])
        drift_scores.append(1 - similarity)  # Higher = more drift
    return drift_scores
```

### Breaking Change Prediction
- **Feature Engineering**: Code metrics + semantic embeddings + commit metadata
- **Heuristic Model**: Risk scoring based on drift patterns and change magnitude
- **Contextual Analysis**: Related issues, commit message sentiment, file importance

## 📁 Project Structure

```
DiffSense/
├── README.md                 # This file
├── TECHNICAL_ROADMAP.md      # Detailed implementation guide
├── start.sh                  # One-command startup script
├── backend/
│   ├── main.py              # FastAPI server
│   ├── demo.py              # Standalone demo script
│   ├── requirements.txt     # Python dependencies
│   └── src/
│       ├── git_analyzer.py     # Git repository analysis
│       ├── embedding_engine.py # AI embedding generation
│       └── drift_detector.py   # Semantic drift detection
└── frontend/
    ├── package.json         # Node.js dependencies
    ├── vite.config.js       # Vite configuration
    ├── tailwind.config.js   # Tailwind CSS config
    └── src/
        ├── App.jsx              # Main React application
        └── components/
            ├── RepositoryCloner.jsx  # Repository input interface
            ├── DriftAnalyzer.jsx     # Main analysis interface
            ├── FileSelector.jsx      # File selection component
            ├── DriftSummary.jsx      # Analysis results summary
            └── DriftTimeline.jsx     # Interactive timeline chart
```

## 🎯 Hackathon Demo Points

### **Technical Innovation**
- Novel application of code embeddings for semantic drift detection
- Hybrid embedding approach combining code and natural language understanding
- Real-time git history analysis with visual feedback

### **Practical Value**
- Addresses real pain points in software development
- Scalable to any git repository
- Immediate actionable insights for development teams

### **User Experience**
- Intuitive web interface with beautiful visualizations
- One-click repository analysis
- Interactive timeline exploration
- Clear risk assessments and explanations

## 🔄 Future Enhancements

- **LLM Integration**: Use Claude/GPT for natural language explanations
- **Advanced ML**: Train custom models for breaking change prediction
- **Integration**: GitHub Apps, VS Code extensions, CI/CD webhooks
- **Collaboration**: Team insights, change approval workflows
- **Scale**: Enterprise deployment, multi-repository analysis

## 🏃‍♂️ Getting Started for Judges

1. **Quick Demo**: `./start.sh` → Open http://localhost:3000
2. **Standalone Demo**: `cd backend && python demo.py`
3. **Example Repository**: Try with `https://github.com/microsoft/vscode`
4. **Explore**: Select a file like `src/vs/editor/editor.api.ts`

## 🤝 Team & Acknowledgments

Built for the AI Berkeley Hackathon. Special thanks to the open-source community for the foundational tools that make this possible.

---

**Ready to detect feature drift in your codebase? Let's get started! 🚀**

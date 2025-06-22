# 🚀 DiffSense Backend

A comprehensive semantic code analysis engine that detects breaking changes, analyzes repository drift, and provides intelligent insights using AI. Now with **user authentication** and **persistent chat conversations**!

## ✨ Features

### 🔐 **User Authentication & Session Management**
- **GitHub OAuth integration** for seamless login
- **Private repository support** using user access tokens
- **Persistent user sessions** with secure token management
- **User-specific repository management**

### 💬 **Intelligent Chat System**
- **Multi-conversation support** with persistent chat history
- **Repository-specific conversations** with full context
- **Claude AI integration** for smart code analysis
- **Chat archiving and management**

### 🔍 **Breaking Change Detection**
- **Multi-language AST analysis** (Python, JavaScript, TypeScript, Java, Go, Rust)
- **ML-powered semantic drift detection** using advanced embeddings
- **12 different breaking change types** with confidence scoring
- **Risk assessment** with mitigation suggestions

### 🧠 **AI-Enhanced Analysis**
- **Claude API integration** for intelligent code review and insights
- **RAG system** for repository knowledge base and intelligent querying
- **Semantic search** across commits and files
- **Natural language explanations** of complex changes

### 📊 **Advanced Analytics**
- **Commit-by-commit analysis** with detailed risk scoring
- **Repository health dashboard** with trend analysis
- **Risk patterns** and stability assessment
- **Performance metrics** and development velocity analysis

## 🛠️ **Quick Start**

### 1. Setup
```bash
cd backend
chmod +x setup.sh
./setup.sh
```

### 2. Configure Environment
Copy the example environment file and configure:
```bash
cp .env.example .env
```

Edit `.env` file and add your API keys:
```bash
# Claude AI Configuration
CLAUDE_API_KEY=your_claude_api_key_here

# GitHub OAuth Configuration (for private repos)
GITHUB_CLIENT_ID=your_github_client_id_here
GITHUB_CLIENT_SECRET=your_github_client_secret_here
GITHUB_REDIRECT_URI=http://localhost:3001/auth/callback
```

### 3. Start Server
```bash
./start_server.sh
```

### 4. Test
```bash
cd ..
./quick_test.sh
```

## 🔐 **Authentication Setup**

To enable GitHub OAuth and private repository support:

1. **Create GitHub OAuth App**:
   - Go to GitHub Settings > Developer settings > OAuth Apps
   - Create a new OAuth App
   - Set Authorization callback URL to: `http://localhost:3001/auth/callback`

2. **Configure Environment**:
   ```bash
   GITHUB_CLIENT_ID=your_client_id
   GITHUB_CLIENT_SECRET=your_client_secret
   ```

3. **Test Authentication**:
   ```bash
   curl http://localhost:8000/api/auth/github
   ```

## 📡 **API Endpoints**

### 🔐 **Authentication**
- `GET /api/auth/github` - Initiate GitHub OAuth
- `POST /api/auth/github/callback` - Handle OAuth callback
- `GET /api/auth/user` - Get current user info
- `POST /api/auth/logout` - Logout user
- `GET /api/auth/github/repositories` - Get user's GitHub repos

### 💬 **Chat Management**
- `POST /api/chats` - Create new chat
- `GET /api/chats` - Get user's chats
- `GET /api/chats/{chat_id}` - Get specific chat
- `GET /api/chats/{chat_id}/messages` - Get chat messages
- `POST /api/chats/{chat_id}/messages` - Send message
- `PUT /api/chats/{chat_id}/title` - Update chat title
- `POST /api/chats/{chat_id}/archive` - Archive chat
- `DELETE /api/chats/{chat_id}` - Delete chat

### 📁 **Repository Management**
- `POST /api/clone-repository` - Clone repository (with auth support)
- `GET /api/user/repositories` - Get user's repositories

### **Core Repository Operations**
```bash
# Clone repository
POST /api/clone-repository
{
  "repo_url": "https://github.com/owner/repo",
  "max_commits": 50
}

# Get repository stats
GET /api/repository/{repo_id}/stats

# List repository files
GET /api/repository/{repo_id}/files

# Clean up repository
DELETE /api/repository/{repo_id}
```

### **Breaking Change Analysis**
```bash
# Analyze single commit
POST /api/analyze-commit/{repo_id}
{
  "commit_hash": "abc123",
  "include_claude_analysis": true
}

# Analyze commit range
POST /api/analyze-commit-range/{repo_id}
{
  "start_commit": "abc123",
  "end_commit": "def456",
  "max_commits": 100,
  "include_claude_analysis": true
}
```

### **RAG & Intelligent Querying**
```bash
# Query repository knowledge base
POST /api/query-repository/{repo_id}
{
  "query": "How does authentication work?",
  "max_results": 10,
  "include_claude_response": true
}

# Semantic commit search
GET /api/repository/{repo_id}/commits/search?q=authentication&limit=10

# Semantic file search
GET /api/repository/{repo_id}/files/search?q=auth&limit=10
```

### **Analytics & Risk Assessment**
```bash
# Repository summary
GET /api/repository/{repo_id}/summary

# Risk dashboard
GET /api/repository/{repo_id}/risk-dashboard

# Health check
GET /api/health
```

## 🏗️ **Architecture**

### **Core Components**

1. **GitAnalyzer** - Repository analysis and git operations
2. **BreakingChangeDetector** - ML-powered change detection
3. **EmbeddingEngine** - Semantic understanding using transformers
4. **RAGSystem** - Knowledge base and intelligent querying
5. **ClaudeAnalyzer** - AI-enhanced insights and explanations

### **Analysis Pipeline**

```
Repository → GitAnalyzer → Commits/Diffs
                ↓
Commits → EmbeddingEngine → Semantic Vectors
                ↓
Vectors → BreakingChangeDetector → Risk Analysis
                ↓
Analysis → ClaudeAnalyzer → Enhanced Insights
                ↓
Everything → RAGSystem → Queryable Knowledge Base
```

## 🔧 **Configuration**

All configuration is managed through the `.env` file:

```bash
# Required for enhanced analysis
ANTHROPIC_API_KEY=your_claude_api_key

# Optional configurations
CLAUDE_MODEL=claude-3-sonnet-20240229
CLAUDE_MAX_TOKENS=1500
EMBEDDING_MODEL=all-MiniLM-L6-v2
RISK_THRESHOLD_HIGH=0.7
SEMANTIC_SIMILARITY_THRESHOLD=0.8
```

## 📊 **Breaking Change Types Detected**

1. **API_SIGNATURE_CHANGE** - Function/method signature modifications
2. **FUNCTION_REMOVAL** - Functions or methods removed
3. **CLASS_INTERFACE_CHANGE** - Class structure modifications
4. **DEPENDENCY_CHANGE** - Import/dependency modifications
5. **CONFIG_CHANGE** - Configuration file changes
6. **DATABASE_SCHEMA_CHANGE** - Database structure changes
7. **URL_ROUTE_CHANGE** - API endpoint modifications
8. **DATA_FORMAT_CHANGE** - Data structure changes
9. **SECURITY_CHANGE** - Security-related modifications
10. **PERFORMANCE_DEGRADATION** - Performance impact changes
11. **COMPATIBILITY_BREAK** - Cross-platform compatibility issues
12. **SEMANTIC_DRIFT** - Semantic meaning changes

## 🎯 **Risk Assessment**

### **Risk Levels**
- **CRITICAL** (0.8-1.0): Almost certainly breaks existing code
- **HIGH** (0.6-0.8): Likely to cause issues
- **MEDIUM** (0.4-0.6): Potentially problematic
- **LOW** (0.2-0.4): Minor risk
- **MINIMAL** (0.0-0.2): Very unlikely to break anything

### **Confidence Scoring**
Each detection includes a confidence score (0.0-1.0) indicating how certain the system is about the analysis.

## 🔍 **Example Usage**

### **Analyze a Repository**
```bash
# Clone blockhunter repository
curl -X POST "http://localhost:8080/api/clone-repository" \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/jalenfran/blockhunter"}'

# Response: {"repo_id": "repo_0", "status": "cloned", ...}
```

### **Check for Breaking Changes**
```bash
# Analyze latest commit
curl -X POST "http://localhost:8080/api/analyze-commit/repo_0" \
  -H "Content-Type: application/json" \
  -d '{"commit_hash": "HEAD", "include_claude_analysis": true}'
```

### **Query Repository**
```bash
# Ask about authentication
curl -X POST "http://localhost:8080/api/query-repository/repo_0" \
  -H "Content-Type: application/json" \
  -d '{"query": "How does user authentication work in this codebase?"}'
```

## 🧪 **Testing**

### **Comprehensive Test**
```bash
./test_comprehensive.sh
```

### **Quick Test**
```bash
./quick_test.sh
```

### **Individual Endpoint Testing**
```bash
# Health check
curl http://localhost:8080/api/health

# Repository analysis
curl -X POST "http://localhost:8080/api/clone-repository" \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/owner/repo"}'
```

## 📦 **Dependencies**

### **Core Requirements**
- FastAPI 0.104.1+
- Python 3.8+
- Git

### **ML/AI Requirements**
- sentence-transformers 2.2.2+
- transformers 4.35.2+
- torch 2.2.0+
- anthropic 0.25.0+

### **Optional Dependencies**
- chromadb 0.4.18 (vector database)
- scikit-learn 1.3.2+ (additional ML features)

## 🚀 **Deployment**

### **Local Development**
```bash
./start_server.sh
```

### **Production Deployment**
```bash
# Using Docker (create Dockerfile)
docker build -t diffsense-backend .
docker run -p 8080:8080 diffsense-backend

# Using systemd service
sudo cp diffsense.service /etc/systemd/system/
sudo systemctl enable diffsense
sudo systemctl start diffsense
```

## 🔐 **Security**

- API keys stored in environment variables
- No sensitive data logged
- Repository data cleaned up automatically
- Rate limiting (configure in production)

## 📚 **API Documentation**

Interactive API documentation available at:
- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc

## 🤝 **Contributing**

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## 📄 **License**

MIT License - see LICENSE file for details.

---

**DiffSense** - Intelligent semantic code analysis for the modern developer! 🚀
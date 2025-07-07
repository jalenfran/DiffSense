# 🚀 DiffSense Backend

**Next-Generation Semantic Code Analysis & Repository Intelligence Platform**

DiffSense is a revolutionary backend system that provides intelligent code analysis, semantic drift detection, and AI-powered repository insights. Built with cutting-edge AI systems and advanced code analysis techniques.

## 🌟 Key Features

### 🧠 Enhanced AI Systems
- **Advanced Breaking Change Detector** - Revolutionary breaking change detection with semantic analysis
- **Enhanced RAG System** - Smart context gathering with deep code understanding  
- **Advanced Claude Analyzer** - Multi-expert AI consultation with domain expertise
- **Smart Code Analyzer** - Intelligent code analysis with complexity metrics
- **Smart Suggestions Engine** - Contextual recommendations and insights
◊
### 📊 Core Capabilities
- **Semantic Drift Detection** - Track code evolution and semantic changes
- **Repository Risk Assessment** - Comprehensive risk analysis and scoring
- **Intelligent Code Search** - AI-powered code discovery and context gathering
- **Real-time Chat Interface** - Interactive repository analysis conversations
- **Multi-Expert Analysis** - Security, Architecture, Performance, and Quality experts

## 🏗️ System Architecture

### Enhanced AI Components

#### 🔍 Advanced Breaking Change Detection
- **AST-based Analysis** - Deep parsing of code structure changes
- **Semantic Impact Assessment** - ML-powered change impact prediction
- **Multi-language Support** - Python, JavaScript, TypeScript, Java, C++, and more
- **Risk Categorization** - Critical, High, Medium, Low severity levels
- **Migration Guidance** - Automated suggestions for breaking change migrations

#### 🧩 Enhanced RAG System
- **Intelligent Context Gathering** - Smart file and commit discovery
- **Code Understanding** - Deep analysis of code architecture and patterns
- **Domain-Specific Analysis** - Security, performance, quality, and architecture insights
- **Multi-Expert Consultation** - Leverages different AI expert personas

#### 🤖 Advanced Claude Integration
- **Expert Personas** - Security Engineer, Software Architect, Performance Engineer, etc.
- **Domain Specialization** - Tailored analysis based on query intent
- **Structured Outputs** - Consistent, actionable insights
- **Advanced Prompting** - Chain-of-thought reasoning and few-shot examples

### Core Backend Services

#### 📦 Repository Management
- **GitAnalyzer** - Core Git operations and repository analysis
- **Storage Manager** - Organized repository storage with caching
- **Database Manager** - SQLite-based persistence with optimization

#### 🔐 Authentication & Security
- **GitHub OAuth Integration** - Secure authentication flow
- **Session Management** - Secure session handling
- **Repository Access Control** - User-based repository permissions

#### 💬 Chat System
- **Real-time Conversations** - Interactive repository analysis
- **Context-Aware Responses** - Repository-specific intelligent responses
- **Message History** - Persistent conversation storage

## 🔧 API Endpoints

### Repository Management
```
POST /api/clone-repository          # Clone and analyze repository
GET  /api/repository/{id}/stats     # Get repository statistics
GET  /api/repository/{id}/commits   # List repository commits
DELETE /api/repository/{id}         # Archive repository
```

### Advanced Analysis
```
POST /api/repository/{id}/query/enhanced           # Enhanced RAG query
GET  /api/repository/{id}/commit/{hash}/analysis   # Detailed commit analysis
POST /api/repository/{id}/commits/analyze-range    # Bulk commit analysis
GET  /api/repository/{id}/dashboard                # Repository dashboard
```

### Breaking Change Detection
```
GET  /api/repository/{id}/breaking-changes         # List breaking changes
GET  /api/repository/{id}/commit/{hash}/breaking   # Commit breaking changes
POST /api/repository/{id}/analyze-commit-range     # Range analysis
```

### Chat Interface
```
POST /api/chats                     # Create new chat
GET  /api/chats                     # List user chats
POST /api/chats/message             # Send chat message
PUT  /api/chats/{id}/title          # Update chat title
POST /api/chats/{id}/archive        # Archive chat
```

### Authentication
```
POST /api/auth/github              # GitHub OAuth login
GET  /api/auth/user                # Get current user
POST /api/auth/logout              # Logout user
GET  /api/auth/github/repos        # List GitHub repositories
```

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Git
- Claude API key (optional, for enhanced AI features)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd DiffSense/backend
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
# Copy and configure environment variables
cp .env.example .env

# Required variables:
CLAUDE_API_KEY=your_claude_api_key_here
GITHUB_CLIENT_ID=your_github_client_id
GITHUB_CLIENT_SECRET=your_github_client_secret
```

4. **Initialize database**
```bash
python -c "from src.database import DatabaseManager; DatabaseManager().init_database()"
```

5. **Start the server**
```bash
python main.py
```

The server will start on `http://localhost:8000`

### Docker Setup (Alternative)
```bash
docker build -t diffsense-backend .
docker run -p 8000:8000 -e CLAUDE_API_KEY=your_key diffsense-backend
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

## 🎯 **AI Expert Domains**

### Security Expert
- **Expertise**: Application Security, Vulnerability Assessment, Secure Code Review
- **Capabilities**: OWASP analysis, threat modeling, security pattern detection
- **Certifications**: CISSP, CEH, OSCP equivalent knowledge

### Software Architect  
- **Expertise**: System Design, Microservices, Scalability, Design Patterns
- **Capabilities**: Architecture analysis, scalability assessment, pattern recognition
- **Certifications**: AWS Solutions Architect, Azure Architect equivalent knowledge

### Performance Engineer
- **Expertise**: Performance Optimization, Profiling, Distributed Systems
- **Capabilities**: Bottleneck identification, optimization recommendations
- **Focus**: Metrics-driven analysis, performance pattern detection

### Code Quality Engineer
- **Expertise**: Code Review, Technical Debt, Refactoring, Testing Strategies
- **Capabilities**: Quality metrics, maintainability assessment, best practices
- **Focus**: Clean code principles, technical debt identification

## 🔧 Configuration

### Environment Variables
```bash
# Claude AI Configuration
CLAUDE_API_KEY=your_claude_api_key
CLAUDE_MODEL=claude-3-sonnet-20240229
CLAUDE_MAX_TOKENS=8000

# GitHub Integration
GITHUB_CLIENT_ID=your_github_client_id
GITHUB_CLIENT_SECRET=your_github_client_secret

# Analysis Settings
SEMANTIC_SIMILARITY_THRESHOLD=0.3
BREAKING_CHANGE_THRESHOLD=0.7
MAX_COMMITS_DEFAULT=50

# Development
DEBUG=false
LOG_LEVEL=INFO
```

### Database Configuration
The system uses SQLite by default with automatic schema management:
- **Repository data** - Cloned repositories and metadata
- **Commit analysis** - Cached commit analysis results
- **User sessions** - Authentication and session management
- **Chat history** - Conversation persistence

## 📈 Performance & Scalability

### Caching Strategy
- **File content caching** - Reduces Git operations
- **Analysis result caching** - Speeds up repeated queries
- **Embedding caching** - Optimizes semantic analysis

### Memory Management
- **Lazy loading** - Load repositories on-demand
- **Smart context limiting** - Manage Claude API token limits
- **Background processing** - Asynchronous repository indexing

### Monitoring
- **Request logging** - Comprehensive API request tracking
- **Error tracking** - Detailed error reporting and recovery
- **Performance metrics** - Response time and throughput monitoring

## 🛠️ Development

### Project Structure
```
backend/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── src/                   # Core source code
│   ├── advanced_breaking_change_detector.py  # Revolutionary breaking change detection
│   ├── enhanced_rag_system.py               # Smart RAG with deep code understanding
│   ├── advanced_claude_analyzer.py          # Multi-expert AI consultation
│   ├── smart_code_analyzer.py               # Intelligent code analysis
│   ├── smart_suggestions_engine.py          # Contextual suggestions
│   ├── claude_analyzer.py                   # Base Claude integration
│   ├── embedding_engine.py                  # AI embeddings and semantic analysis
│   ├── git_analyzer.py                      # Git operations and repository analysis
│   ├── database.py                          # Data persistence layer
│   ├── storage_manager.py                   # Repository file management
│   ├── github_service.py                    # GitHub API integration
│   └── config.py                           # Configuration management
├── repos/                 # Repository storage
│   ├── active/           # Active repositories
│   ├── archived/         # Archived repositories
│   └── temp/            # Temporary cloning space
└── diffsense.db          # SQLite database
```

### Testing
```bash
# Run basic syntax check
python -m py_compile main.py

# Test API endpoints
curl http://localhost:8000/

# Run with debug logging
DEBUG=true python main.py
```

### Adding New AI Experts
To add a new expert domain:

1. **Define Expert Persona** in `advanced_claude_analyzer.py`:
```python
'new_domain': {
    'title': 'Senior Domain Expert',
    'years_experience': '10+ years',
    'specialties': ['Domain Knowledge', 'Best Practices'],
    'thinking_style': 'domain-specific-approach'
}
```

2. **Create Domain Analysis** in `enhanced_rag_system.py`:
```python
def _create_domain_analysis(self, file_analyses, commits):
    # Domain-specific analysis logic
    return analysis_results
```

3. **Add Expert Prompt** for specialized domain prompting

## 🤝 Contributing

### Code Standards
- **Type hints** - Use Python type annotations
- **Docstrings** - Document all public methods
- **Error handling** - Comprehensive exception management
- **Logging** - Structured logging for debugging

### Pull Request Process
1. Fork the repository
2. Create feature branch
3. Implement changes with tests
4. Update documentation
5. Submit pull request

## 📝 Changelog

### Version 2.0.0 (Current)
- ✅ **Enhanced AI Systems** - Advanced breaking change detection and RAG
- ✅ **Multi-Expert Analysis** - Domain-specific AI consultation
- ✅ **Smart Code Analysis** - Intelligent code understanding
- ✅ **Legacy Cleanup** - Removed deprecated components
- ✅ **Performance Optimization** - Improved caching and response times

### Version 1.0.0 (Deprecated)
- Basic breaking change detection
- Simple RAG system
- Basic Claude integration

## 🎯 Roadmap

### Upcoming Features
- **Multi-repository analysis** - Cross-repository intelligence
- **CI/CD integration** - GitHub Actions and pipeline analysis
- **Advanced visualizations** - Interactive code evolution charts
- **Team collaboration** - Shared repository analysis
- **Custom AI models** - Fine-tuned models for specific domains

### Performance Improvements
- **Redis caching** - Distributed caching for scalability
- **Background workers** - Async processing optimization
- **API rate limiting** - Intelligent request management
- **Database optimization** - PostgreSQL migration for production

## 📞 Support

### Documentation
- **API Documentation** - Available at `/docs` when server is running
- **OpenAPI Spec** - Machine-readable API specification

### Issues & Support
- Create GitHub issues for bugs and feature requests
- Check existing documentation before reporting issues
- Provide detailed reproduction steps for bugs

### Community
- Join discussions in repository issues
- Share use cases and feedback
- Contribute to documentation improvements

---

**Built with ❤️ by the DiffSense Team**

*Revolutionizing code analysis through AI-powered semantic understanding*

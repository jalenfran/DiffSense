# DiffSense API Documentation

## Overview

DiffSense is a comprehensive breaking change detection and repository analysis API that combines machine learning, semantic analysis, and intelligent querying to help developers understand code changes and their potential impact.
Expand
message.txt
10 KB
﻿
Jayson
jayson_clark
he/him
# DiffSense API Documentation

## Overview

DiffSense is a comprehensive breaking change detection and repository analysis API that combines machine learning, semantic analysis, and intelligent querying to help developers understand code changes and their potential impact.

## Core Features

### 🔍 Breaking Change Detection
- **Commit-by-commit analysis** with ML-powered detection
- **Language-specific parsing** (Python, JavaScript, TypeScript, Java, Go, Rust)
- **Risk assessment** with confidence scoring
- **Mitigation suggestions** for detected issues

### 🧠 RAG (Retrieval-Augmented Generation) System
- **Semantic search** across commits and files
- **Intelligent querying** of repository knowledge
- **Context-aware responses** using embeddings
- **Repository knowledge base** with persistent storage

### 🤖 Claude API Integration
- **Enhanced analysis** with AI-powered insights
- **Risk assessment** with detailed explanations
- **Code review** capabilities
- **Natural language explanations** of complex changes

### 📊 Advanced Analytics
- **Risk dashboards** with trend analysis
- **Repository health metrics**
- **Semantic drift detection**
- **Performance monitoring**

## API Endpoints

### Repository Management

#### Clone Repository
```bash
POST /api/clone-repository
```
Clone and analyze a git repository.

**Request:**
```json
{
  "repo_url": "https://github.com/user/repo",
  "max_commits": 50
}
```

**Response:**
```json
{
  "repo_id": "repo_0",
  "status": "cloned",
  "stats": {
    "total_commits": 150,
    "contributors": 5,
    "file_count": 45
  }
}
```

#### Repository Summary
```bash
GET /api/repository/{repo_id}/summary
```
Get comprehensive repository analysis with AI-enhanced insights.

#### Repository Stats
```bash
GET /api/repository/{repo_id}/stats
```
Get basic repository statistics.

#### List Files
```bash
GET /api/repository/{repo_id}/files
```
List all files in the repository with metadata.

### Breaking Change Detection

#### Analyze Single Commit
```bash
POST /api/analyze-commit/{repo_id}
```
Analyze a specific commit for breaking changes.

**Request:**
```json
{
  "commit_hash": "abc123...",
  "include_claude_analysis": true
}
```

**Response:**
```json
{
  "commit_hash": "abc123...",
  "overall_risk_score": 0.75,
  "breaking_changes": [
    {
      "change_type": "function_removal",
      "risk_level": "high",
      "confidence": 0.95,
      "file_path": "src/api.py",
      "description": "Function 'deprecated_method' was removed",
      "mitigation_suggestions": [
        "Add deprecation warning before removal",
        "Provide migration guide"
      ]
    }
  ],
  "claude_analysis": {
    "content": "This commit removes a public API function...",
    "confidence": 0.9,
    "suggestions": ["Consider gradual deprecation"]
  }
}
```

#### Analyze Commit Range
```bash
POST /api/analyze-commit-range/{repo_id}
```
Analyze multiple commits for breaking changes and trends.

**Request:**
```json
{
  "start_commit": "abc123...",
  "end_commit": "def456...",
  "max_commits": 20,
  "include_claude_analysis": true
}
```

### RAG System & Intelligent Queries

#### Query Repository
```bash
POST /api/query-repository/{repo_id}
```
Ask natural language questions about the repository.

**Request:**
```json
{
  "query": "What are the main components of this project?",
  "max_results": 10,
  "include_claude_response": true
}
```

**Response:**
```json
{
  "query": "What are the main components...",
  "response": "Based on the repository analysis, the main components are...",
  "confidence": 0.85,
  "sources": [
    {
      "type": "file",
      "path": "src/main.py",
      "relevance": 0.9
    }
  ],
  "claude_enhanced": true
}
```

#### Search Commits
```bash
GET /api/repository/{repo_id}/commits/search?query=bug fix&max_results=10
```
Semantic search across commit messages and changes.

#### Search Files
```bash
GET /api/repository/{repo_id}/files/search?query=authentication&max_results=10
```
Semantic search across files and their content.

### Risk Analysis & Dashboards

#### Risk Dashboard
```bash
GET /api/repository/{repo_id}/risk-dashboard
```
Get comprehensive risk analysis dashboard.

**Response:**
```json
{
  "overall_risk_score": 0.65,
  "total_commits_analyzed": 50,
  "high_risk_commits": 8,
  "breaking_changes_by_type": {
    "function_removal": 3,
    "api_signature_change": 5,
    "dependency_change": 2
  },
  "risk_trend": [
    {
      "commit_hash": "abc123",
      "timestamp": "2024-01-15T10:30:00Z",
      "risk_score": 0.8
    }
  ],
  "most_risky_files": {
    "src/api.py": 2.4,
    "src/core.py": 1.8
  }
}
```

## Breaking Change Types

The system detects various types of breaking changes:

- **API_SIGNATURE_CHANGE**: Function/method signature modifications
- **FUNCTION_REMOVAL**: Deleted functions or methods
- **CLASS_REMOVAL**: Deleted classes or interfaces
- **PARAMETER_CHANGE**: Changed function parameters
- **RETURN_TYPE_CHANGE**: Modified return types
- **ACCESS_MODIFIER_CHANGE**: Visibility changes (public to private)
- **DEPENDENCY_CHANGE**: Added/removed dependencies
- **CONFIG_CHANGE**: Configuration file modifications
- **DATABASE_SCHEMA_CHANGE**: Database structure changes
- **FILE_STRUCTURE_CHANGE**: Project structure modifications
- **SEMANTIC_CHANGE**: Semantic meaning changes detected via embeddings

## Risk Levels

- **LOW (0.0-0.3)**: Minor changes, low impact
- **MEDIUM (0.3-0.6)**: Moderate changes, potential impact
- **HIGH (0.6-0.8)**: Significant changes, likely impact
- **CRITICAL (0.8-1.0)**: Major changes, definite impact

## Configuration

### Environment Variables

```bash
# Claude API (optional, for enhanced analysis)
export ANTHROPIC_API_KEY="your-claude-api-key"

# Server configuration
export PORT=8080
export HOST="0.0.0.0"
```

### Embedding Engine Options

The system supports multiple embedding approaches:
- **Sentence Transformers**: Default, good balance of speed and accuracy
- **OpenAI Embeddings**: Higher quality, requires API key
- **Custom Models**: For specialized domains

## Example Usage

### 1. Basic Repository Analysis
```bash
# Clone repository
curl -X POST "http://localhost:8080/api/clone-repository" \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/user/repo"}'

# Get risk dashboard
curl -X GET "http://localhost:8080/api/repository/repo_0/risk-dashboard"
```

### 2. Breaking Change Detection
```bash
# Analyze specific commit
curl -X POST "http://localhost:8080/api/analyze-commit/repo_0" \
  -H "Content-Type: application/json" \
  -d '{"commit_hash": "abc123", "include_claude_analysis": true}'
```

### 3. Intelligent Querying
```bash
# Ask about repository
curl -X POST "http://localhost:8080/api/query-repository/repo_0" \
  -H "Content-Type: application/json" \
  -d '{"query": "What files handle user authentication?", "include_claude_response": true}'
```

## Performance Considerations

- **Repository Size**: Large repositories (>1000 commits) may take longer to index
- **Embedding Generation**: First-time analysis includes embedding computation
- **Claude API**: Optional but adds ~1-2 seconds per request
- **Memory Usage**: Scales with repository size and embedding cache

## Error Handling

The API returns standard HTTP status codes:
- **200**: Success
- **400**: Bad request (invalid parameters)
- **404**: Repository not found
- **500**: Internal server error

Error responses include detailed messages:
```json
{
  "detail": "Repository not found. Please clone first.",
  "error_code": "REPO_NOT_FOUND"
}
```

## Integration Examples

### CI/CD Integration
```yaml
# GitHub Actions example
- name: Analyze Breaking Changes
  run: |
    REPO_ID=$(curl -X POST "$API_URL/api/clone-repository" \
      -d '{"repo_url": "${{ github.repository_url }}"}' | jq -r '.repo_id')
    
    curl -X POST "$API_URL/api/analyze-commit/$REPO_ID" \
      -d '{"commit_hash": "${{ github.sha }}"}'
```

### Development Workflow
```bash
#!/bin/bash
# Pre-commit hook for breaking change detection
LATEST_COMMIT=$(git rev-parse HEAD)
ANALYSIS=$(curl -X POST "http://localhost:8080/api/analyze-commit/repo_0" \
  -d "{\"commit_hash\": \"$LATEST_COMMIT\"}")

RISK_SCORE=$(echo "$ANALYSIS" | jq -r '.overall_risk_score')
if (( $(echo "$RISK_SCORE > 0.7" | bc -l) )); then
  echo "⚠️  High risk changes detected! Review before pushing."
  exit 1
fi
```

## Roadmap

### Upcoming Features
- **Real-time Analysis**: Pre-commit hooks and IDE integration
- **Machine Learning Models**: Custom trained models for specific languages
- **Advanced Visualizations**: Interactive risk trend charts
- **Team Analytics**: Contributor risk profiles and patterns
- **Integration APIs**: Slack, Teams, and other notification systems

### Performance Improvements
- **Incremental Analysis**: Only analyze new commits
- **Distributed Processing**: Scale across multiple workers
- **Caching Strategies**: Improved embedding and analysis caching
- **Streaming Responses**: Real-time progress updates
message.txt
10 KB
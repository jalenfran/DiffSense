# DiffSense API Documentation

## Overview

DiffSense provides repository cloning, breaking-change analysis, RAG-based queries, persistent chat, and GitHub OAuth authentication.

---

## Health Check

GET `/`
- Returns service status and configuration health.

Response:
```json
{ "message": "DiffSense API is running", "timestamp": "2025-06-22T...Z", "config_status": true }
```

---

## Authentication

### Initiate GitHub OAuth
GET `/api/auth/github`
- Returns OAuth URL and state.

### OAuth Callback
GET `/api/auth/github/callback?code=<code>&state=<state>`
- GitHub redirects here after user authorization
- Uses Referer header to determine frontend origin for redirect
- Redirects to frontend with session_id: `{frontend_origin}/auth/callback?session_id={session_id}&success=true`
- On error: `{frontend_origin}/auth/callback?error={message}&success=false`

### Get Current User
GET `/api/auth/user`
- Header: `Authorization: Bearer <session_id>`
- Returns user profile.

### Logout
POST `/api/auth/logout`
- Header: `Authorization: Bearer <session_id>`
- Invalidates session.

---

## Repository Management

### Clone Repository
POST `/api/clone-repository`
```json
{ "repo_url": "https://github.com/user/repo", "use_auth": false }
```
- Clones (supports private repos), returns `{ repo_id, status, stats, message }`.

### List Files
GET `/api/repository/{repo_id}/files`
- Lists code files.

### Repository Stats
GET `/api/repository/{repo_id}/stats`
- Returns stats: total_commits, contributors, branches, file_count.

### List Commits
GET `/api/repository/{repo_id}/commits?limit=50`
- Returns recent commits with metadata.

### Get Commit Files
GET `/api/repository/{repo_id}/commit/{commit_hash}/files?include_diff_stats=false`
- Lists all files changed in a specific commit with optional diff statistics.

### Cleanup Repository
DELETE `/api/repository/{repo_id}`
- Deletes local clone and resources.

---

## Breaking Change Analysis

### Analyze Single Commit
POST `/api/analyze-commit/{repo_id}`
```json
{ "commit_hash": "<hash>", "include_claude_analysis": true }
```
- Returns risk score, breaking_changes, optional Claude analysis.

### Analyze Commit Range
POST `/api/analyze-commit-range/{repo_id}`
```json
{ "start_commit": "<hash>", "end_commit": "<hash>", "max_commits": 100, "include_claude_analysis": false }
```
- Batch analysis with trends.

---

## RAG Queries & Intelligent Search

### Repository Query
POST `/api/query-repository/{repo_id}`
```json
{ "query": "...", "max_results": 10 }
```
- QA over repository, returns `{ query, response, confidence, sources, context_used, suggestions, claude_enhanced }`.

### Enhanced RAG Query
POST `/api/repository/{repo_id}/query/enhanced`
```json
{ "query": "...", "max_results": 10 }
```
- RAG + Claude-enhanced response with related content.

### Search Commits
GET `/api/repository/{repo_id}/commits/search?query=...&max_results=10`

### Search Files
GET `/api/repository/{repo_id}/files/search?query=...&max_results=10`

---

## Advanced Insights

### Repository Summary
GET `/api/repository/{repo_id}/summary`
- High-level summary with AI risk assessment.

### Risk Dashboard
GET `/api/repository/{repo_id}/risk-dashboard`
- Trends, high-risk commits, file-level risks.

### Commit Analysis
GET `/api/repository/{repo_id}/commit/{commit_hash}/analysis`
- Detailed commit analysis with RAG context.

### File Insights
GET `/api/repository/{repo_id}/file/{file_path:path}/insights`
- File history, risk metrics, RAG-powered analysis.

### Semantic Drift Analysis
GET `/api/repository/{repo_id}/insights/semantic-drift?days=30`
- Drift patterns, themes, developer impact.

---

## Chat System (Persistent)

All chat endpoints require `Authorization: Bearer <session_id>`.

### Create Chat
POST `/api/chats`
```json
{ "repo_id": "optional-repo-id", "title": "Chat Title" }
```
- Returns chat metadata.

### List Chats
GET `/api/chats?include_archived=false`
- Returns user’s chats.

### Get Chat & Messages
GET `/api/chats/{chat_id}`
- Returns chat details and messages.

### Send Message
POST `/api/chats/{chat_id}/messages`
```json
{ "chat_id": "<id>", "message": "..." }
```
- Appends user message and returns AI response.

### Update Chat Title
PUT `/api/chats/{chat_id}`
```json
{ "title": "New Title" }
```

### Delete Chat
DELETE `/api/chats/{chat_id}`

### Archive Chat
POST `/api/chats/{chat_id}/archive`

---

## Deprecated Endpoints

- POST `/api/analyze-file/{repo_id}` (use File Insights)
- POST `/api/analyze-function/{repo_id}` (use Query Repository)

---

## Error Handling

- 200 OK
- 400 Bad Request
- 401 Unauthorized
- 403 Forbidden
- 404 Not Found
- 500 Internal Server Error

Error response:
```json
{ "detail": "<message>" }
```
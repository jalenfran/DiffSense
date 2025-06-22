# DiffSense Frontend Demo

This document provides a quick demo of the new DiffSense frontend features.

## Demo Setup

1. **DiffSense API Server**
   ```bash
   # The DiffSense API is running on http://76.125.217.28:8080
   # You should see endpoints like:
   # - POST /api/clone-repository
   # - GET /api/repository/{id}/stats
   # - POST /api/query-repository/{id}
   ```

2. **Start the Frontend**
   ```bash
   cd frontend
   npm run dev
   # Visit http://localhost:5173
   # OAuth server (if needed) runs on localhost:3000
   ```

## Demo Walkthrough

### Step 1: Add a Repository via URL

1. Open the application (auth is optional for this demo)
2. Click "Add Repository" in the sidebar
3. Switch to "Add by URL" tab
4. Enter a repository URL like: `https://github.com/octocat/Hello-World`
5. Click "Add Repository"

**What happens:**
- The frontend calls the DiffSense API to clone and analyze the repository
- Repository stats, files, and risk analysis are fetched
- The repository appears in the sidebar

### Step 2: Explore the Risk Dashboard

Once a repository is selected, you'll see:

1. **Repository Header**
   - Basic stats (stars, forks, language, file count)
   - Links to GitHub

2. **Risk Analysis Panel**
   - Overall risk score with visual indicator
   - High-risk commits count
   - Total commits analyzed
   - Breaking changes by category

3. **Error Handling**
   - If the API is not running, you'll see connection errors
   - Repository analysis failures are displayed clearly

### Step 3: Use the Chat Interface

The bottom panel contains an interactive chat:

1. **Quick Prompts**: Click any of the predefined prompts:
   - "Analyze recent changes"
   - "Find breaking changes"  
   - "Code structure overview"
   - "Security concerns"

2. **Custom Questions**: Type your own questions like:
   - "What files handle authentication?"
   - "Are there any deprecated functions?"
   - "What are the main dependencies?"

3. **File Context**: 
   - Click "Select files" to choose specific files
   - The AI responses will focus on those files

**Chat Response Features:**
- Confidence scores for AI responses
- Source citations with relevance ratings
- Expandable sections for detailed information
- Copy functionality for responses

### Step 4: Breaking Change Analysis

When the API detects breaking changes, you'll see:

1. **Risk Indicators**: Color-coded risk levels (green/yellow/red)
2. **Change Categories**: 
   - Function removals
   - API signature changes
   - Parameter modifications
   - Dependency changes
3. **AI Analysis**: Claude-powered insights and suggestions

## Sample API Responses

### Repository Query Response
```json
{
  "response": "This project appears to be a simple Hello World repository...",
  "confidence": 0.85,
  "sources": [
    {
      "type": "file",
      "path": "README.md",
      "relevance": 0.9
    }
  ],
  "claude_enhanced": true
}
```

### Commit Analysis Response
```json
{
  "commit_hash": "abc123",
  "overall_risk_score": 0.75,
  "breaking_changes": [
    {
      "change_type": "function_removal",
      "risk_level": "high",
      "confidence": 0.95,
      "file_path": "src/api.js",
      "description": "Function 'deprecated_method' was removed"
    }
  ],
  "claude_analysis": {
    "content": "This commit removes a public API function...",
    "suggestions": ["Consider gradual deprecation"]
  }
}
```

## Testing Without the API

If the DiffSense API is not available, you can still test the frontend:

1. **Repository Addition**: Works with GitHub URLs (creates mock repository objects)
2. **UI Components**: All UI elements are functional
3. **Error Handling**: You'll see proper error messages for API failures
4. **Dark Mode**: Theme switching works independently

## Troubleshooting

### Common Issues

1. **"Failed to analyze repository"**
   - Check if DiffSense API is accessible at 76.125.217.28:8080
   - Verify the repository URL is accessible
   - Check browser console for detailed errors
   - Ensure CORS is properly configured on the API server

2. **Chat not responding**
   - Ensure repository was successfully added
   - Check if Claude API key is configured on the backend
   - Verify network connectivity to the API server

3. **CORS errors**
   - The Vite proxy should handle this automatically
   - Check vite.config.js proxy settings
   - Ensure the API server allows requests from your domain

### Debug Information

Open browser dev tools and check:
- Network tab for API requests
- Console for JavaScript errors
- Application tab for localStorage data

## Next Steps

This integration provides:
- ✅ Full API connectivity
- ✅ Repository management via URL
- ✅ Interactive chat interface
- ✅ Risk dashboard and analysis
- ✅ File selection for context
- ✅ Error handling and loading states

Future enhancements could include:
- Real-time updates via WebSockets  
- Advanced visualization components
- Integration with GitHub webhooks
- Team collaboration features
- Custom risk model configuration

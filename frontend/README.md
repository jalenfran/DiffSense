# DiffSense Frontend

A React-based frontend for the DiffSense API that provides breaking change detection and repository analysis with AI-powered insights.

## New Features

### 🔗 API Integration
- Full integration with DiffSense API (http://localhost:8080)
- Repository cloning and analysis via URL input
- Real-time breaking change detection
- Risk assessment and dashboard

### 💬 Chat Interface
- Interactive chat for querying repositories
- AI-powered responses using Claude API
- File-specific context selection
- Structured responses with confidence scores

### 📊 Risk Dashboard
- Visual risk assessment metrics
- Breaking change categorization
- Commit-by-commit analysis
- Trend analysis and insights

## Getting Started

### Prerequisites
- Node.js (v16 or higher)
- DiffSense API server accessible at http://76.125.217.28:8080
- Optional: Authentication server on http://localhost:3000

### Installation

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

The frontend will be available at http://localhost:5173

## Usage

### Adding Repositories

#### Method 1: URL Input
1. Click "Add Repository" in the sidebar
2. Switch to "Add by URL" tab
3. Paste a GitHub repository URL (e.g., `https://github.com/user/repo`)
4. Click "Add Repository"

The system will:
- Clone and analyze the repository via the DiffSense API
- Extract file structure and commit history
- Generate risk assessments and breaking change analysis

#### Method 2: Browse Mode (OAuth Required)
1. Click "Add Repository" in the sidebar
2. Use the "Browse Your Repos" tab to select from your GitHub repositories
3. Click on any repository to add it

### Chat Interface

Once a repository is selected:

1. **Basic Queries**: Ask natural language questions about the repository
   - "What are the main components of this project?"
   - "Are there any breaking changes in recent commits?"
   - "What security vulnerabilities exist?"

2. **File Context**: Select specific files to focus the conversation
   - Click "Select files" to choose relevant files
   - The AI will prioritize these files in responses

3. **Quick Prompts**: Use predefined prompts for common analyses
   - Analyze recent changes
   - Find breaking changes
   - Code structure overview
   - Security concerns

### Risk Dashboard

The risk dashboard shows:
- **Overall Risk Score**: Percentage-based risk assessment
- **High Risk Commits**: Number of commits with significant risk
- **Breaking Changes by Type**: Categorized breaking changes
- **Risk Trends**: Historical risk analysis

## API Endpoints Used

- `POST /api/clone-repository` - Clone and analyze repositories
- `GET /api/repository/{id}/stats` - Get repository statistics
- `GET /api/repository/{id}/files` - List repository files
- `GET /api/repository/{id}/risk-dashboard` - Risk analysis dashboard
- `POST /api/query-repository/{id}` - Chat with repository
- `POST /api/analyze-commit/{id}` - Analyze specific commits

## Configuration

The API base URL and other settings can be configured in `src/config/index.js`:

```javascript
export const config = {
  API_BASE_URL: 'http://76.125.217.28:8080/api',
  API_TIMEOUT: 30000,
  DEFAULT_MAX_COMMITS: 50,
  DEFAULT_MAX_RESULTS: 10
}
```

## Architecture

### Components
- **ChatInterface**: Interactive chat for repository queries
- **MainContent**: Repository dashboard and risk analysis
- **Sidebar**: Repository management and selection

### Contexts
- **RepositoryContext**: Manages repository state and API interactions
- **DarkModeContext**: Theme management

### Services
- **api.js**: DiffSense API client with axios

## Features in Detail

### Breaking Change Detection
- Function/method signature changes
- Class and interface removals
- Parameter modifications
- Return type changes
- Access modifier changes
- Dependency changes
- Database schema changes

### Risk Assessment
- **Low (0-30%)**: Minor changes, low impact
- **Medium (30-60%)**: Moderate changes, potential impact
- **High (60-80%)**: Significant changes, likely impact
- **Critical (80-100%)**: Major changes, definite impact

### Chat Responses
- Structured responses with confidence scores
- Source citations with relevance ratings
- AI-enhanced analysis with Claude integration
- Expandable sections for detailed information

## Development

### File Structure
```
src/
├── components/
│   ├── ChatInterface.jsx      # Interactive chat component
│   ├── Dashboard.jsx          # Main dashboard layout
│   ├── MainContent.jsx        # Repository content area
│   └── Sidebar.jsx           # Repository sidebar
├── contexts/
│   ├── DarkModeContext.jsx   # Theme context
│   └── RepositoryContext.jsx # Repository state management
├── services/
│   └── api.js               # API client
└── config/
    └── index.js            # Configuration settings
```

### State Management
- Repository state managed via React Context
- Local storage for favorites and recently viewed
- Real-time updates via API polling (future enhancement)

## Troubleshooting

### Common Issues

1. **API Connection Failed**
   - Ensure DiffSense API is running on port 8080
   - Check CORS settings on the API server
   - Verify network connectivity

2. **Repository Clone Failed**
   - Check if the repository URL is valid
   - Ensure the repository is public or you have access
   - Verify GitHub API rate limits

3. **Chat Not Responding**
   - Check if Claude API key is configured on the backend
   - Verify repository has been successfully cloned
   - Check browser console for error messages

### Debug Mode
Enable debug logging by opening browser developer tools and setting:
```javascript
localStorage.setItem('debug', 'diffsense:*')
```

## Future Enhancements

- Real-time commit analysis
- IDE integration (VS Code extension)
- Slack/Teams notifications
- Custom risk models
- Advanced visualizations
- Team collaboration features

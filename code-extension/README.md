# DiffSense VS Code Extension

An AI-powered extension for analyzing Git diffs and providing code insights directly within VS Code.

## Features

- **GitHub OAuth Integration**: Uses VS Code's built-in GitHub authentication
- **AI Chat Interface**: Interactive chat for discussing code changes and repository insights  
- **VS Code Theme Integration**: Seamlessly blends with your VS Code theme (light/dark mode)
- **Modern UI**: Clean, responsive interface matching the DiffSense frontend aesthetic

## Setup

1. **Install the Extension**: The extension should be automatically loaded in your VS Code workspace

2. **Authenticate**: 
   - Open the DiffSense panel in the VS Code sidebar
   - Click "Continue with GitHub" to authenticate using VS Code's built-in GitHub authentication
   - VS Code will handle the OAuth flow natively - no browser redirects needed!

## Usage

- **Chat Interface**: Once authenticated, you can chat with the DiffSense AI about your code changes
- **Repository Analysis**: Ask questions about your Git diffs, code patterns, and development workflow
- **Refresh Authentication**: Use the refresh button if you need to re-check authentication status

## Authentication Flow

The extension uses VS Code's native GitHub authentication provider:

1. When you click "Continue with GitHub", VS Code handles the OAuth flow
2. No external servers or redirect URLs needed
3. Authentication tokens are managed securely by VS Code
4. The extension receives the authentication session with GitHub API access

## Development

To build the extension:

```bash
npm run compile
```

To watch for changes:

```bash
npm run watch
```

## Dependencies

- `react`: UI framework for the webview
- `react-dom`: React DOM renderer

Note: This extension uses VS Code's built-in GitHub authentication, so it doesn't require a separate OAuth server.

import * as vscode from 'vscode';

interface User {
    id: string;
    username: string;
    displayName?: string;
    photos?: Array<{ value: string }>;
    accessToken?: string;
}

export function activate(context: vscode.ExtensionContext) {
    console.log('DiffSense Extension is now active!');
    
    // Register a test command
    const disposable = vscode.commands.registerCommand('myChatView.focus', () => {
        vscode.window.showInformationMessage('DiffSense chat view focused!');
    });
    context.subscriptions.push(disposable);
    
    const provider = new ChatViewProvider(context.extensionUri);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(
            ChatViewProvider.viewType,
            provider
        )
    );
    console.log('DiffSense chat view provider registered successfully');
}

class ChatViewProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = 'myChatView';
    private currentUser: User | null = null;
    private webviewView: vscode.WebviewView | null = null;
    
    constructor(private readonly extensionUri: vscode.Uri) { }

    public resolveWebviewView(
        webviewView: vscode.WebviewView,
        context: vscode.WebviewViewResolveContext
    ) {
        console.log('resolveWebviewView called!');
        this.webviewView = webviewView;
        
        webviewView.webview.options = {
            // Allow scripts & local resource loading
            enableScripts: true,
            localResourceRoots: [this.extensionUri]
        };

        // Check authentication status on load
        this.checkAuthStatus().then(() => {
            // Set the HTML content
            webviewView.webview.html = this.getHtmlForWebview(webviewView.webview);
        });

        // Handle messages from the webview
        webviewView.webview.onDidReceiveMessage((msg) => {
            switch (msg.command) {
                case 'sendMessage':
                    this.handleUserMessage(msg.text, webviewView);
                    return;
                case 'githubLogin':
                    this.handleGitHubLogin();
                    return;                case 'checkAuth':
                    this.checkAuthStatus().then(() => {
                        webviewView.webview.postMessage({
                            command: 'authStatus',
                            authenticated: !!this.currentUser,
                            user: this.currentUser
                        });
                    });
                    return;
                case 'logout':
                    this.handleLogout();
                    return;
            }
        });
    }    private async checkAuthStatus(): Promise<void> {
        try {
            // Try to get GitHub authentication session
            const session = await vscode.authentication.getSession('github', ['repo', 'user:email'], { silent: true });
            
            if (session) {
                // Create user object from GitHub session
                this.currentUser = {
                    id: session.account.id,
                    username: session.account.label,
                    displayName: session.account.label,
                    accessToken: session.accessToken
                };
            } else {
                this.currentUser = null;
            }
        } catch (error) {
            console.log('GitHub authentication not available:', error);
            this.currentUser = null;
        }
    }    private async handleGitHubLogin(): Promise<void> {
        try {
            // Use VS Code's built-in GitHub authentication
            const session = await vscode.authentication.getSession('github', ['repo', 'user:email'], { createIfNone: true });
            
            if (session) {
                this.currentUser = {
                    id: session.account.id,
                    username: session.account.label, 
                    displayName: session.account.label,
                    accessToken: session.accessToken
                };

                // Send updated auth status to webview
                if (this.webviewView) {
                    this.webviewView.webview.postMessage({
                        command: 'authStatus',
                        authenticated: true,
                        user: this.currentUser
                    });
                }

                vscode.window.showInformationMessage('Successfully authenticated with GitHub!');
            }
        } catch (error) {
            console.error('GitHub authentication failed:', error);
            vscode.window.showErrorMessage('Failed to authenticate with GitHub. Please try again.');
        }
    }    private async handleLogout(): Promise<void> {
        try {
            // Clear the current user
            this.currentUser = null;
            
            // Send updated auth status to webview
            if (this.webviewView) {
                this.webviewView.webview.postMessage({
                    command: 'authStatus',
                    authenticated: false,
                    user: null
                });
            }

            vscode.window.showInformationMessage('Successfully logged out.');
        } catch (error) {
            console.error('Logout failed:', error);
        }
    }

    private async refreshAuth(): Promise<void> {
        await this.checkAuthStatus();
        // Send updated auth status to webview
        if (this.webviewView) {
            this.webviewView.webview.postMessage({
                command: 'authStatus',
                authenticated: !!this.currentUser,
                user: this.currentUser
            });
        }
    }private getHtmlForWebview(webview: vscode.Webview): string {
        // Local path to main script run in the webview
        const scriptUri = webview.asWebviewUri(
            vscode.Uri.joinPath(this.extensionUri, 'media', 'index.js')
        );

        // And any CSS
        const styleUri = webview.asWebviewUri(
            vscode.Uri.joinPath(this.extensionUri, 'media', 'chat.css')
        );

        return /* html */`
      <!DOCTYPE html>
      <html lang="en">
      <head>
        <meta charset="UTF-8" />
        <meta
          http-equiv="Content-Security-Policy"
          content="default-src 'none'; img-src https: data: ${webview.cspSource}; script-src ${webview.cspSource}; style-src ${webview.cspSource} 'unsafe-inline'; font-src ${webview.cspSource};"
        />
        <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
        <link href="${styleUri}" rel="stylesheet" />
        <title>DiffSense AI</title>
      </head>
      <body>
        <div id="root"></div>
        <script type="module" src="${scriptUri}"></script>
      </body>
      </html>`;
    }private async handleUserMessage(text: string, view: vscode.WebviewView) {
        // Simulate typing delay
        setTimeout(() => {
            const responses = [
                "That's an interesting question! Let me think about that...",
                "I understand what you're asking. Here's my take on it:",
                "Great question! I'd be happy to help you with that.",
                "Thanks for asking! Here's what I think:",
                "That's a thoughtful question. Let me share some insights:",
            ];
            
            const randomResponse = responses[Math.floor(Math.random() * responses.length)];
            
            view.webview.postMessage({
                command: 'newMessage',
                text: `${randomResponse}\n\n${text}`,
                from: 'bot'
            });
        }, 1000 + Math.random() * 2000); // Random delay between 1-3 seconds
    }
}

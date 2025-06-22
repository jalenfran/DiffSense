import { useState, useEffect } from 'react'
import { diffSenseAPI } from './services/api'
import LandingPage from './components/LandingPage'
import Dashboard from './components/Dashboard'
import LoadingSpinner from './components/LoadingSpinner'
import { DarkModeProvider } from './contexts/DarkModeContext'
import { RepositoryProvider } from './contexts/RepositoryContext'
import { FileViewerProvider } from './contexts/FileViewerContext'
import { CommitViewerProvider } from './contexts/CommitViewerContext'

function App() {
    const [user, setUser] = useState(null)
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        checkAuthStatus()
        handleOAuthCallback()
    }, [])

    const checkAuthStatus = async () => {
        try {
            // Check if we have a session token
            if (diffSenseAPI.isAuthenticated()) {
                const userData = await diffSenseAPI.getCurrentUser()
                console.log('Auth status response:', userData)
                setUser(userData)
            }
        } catch (error) {
            console.log('Not authenticated:', error)
            // Clear invalid token
            diffSenseAPI.setSessionToken(null)
        } finally {
            setLoading(false)
        }
    }

    const handleOAuthCallback = async () => {
        // Check if we're returning from GitHub OAuth
        const urlParams = new URLSearchParams(window.location.search)
        const code = urlParams.get('code')
        const state = urlParams.get('state')
        const sessionId = urlParams.get('session_id')
        const authSuccess = urlParams.get('auth_success')

        // Handle different callback patterns
        if (sessionId) {
            // Backend redirected with session_id in URL
            try {
                setLoading(true)
                diffSenseAPI.setSessionToken(sessionId)
                const userData = await diffSenseAPI.getCurrentUser()
                console.log('OAuth success with session_id:', userData)
                setUser(userData)

                // Clean up URL
                window.history.replaceState({}, document.title, window.location.pathname)
            } catch (error) {
                console.error('Failed to get user with session_id:', error)
                diffSenseAPI.setSessionToken(null)
            } finally {
                setLoading(false)
            }
        } else if (authSuccess === 'true') {
            // Backend set session via cookie or other method
            try {
                setLoading(true)
                const userData = await diffSenseAPI.getCurrentUser()
                console.log('OAuth success via cookie:', userData)
                setUser(userData)

                // Clean up URL
                window.history.replaceState({}, document.title, window.location.pathname)
            } catch (error) {
                console.error('Failed to get user after auth_success:', error)
            } finally {
                setLoading(false)
            }
        } else if (code && state) {
            // Frontend needs to handle the OAuth callback (original pattern)
            try {
                setLoading(true)
                const authResult = await diffSenseAPI.handleGitHubCallback(code, state)
                console.log('OAuth callback result:', authResult)

                if (authResult.user) {
                    setUser(authResult.user)
                }

                // Clean up URL
                window.history.replaceState({}, document.title, window.location.pathname)
            } catch (error) {
                console.error('OAuth callback error:', error)
            } finally {
                setLoading(false)
            }
        }
    }

    const handleLogout = async () => {
        try {
            await diffSenseAPI.logout()
        } catch (error) {
            console.error('Logout error:', error)
        } finally {
            setUser(null)
        }
    }

    if (loading) {
        return <LoadingSpinner />
    }    return (
        <DarkModeProvider>
            <RepositoryProvider>
                <FileViewerProvider>
                    <CommitViewerProvider>
                        <div className="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors">
                            {user ? (
                                <Dashboard
                                    user={user}
                                    onLogout={handleLogout}
                                />
                            ) : (
                                <LandingPage />
                            )}
                        </div>
                    </CommitViewerProvider>
                </FileViewerProvider>
            </RepositoryProvider>
        </DarkModeProvider>
    )
}

export default App

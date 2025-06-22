import { useState, useEffect } from 'react'
import axios from 'axios'
import LandingPage from './components/LandingPage'
import Dashboard from './components/Dashboard'
import LoadingSpinner from './components/LoadingSpinner'
import { DarkModeProvider } from './contexts/DarkModeContext'
import { RepositoryProvider } from './contexts/RepositoryContext'

function App() {
    const [user, setUser] = useState(null)
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        checkAuthStatus()
    }, [])

    const checkAuthStatus = async () => {
        try {
            const response = await axios.get('/auth/status')
            console.log('Auth status response:', response.data)
            if (response.data.authenticated) {
                setUser(response.data.user)
            }
        } catch (error) {
            console.log('Not authenticated:', error)
        } finally {
            setLoading(false)
        }
    }

    if (loading) {
        return <LoadingSpinner />
    }    return (
        <DarkModeProvider>
            <RepositoryProvider>
                <div className="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors">
                    {user ? (
                        <Dashboard
                            user={user}
                            onLogout={() => {
                                setUser(null)
                            }}
                        />
                    ) : (
                        <LandingPage />
                    )}
                </div>
            </RepositoryProvider>
        </DarkModeProvider>
    )
}

export default App

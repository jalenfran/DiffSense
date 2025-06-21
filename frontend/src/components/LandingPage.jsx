import { Github, ArrowRight, Zap, GitBranch, Search, AlertCircle, Moon, Sun } from 'lucide-react'
import { useState, useEffect } from 'react'
import { useDarkMode } from '../contexts/DarkModeContext'

function LandingPage() {
    const [oauthError, setOauthError] = useState(false)
    const [isRedirecting, setIsRedirecting] = useState(false)
    const { darkMode, toggleDarkMode } = useDarkMode()

    useEffect(() => {
        // Check if there's an OAuth error in the URL
        const urlParams = new URLSearchParams(window.location.search)
        if (urlParams.get('error') === 'oauth_failed') {
            setOauthError(true)
        }
    }, [])

    const handleGitHubLogin = () => {
        if (isRedirecting) return // Prevent multiple clicks
        setIsRedirecting(true)
        window.location.href = 'http://localhost:3000/auth/github'
    }
    return (
        <div className="min-h-screen bg-gradient-to-br from-primary-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 transition-colors">
            {/* Header */}
            <header className="relative overflow-hidden">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <div className="bg-primary-600 p-2 rounded-lg">
                                <GitBranch className="w-6 h-6 text-white" />
                            </div>
                            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">DiffSense</h1>
                        </div>
                        <button
                            onClick={toggleDarkMode}
                            className="text-gray-400 hover:text-gray-600 dark:text-gray-400 dark:hover:text-gray-200 p-2 rounded-lg hover:bg-white/20 dark:hover:bg-gray-700/50 transition-colors"
                            title={darkMode ? "Switch to light mode" : "Switch to dark mode"}
                        >
                            {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
                        </button>
                    </div>
                </div>
            </header>      {/* Hero Section */}
            <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="text-center py-20">
                    <h1 className="text-5xl md:text-6xl font-bold text-gray-900 dark:text-white mb-6">
                        Analyze Your
                        <span className="text-primary-600"> Git Diffs</span>
                        <br />
                        Like Never Before
                    </h1>
                    <p className="text-xl text-gray-600 dark:text-gray-300 mb-8 max-w-3xl mx-auto">
                        DiffSense provides powerful insights into your GitHub repositories by analyzing
                        code changes, tracking patterns, and helping you understand your development workflow.
                    </p>

                    {oauthError && (
                        <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6 max-w-md mx-auto">
                            <div className="flex items-center gap-2 text-red-800">
                                <AlertCircle className="w-5 h-5" />
                                <p className="font-medium">GitHub Authentication Failed</p>
                            </div>
                            <p className="text-red-600 text-sm mt-1">
                                Please check that the GitHub OAuth app is properly configured and try again.
                            </p>
                        </div>
                    )}
                    <button
                        onClick={handleGitHubLogin}
                        disabled={isRedirecting}
                        className={`inline-flex items-center gap-3 font-semibold py-4 px-8 rounded-xl transition-all duration-200 transform shadow-lg ${isRedirecting
                            ? 'bg-gray-600 cursor-not-allowed'
                            : 'bg-gray-900 hover:bg-gray-800 hover:scale-105'
                            } text-white`}
                    >
                        <Github className="w-6 h-6" />
                        {isRedirecting ? 'Redirecting...' : 'Continue with GitHub'}
                        <ArrowRight className="w-5 h-5" />
                    </button>
                </div>

                {/* Features */}
                <div className="py-20">
                    <div className="grid md:grid-cols-3 gap-8">
                        <div className="card text-center">
                            <div className="bg-primary-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                                <Search className="w-8 h-8 text-primary-600" />
                            </div>
                            <h3 className="text-xl font-semibold mb-3">Smart Analysis</h3>
                            <p className="text-gray-600">
                                Advanced algorithms analyze your code changes to provide meaningful insights
                                and patterns in your development process.
                            </p>
                        </div>

                        <div className="card text-center">
                            <div className="bg-primary-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                                <GitBranch className="w-8 h-8 text-primary-600" />
                            </div>
                            <h3 className="text-xl font-semibold mb-3">Repository Insights</h3>
                            <p className="text-gray-600">
                                Get detailed insights into your repositories including commit patterns,
                                file changes, and development trends over time.
                            </p>
                        </div>

                        <div className="card text-center">
                            <div className="bg-primary-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                                <Zap className="w-8 h-8 text-primary-600" />
                            </div>
                            <h3 className="text-xl font-semibold mb-3">Real-time Updates</h3>
                            <p className="text-gray-600">
                                Stay up-to-date with real-time analysis of your latest commits
                                and changes across all your repositories.
                            </p>
                        </div>
                    </div>
                </div>
            </main>

            {/* Footer */}
            <footer className="border-t border-gray-200 mt-20">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                    <div className="text-center text-gray-500">
                        <p>&copy; 2025 DiffSense. Analyze your code changes with confidence.</p>
                    </div>
                </div>
            </footer>
        </div>
    )
}

export default LandingPage

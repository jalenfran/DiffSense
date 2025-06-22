import { useState, useEffect } from 'react'
import {
    GitBranch,
    Star,
    GitFork,
    Clock,
    Code,
    ExternalLink,
    Activity,
    Calendar,
    Users,
    FileText,
    TrendingUp,
    Menu,
    User,
    LogOut,
    AlertTriangle,
    Shield,
    BarChart3
} from 'lucide-react'
import ChatInterface from './ChatInterface'
import { useRepository } from '../contexts/RepositoryContext'

function MainContent({ user, onToggleMobileSidebar, isMobileSidebarOpen }) {
    const { 
        selectedRepo, 
        repoStats, 
        repoFiles, 
        riskDashboard, 
        isLoading,
        error,
        messages,
        selectedFiles,
        isChatLoading,
        sendChatMessage,
        setSelectedFiles
    } = useRepository()

    const formatDate = (dateString) => {
        return new Date(dateString).toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        })
    }

    const getLanguageColor = (language) => {
        const colors = {
            'JavaScript': 'bg-yellow-400',
            'TypeScript': 'bg-blue-400',
            'Python': 'bg-blue-500',
            'Java': 'bg-orange-500',
            'C++': 'bg-pink-500',
            'C#': 'bg-purple-500',
            'Go': 'bg-cyan-400',
            'Rust': 'bg-orange-600',
            'PHP': 'bg-indigo-500',
            'Ruby': 'bg-red-500',
        }
        return colors[language] || 'bg-gray-400'
    }

    if (!selectedRepo) {
        return (
            <div className="flex-1 flex flex-col bg-gray-50 dark:bg-gray-900 duration-200 transition-colors">
                {/* Mobile Header with Hamburger Menu */}
                <div className="lg:hidden bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-4 sticky top-0 z-30">
                    <div className="flex items-center justify-between">
                        <button
                            onClick={onToggleMobileSidebar}
                            className="p-2 text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
                            title="Toggle sidebar"
                        >
                            <Menu className="w-6 h-6" />
                        </button>
                        
                        <div className="flex items-center gap-3">
                            <div className="bg-primary-600 p-2 rounded-lg">
                                <GitBranch className="w-5 h-5 text-white" />
                            </div>
                            <h1 className="text-lg font-bold text-gray-900 dark:text-white">DiffSense</h1>
                        </div>

                        <div className="flex items-center gap-2">
                            <div className="w-8 h-8 bg-gray-300 dark:bg-gray-600 rounded-full flex items-center justify-center">
                                {user?.photos && user.photos[0] ? (
                                    <img
                                        src={user.photos[0].value}
                                        alt={user.displayName || user.username}
                                        className="w-8 h-8 rounded-full"
                                    />
                                ) : (
                                    <User className="w-4 h-4 text-gray-600 dark:text-gray-300" />
                                )}
                            </div>
                        </div>
                    </div>
                </div>

                {/* Welcome Content */}
                <div className="flex-1 flex items-center justify-center p-4">
                    <div className="text-center">
                        <GitBranch className="w-16 h-16 text-gray-300 dark:text-gray-600 mx-auto mb-4" />
                        <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">Welcome to DiffSense</h2>
                        <p className="text-gray-600 dark:text-gray-400 max-w-md">
                            Select a repository from the sidebar to view detailed analysis and insights
                            about your code changes and development patterns.
                        </p>
                    </div>
                </div>
            </div>
        )
    }
    return (
        <div className="flex-1 flex flex-col bg-gray-50 dark:bg-gray-900 duration-200 transition-colors">
            {/* Mobile Header with Hamburger Menu */}
            <div className="lg:hidden bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-4 sticky top-0 z-30">
                <div className="flex items-center justify-between">
                    <button
                        onClick={onToggleMobileSidebar}
                        className="p-2 text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
                        title="Toggle sidebar"
                    >
                        <Menu className="w-6 h-6" />
                    </button>
                    
                    <div className="flex items-center gap-3">
                        <div className="bg-primary-600 p-2 rounded-lg">
                            <GitBranch className="w-5 h-5 text-white" />
                        </div>
                        <h1 className="text-lg font-bold text-gray-900 dark:text-white">DiffSense</h1>
                    </div>

                    <div className="flex items-center gap-2">
                        <div className="w-8 h-8 bg-gray-300 dark:bg-gray-600 rounded-full flex items-center justify-center">
                            {user?.photos && user.photos[0] ? (
                                <img
                                    src={user.photos[0].value}
                                    alt={user.displayName || user.username}
                                    className="w-8 h-8 rounded-full"
                                />
                            ) : (
                                <User className="w-4 h-4 text-gray-600 dark:text-gray-300" />
                            )}
                        </div>
                    </div>
                </div>
            </div>            {/* Main Content */}
            <div className="flex-1 overflow-y-auto">
                <div className="p-4 sm:p-6">
                    {/* Repository Header */}
                    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4 sm:p-6 mb-6">
                        <div className="flex flex-col sm:flex-row sm:items-start justify-between mb-4 gap-4">
                            <div className="flex-1">
                                <h1 className="text-xl sm:text-2xl font-bold text-gray-900 dark:text-white mb-2">
                                    {selectedRepo.name}
                                </h1>
                                <p className="text-gray-600 dark:text-gray-300 mb-4">
                                    {selectedRepo.description || 'No description available'}
                                </p>
                                <div className="flex flex-wrap items-center gap-4 text-sm text-gray-600 dark:text-gray-300">
                                    <div className="flex items-center gap-1">
                                        <Users className="w-4 h-4" />
                                        <span>{selectedRepo.owner.login}</span>
                                    </div>
                                    <div className="flex items-center gap-1">
                                        <Calendar className="w-4 h-4" />
                                        <span>Created {formatDate(selectedRepo.created_at)}</span>
                                    </div>
                                    <div className="flex items-center gap-1">
                                        <Clock className="w-4 h-4" />
                                        <span>Updated {formatDate(selectedRepo.updated_at)}</span>
                                    </div>
                                </div>
                            </div>
                            <a
                                href={selectedRepo.html_url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="flex items-center justify-center gap-2 px-4 py-2 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300 rounded-lg transition-colors text-sm font-medium whitespace-nowrap"
                            >
                                <ExternalLink className="w-4 h-4" />
                                View on GitHub
                            </a>
                        </div>

                        {/* Repository Stats */}
                        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                                <div className="flex items-center gap-2 mb-1">
                                    <Star className="w-4 h-4 text-yellow-500" />
                                    <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Stars</span>
                                </div>
                                <p className="text-2xl font-bold text-gray-900 dark:text-white">
                                    {selectedRepo.stargazers_count?.toLocaleString() || '0'}
                                </p>
                            </div>

                            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                                <div className="flex items-center gap-2 mb-1">
                                    <GitFork className="w-4 h-4 text-blue-500" />
                                    <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Forks</span>
                                </div>
                                <p className="text-2xl font-bold text-gray-900 dark:text-white">
                                    {selectedRepo.forks_count?.toLocaleString() || '0'}
                                </p>
                            </div>

                            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                                <div className="flex items-center gap-2 mb-1">
                                    <Code className="w-4 h-4 text-green-500" />
                                    <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Language</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    {selectedRepo.language && (
                                        <>
                                            <div className={`w-3 h-3 rounded-full ${getLanguageColor(selectedRepo.language)}`} />
                                            <span className="text-lg font-semibold text-gray-900 dark:text-white">
                                                {selectedRepo.language}
                                            </span>
                                        </>
                                    )}
                                    {!selectedRepo.language && (
                                        <span className="text-lg font-semibold text-gray-500 dark:text-gray-400">
                                            Unknown
                                        </span>
                                    )}
                                </div>
                            </div>

                            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                                <div className="flex items-center gap-2 mb-1">
                                    <FileText className="w-4 h-4 text-purple-500" />
                                    <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Files</span>
                                </div>
                                <p className="text-2xl font-bold text-gray-900 dark:text-white">
                                    {repoFiles?.length || 0}
                                </p>
                            </div>
                        </div>

                        {/* Error Display */}
                        {error && (
                            <div className="mt-4 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
                                <div className="flex items-center gap-2">
                                    <AlertTriangle className="w-5 h-5 text-red-500" />
                                    <span className="font-medium text-red-700 dark:text-red-300">Error</span>
                                </div>
                                <p className="text-sm text-red-600 dark:text-red-400 mt-1">{error}</p>
                            </div>
                        )}

                        {/* Loading Indicator */}
                        {isLoading && (
                            <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
                                <div className="flex items-center gap-2">
                                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
                                    <span className="text-blue-700 dark:text-blue-300">Analyzing repository...</span>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Risk Dashboard */}
                    {riskDashboard && (
                        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6 mb-6">
                            <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                                <Shield className="w-5 h-5 text-red-500" />
                                Risk Analysis
                            </h2>
                            
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                                <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                                    <div className="flex items-center gap-2 mb-2">
                                        <BarChart3 className="w-4 h-4 text-blue-500" />
                                        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Overall Risk</span>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <div className="w-20 bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                                            <div 
                                                className={`h-2 rounded-full ${
                                                    riskDashboard.overall_risk_score > 0.7 ? 'bg-red-500' : 
                                                    riskDashboard.overall_risk_score > 0.4 ? 'bg-yellow-500' : 'bg-green-500'
                                                }`}
                                                style={{ width: `${riskDashboard.overall_risk_score * 100}%` }}
                                            />
                                        </div>
                                        <span className="text-lg font-bold text-gray-900 dark:text-white">
                                            {Math.round(riskDashboard.overall_risk_score * 100)}%
                                        </span>
                                    </div>
                                </div>

                                <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                                    <div className="flex items-center gap-2 mb-2">
                                        <AlertTriangle className="w-4 h-4 text-red-500" />
                                        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">High Risk Commits</span>
                                    </div>
                                    <p className="text-2xl font-bold text-gray-900 dark:text-white">
                                        {riskDashboard.high_risk_commits}
                                    </p>
                                </div>

                                <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                                    <div className="flex items-center gap-2 mb-2">
                                        <GitBranch className="w-4 h-4 text-green-500" />
                                        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Commits Analyzed</span>
                                    </div>
                                    <p className="text-2xl font-bold text-gray-900 dark:text-white">
                                        {riskDashboard.total_commits_analyzed}
                                    </p>
                                </div>
                            </div>

                            {/* Breaking Changes by Type */}
                            {riskDashboard.breaking_changes_by_type && Object.keys(riskDashboard.breaking_changes_by_type).length > 0 && (
                                <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-4">
                                    <h3 className="font-medium text-red-700 dark:text-red-300 mb-3">Breaking Changes by Type</h3>
                                    <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                                        {Object.entries(riskDashboard.breaking_changes_by_type).map(([type, count]) => (
                                            <div key={type} className="flex justify-between items-center p-2 bg-white dark:bg-gray-800 rounded">
                                                <span className="text-sm text-gray-700 dark:text-gray-300 capitalize">
                                                    {type.replace('_', ' ')}
                                                </span>
                                                <span className="font-medium text-red-600 dark:text-red-400">{count}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    )}

                    {/* Chat Interface */}
                    <div className="h-96">
                        <ChatInterface
                            repoId={selectedRepo?.id}
                            onSendMessage={sendChatMessage}
                            messages={messages}
                            isLoading={isChatLoading}
                            onFileSelect={setSelectedFiles}
                            selectedFiles={selectedFiles}
                            availableFiles={repoFiles || []}
                        />
                    </div>
                </div>
            </div>
        </div>
    )
}

export default MainContent

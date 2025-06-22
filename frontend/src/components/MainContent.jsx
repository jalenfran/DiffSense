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
        setSelectedFiles    } = useRepository()

    // Chat maximize state
    const [isChatMaximized, setIsChatMaximized] = useState(false)

    const handleToggleChatMaximize = () => {
        setIsChatMaximized(!isChatMaximized)
    }    // Check if repository was added via URL (has placeholder data)
    const isUrlAddedRepo = selectedRepo?.description === 'Repository added via URL'

    // Handle escape key to close maximized chat and body scroll lock
    useEffect(() => {
        const handleEscape = (e) => {
            if (e.key === 'Escape' && isChatMaximized) {
                setIsChatMaximized(false)
            }
        }

        if (isChatMaximized) {
            document.body.style.overflow = 'hidden'
            document.addEventListener('keydown', handleEscape)
        } else {
            document.body.style.overflow = 'unset'
        }

        return () => {
            document.body.style.overflow = 'unset'
            document.removeEventListener('keydown', handleEscape)
        }
    }, [isChatMaximized])

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

    if (!selectedRepo) {        return (
            <div className="flex-1 flex flex-col bg-gray-50 dark:bg-gray-900 duration-200 transition-colors h-full overflow-hidden">
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
            </div>        )
    }
    return (
        <div className="flex-1 flex flex-col bg-gray-50 dark:bg-gray-900 duration-200 transition-colors relative h-full overflow-hidden">            {/* Chat Backdrop */}
            {isChatMaximized && (
                <div 
                    className="fixed inset-0 bg-black bg-opacity-50 z-40 transition-opacity duration-300"
                    onClick={handleToggleChatMaximize}
                />
            )}

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
            <div className="flex-1 overflow-y-auto bg-gray-50 dark:bg-gray-900">
                <div className="p-4 sm:p-6">
                    {/* Repository Header */}
                    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4 sm:p-6 mb-6">
                        <div className="flex flex-col sm:flex-row sm:items-start justify-between mb-4 gap-4">
                            <div className="flex-1">                                <h1 className="text-xl sm:text-2xl font-bold text-gray-900 dark:text-white mb-2">
                                    {selectedRepo.name}
                                </h1>
                                {!isUrlAddedRepo && selectedRepo.description && (
                                    <p className="text-gray-600 dark:text-gray-300 mb-4">
                                        {selectedRepo.description}
                                    </p>
                                )}
                                <div className="flex flex-wrap items-center gap-4 text-sm text-gray-600 dark:text-gray-300">
                                    <div className="flex items-center gap-1">
                                        <Users className="w-4 h-4" />
                                        <span>{selectedRepo.owner.login}</span>
                                    </div>
                                    {!isUrlAddedRepo && (
                                        <>
                                            <div className="flex items-center gap-1">
                                                <Calendar className="w-4 h-4" />
                                                <span>Created {formatDate(selectedRepo.created_at)}</span>
                                            </div>
                                            <div className="flex items-center gap-1">
                                                <Clock className="w-4 h-4" />
                                                <span>Updated {formatDate(selectedRepo.updated_at)}</span>
                                            </div>
                                        </>
                                    )}
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
                        </div>                        {/* Repository Stats */}
                        <div className={`grid gap-4 ${isUrlAddedRepo ? 'grid-cols-1 lg:grid-cols-2' : 'grid-cols-2 lg:grid-cols-4'}`}>
                            {!isUrlAddedRepo && (
                                <>
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
                                </>
                            )}

                            {selectedRepo.language && (
                                <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                                    <div className="flex items-center gap-2 mb-1">
                                        <Code className="w-4 h-4 text-green-500" />
                                        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Language</span>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <div className={`w-3 h-3 rounded-full ${getLanguageColor(selectedRepo.language)}`} />
                                        <span className="text-lg font-semibold text-gray-900 dark:text-white">
                                            {selectedRepo.language}
                                        </span>
                                    </div>
                                </div>
                            )}

                            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                                <div className="flex items-center gap-2 mb-1">
                                    <FileText className="w-4 h-4 text-purple-500" />
                                    <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Files Analyzed</span>
                                </div>
                                <p className="text-2xl font-bold text-gray-900 dark:text-white">
                                    {repoFiles?.length || 0}                                </p>
                            </div>
                        </div>

                        {/* URL Repository Info */}
                        {isUrlAddedRepo && (
                            <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
                                <p className="text-sm text-blue-700 dark:text-blue-300">
                                    <strong>Repository added via URL:</strong> Once analysis is complete, risk data and file information will be populated based on the actual repository content.
                                </p>
                            </div>
                        )}

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
                            </div>                        )}
                    </div>

                    {/* Chat Interface */}
                    <div className="h-[32rem] mb-6">
                        <ChatInterface
                            repoId={selectedRepo?.id}
                            onSendMessage={sendChatMessage}
                            messages={messages}
                            isLoading={isChatLoading}
                            onFileSelect={setSelectedFiles}
                            selectedFiles={selectedFiles}
                            availableFiles={repoFiles || []}
                            isMaximized={isChatMaximized}
                            onToggleMaximize={handleToggleChatMaximize}
                        />
                    </div>

                    {/* Risk Dashboard */}
                    {riskDashboard && (
                        <div className="space-y-6 mb-6">
                            {/* Overview Cards */}
                            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
                                <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                                    <Shield className="w-5 h-5 text-red-500" />
                                    Risk Analysis Overview
                                </h2>
                                
                                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
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
                                                    style={{ width: `${Math.max(riskDashboard.overall_risk_score * 100, 2)}%` }}
                                                />
                                            </div>
                                            <span className="text-lg font-bold text-gray-900 dark:text-white">
                                                {Math.round(riskDashboard.overall_risk_score * 100)}%
                                            </span>
                                        </div>
                                    </div>                                    <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                                        <div className="flex items-center gap-2 mb-2">
                                            <AlertTriangle className="w-4 h-4 text-orange-500" />
                                            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Elevated Risk Commits</span>
                                        </div>
                                        <p className="text-2xl font-bold text-gray-900 dark:text-white">
                                            {riskDashboard.risk_trend ? 
                                                riskDashboard.risk_trend.filter(commit => commit.risk_score > 0.25).length : 
                                                (riskDashboard.high_risk_commits || 0)
                                            }
                                        </p>
                                    </div>

                                    <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                                        <div className="flex items-center gap-2 mb-2">
                                            <GitBranch className="w-4 h-4 text-green-500" />
                                            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Commits Analyzed</span>
                                        </div>
                                        <p className="text-2xl font-bold text-gray-900 dark:text-white">
                                            {riskDashboard.total_commits_analyzed || 0}
                                        </p>                                    </div>
                                </div>
                            </div>

                            {/* Risk Distribution */}
                            {riskDashboard.risk_trend && riskDashboard.risk_trend.length > 0 && (
                                <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
                                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                                        <BarChart3 className="w-5 h-5 text-blue-500" />
                                        Risk Distribution
                                    </h3>
                                    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                                        {(() => {
                                            const commits = riskDashboard.risk_trend;
                                            const lowRisk = commits.filter(c => c.risk_score < 0.2).length;
                                            const mediumRisk = commits.filter(c => c.risk_score >= 0.2 && c.risk_score < 0.5).length;
                                            const highRisk = commits.filter(c => c.risk_score >= 0.5 && c.risk_score < 0.8).length;
                                            const criticalRisk = commits.filter(c => c.risk_score >= 0.8).length;
                                            
                                            return (
                                                <>
                                                    <div className="text-center p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
                                                        <div className="text-2xl font-bold text-green-600 dark:text-green-400">{lowRisk}</div>
                                                        <div className="text-sm text-green-700 dark:text-green-300">Low Risk</div>
                                                        <div className="text-xs text-green-600 dark:text-green-400">(&lt; 20%)</div>
                                                    </div>
                                                    <div className="text-center p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
                                                        <div className="text-2xl font-bold text-yellow-600 dark:text-yellow-400">{mediumRisk}</div>
                                                        <div className="text-sm text-yellow-700 dark:text-yellow-300">Medium Risk</div>
                                                        <div className="text-xs text-yellow-600 dark:text-yellow-400">(20-50%)</div>
                                                    </div>
                                                    <div className="text-center p-4 bg-orange-50 dark:bg-orange-900/20 rounded-lg border border-orange-200 dark:border-orange-800">
                                                        <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">{highRisk}</div>
                                                        <div className="text-sm text-orange-700 dark:text-orange-300">High Risk</div>
                                                        <div className="text-xs text-orange-600 dark:text-orange-400">(50-80%)</div>
                                                    </div>
                                                    <div className="text-center p-4 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
                                                        <div className="text-2xl font-bold text-red-600 dark:text-red-400">{criticalRisk}</div>
                                                        <div className="text-sm text-red-700 dark:text-red-300">Critical Risk</div>
                                                        <div className="text-xs text-red-600 dark:text-red-400">(≥ 80%)</div>
                                                    </div>
                                                </>
                                            );
                                        })()}
                                    </div>
                                </div>
                            )}

                            {/* Most Risky Files */}
                            {riskDashboard.most_risky_files && Object.keys(riskDashboard.most_risky_files).length > 0 && (
                                <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
                                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                                        <FileText className="w-5 h-5 text-orange-500" />
                                        Most Risky Files
                                    </h3>
                                    <div className="space-y-3">
                                        {Object.entries(riskDashboard.most_risky_files)
                                            .sort(([,a], [,b]) => b - a)
                                            .slice(0, 10)
                                            .map(([filePath, riskScore]) => (
                                            <div key={filePath} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                                                <div className="flex-1 min-w-0">
                                                    <p className="text-sm font-mono text-gray-900 dark:text-white truncate" title={filePath}>
                                                        {filePath}
                                                    </p>
                                                </div>
                                                <div className="flex items-center gap-3 ml-4">
                                                    <div className="w-24 bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                                                        <div 
                                                            className={`h-2 rounded-full ${
                                                                riskScore > 2 ? 'bg-red-500' : 
                                                                riskScore > 1 ? 'bg-yellow-500' : 'bg-green-500'
                                                            }`}
                                                            style={{ width: `${Math.min((riskScore / 5) * 100, 100)}%` }}
                                                        />
                                                    </div>
                                                    <span className="text-sm font-medium text-gray-900 dark:text-white w-12 text-right">
                                                        {riskScore.toFixed(2)}
                                                    </span>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>                            )}

                            {/* Risk Insights */}
                            {riskDashboard.risk_trend && riskDashboard.risk_trend.length > 0 && (
                                <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
                                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                                        <TrendingUp className="w-5 h-5 text-blue-500" />
                                        Risk Insights
                                    </h3>
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                        {(() => {
                                            const commits = riskDashboard.risk_trend;
                                            const avgRisk = commits.reduce((sum, c) => sum + c.risk_score, 0) / commits.length;
                                            const maxRisk = Math.max(...commits.map(c => c.risk_score));
                                            const maxRiskCommit = commits.find(c => c.risk_score === maxRisk);
                                            const recentCommits = commits.slice(0, 5);
                                            const avgRecentRisk = recentCommits.reduce((sum, c) => sum + c.risk_score, 0) / recentCommits.length;
                                            
                                            return (
                                                <>
                                                    <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
                                                        <div className="flex items-center gap-2 mb-2">
                                                            <BarChart3 className="w-4 h-4 text-blue-500" />
                                                            <span className="text-sm font-medium text-blue-700 dark:text-blue-300">Average Risk Score</span>
                                                        </div>
                                                        <div className="text-2xl font-bold text-blue-900 dark:text-blue-100">
                                                            {Math.round(avgRisk * 100)}%
                                                        </div>
                                                        <div className="text-xs text-blue-600 dark:text-blue-400">
                                                            Across {commits.length} commits
                                                        </div>
                                                    </div>
                                                    
                                                    <div className="p-4 bg-orange-50 dark:bg-orange-900/20 rounded-lg border border-orange-200 dark:border-orange-800">
                                                        <div className="flex items-center gap-2 mb-2">
                                                            <AlertTriangle className="w-4 h-4 text-orange-500" />
                                                            <span className="text-sm font-medium text-orange-700 dark:text-orange-300">Highest Risk Commit</span>
                                                        </div>
                                                        <div className="text-2xl font-bold text-orange-900 dark:text-orange-100">
                                                            {Math.round(maxRisk * 100)}%
                                                        </div>
                                                        <div className="text-xs text-orange-600 dark:text-orange-400 font-mono">
                                                            {maxRiskCommit?.commit_hash.substring(0, 8)}...
                                                        </div>
                                                    </div>
                                                </>
                                            );
                                        })()}
                                    </div>
                                </div>
                            )}

                            {/* Risk Trend */}
                            {riskDashboard.risk_trend && riskDashboard.risk_trend.length > 0 && (
                                <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
                                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                                        <TrendingUp className="w-5 h-5 text-blue-500" />
                                        Recent Commit Risk Trend
                                    </h3>
                                    <div className="space-y-2">
                                        {riskDashboard.risk_trend.slice(0, 10).map((commit, index) => (
                                            <div key={commit.commit_hash} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                                                <div className="flex items-center gap-3">
                                                    <span className="text-xs font-mono text-gray-500 dark:text-gray-400">
                                                        {commit.commit_hash}
                                                    </span>
                                                    <span className="text-xs text-gray-600 dark:text-gray-300">
                                                        {new Date(commit.timestamp).toLocaleDateString()} {new Date(commit.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                                                    </span>
                                                </div>
                                                <div className="flex items-center gap-3">
                                                    <div className="w-20 bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                                                        <div 
                                                            className={`h-2 rounded-full ${
                                                                commit.risk_score > 0.7 ? 'bg-red-500' : 
                                                                commit.risk_score > 0.4 ? 'bg-yellow-500' : 'bg-green-500'
                                                            }`}
                                                            style={{ width: `${Math.max(commit.risk_score * 100, 2)}%` }}
                                                        />
                                                    </div>
                                                    <span className="text-sm font-medium text-gray-900 dark:text-white w-12 text-right">
                                                        {Math.round(commit.risk_score * 100)}%
                                                    </span>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Breaking Changes by Type */}
                            {riskDashboard.breaking_changes_by_type && Object.keys(riskDashboard.breaking_changes_by_type).length > 0 && (
                                <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
                                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                                        <AlertTriangle className="w-5 h-5 text-red-500" />
                                        Breaking Changes by Type
                                    </h3>
                                    <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                                        {Object.entries(riskDashboard.breaking_changes_by_type).map(([type, count]) => (
                                            <div key={type} className="flex justify-between items-center p-3 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
                                                <span className="text-sm font-medium text-gray-700 dark:text-gray-300 capitalize">
                                                    {type.replace(/_/g, ' ')}
                                                </span>
                                                <span className="text-lg font-bold text-red-600 dark:text-red-400">{count}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}                            {/* Notable Risk Commits */}
                            {riskDashboard.risk_trend && riskDashboard.risk_trend.filter(c => c.risk_score > 0.2).length > 0 && (
                                <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
                                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                                        <AlertTriangle className="w-5 h-5 text-orange-500" />
                                        Notable Risk Commits
                                    </h3>
                                    <div className="space-y-3">
                                        {riskDashboard.risk_trend
                                            .filter(commit => commit.risk_score > 0.2)
                                            .sort((a, b) => b.risk_score - a.risk_score)
                                            .slice(0, 10)
                                            .map((commit, index) => (
                                            <div key={commit.commit_hash || index} className={`p-3 rounded-lg border ${
                                                commit.risk_score > 0.5 ? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800' :
                                                'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800'
                                            }`}>
                                                <div className="flex items-center justify-between mb-2">
                                                    <span className="text-sm font-mono text-gray-900 dark:text-white">
                                                        {commit.commit_hash}
                                                    </span>
                                                    <div className="flex items-center gap-2">
                                                        <span className={`text-sm font-bold ${
                                                            commit.risk_score > 0.5 ? 'text-red-600 dark:text-red-400' :
                                                            'text-yellow-600 dark:text-yellow-400'
                                                        }`}>
                                                            {Math.round(commit.risk_score * 100)}% Risk
                                                        </span>
                                                    </div>
                                                </div>
                                                <div className="text-xs text-gray-500 dark:text-gray-400">
                                                    {new Date(commit.timestamp).toLocaleDateString()} {new Date(commit.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                                                </div>
                                                {commit.message && (
                                                    <p className="text-sm text-gray-600 dark:text-gray-300 mt-2 truncate">
                                                        {commit.message}
                                                    </p>
                                                )}
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Legacy High Risk Commits (fallback) */}
                            {riskDashboard.recent_high_risk_commits && riskDashboard.recent_high_risk_commits.length > 0 && (
                                <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
                                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                                        <AlertTriangle className="w-5 h-5 text-red-500" />
                                        High Risk Commits
                                    </h3>
                                    <div className="space-y-3">
                                        {riskDashboard.recent_high_risk_commits.map((commit, index) => (
                                            <div key={commit.commit_hash || index} className="p-3 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
                                                <div className="flex items-center justify-between mb-2">
                                                    <span className="text-sm font-mono text-gray-900 dark:text-white">
                                                        {commit.commit_hash}
                                                    </span>
                                                    <span className="text-sm font-bold text-red-600 dark:text-red-400">
                                                        {Math.round(commit.risk_score * 100)}% Risk
                                                    </span>
                                                </div>
                                                {commit.message && (
                                                    <p className="text-sm text-gray-600 dark:text-gray-300 truncate">
                                                        {commit.message}
                                                    </p>
                                                )}
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}                            {/* Empty State */}
                            {(!riskDashboard.risk_trend || riskDashboard.risk_trend.length === 0) && 
                             (!riskDashboard.most_risky_files || Object.keys(riskDashboard.most_risky_files).length === 0) && 
                             (!riskDashboard.breaking_changes_by_type || Object.keys(riskDashboard.breaking_changes_by_type).length === 0) &&
                             (!riskDashboard.recent_high_risk_commits || riskDashboard.recent_high_risk_commits.length === 0) && (
                                <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-8">
                                    <div className="text-center">
                                        <Shield className="w-12 h-12 text-gray-400 dark:text-gray-500 mx-auto mb-3" />
                                        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                                            No Risk Data Available
                                        </h3>
                                        <p className="text-gray-600 dark:text-gray-400">
                                            Repository analysis is still in progress or no commits have been analyzed yet.
                                        </p>
                                    </div>
                                </div>                            )}
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}

export default MainContent

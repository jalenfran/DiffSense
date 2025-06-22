import { useState, useEffect } from 'react'
import {
    GitBranch,
    Clock,
    ExternalLink,
    Activity,
    Users,
    FileText,
    TrendingUp,
    Menu,
    User,
    LogOut,
    AlertTriangle,
    Shield,
    BarChart3,
    MessageSquare
} from 'lucide-react'
import ChatInterface from './ChatInterface'
import LinkifyContent from './LinkifyContent'
import BreakingChangeAnalyzer from './BreakingChangeAnalyzer'
import FileExplorer from './FileExplorer'
import { useRepository } from '../contexts/RepositoryContext'

function MainContent({ user, onToggleMobileSidebar, isMobileSidebarOpen }) {
    const {
        selectedRepo,
        repoId,
        repoStats,
        repoFiles,
        riskDashboard,
        isLoading,
        error,
        messages,
        selectedFiles,
        isChatLoading,
        sendChatMessage,
        setSelectedFiles } = useRepository()    // Chat maximize state
    const [isChatMaximized, setIsChatMaximized] = useState(false)
    const [isFileExplorerMaximized, setIsFileExplorerMaximized] = useState(false)
      // Tab state for main content
    const [activeTab, setActiveTab] = useState('overview') // 'overview', 'analysis', 'files', 'risk'

    const handleToggleChatMaximize = () => {
        setIsChatMaximized(!isChatMaximized)
    }
    
    const handleToggleFileExplorerMaximize = () => {
        setIsFileExplorerMaximized(!isFileExplorerMaximized)
    }// Check if repository was added via URL (has placeholder data)
    const isUrlAddedRepo = selectedRepo?.description === 'Repository added via URL'    // Handle escape key to close maximized chat and body scroll lock
    useEffect(() => {
        const handleEscape = (e) => {
            if (e.key === 'Escape') {
                if (isChatMaximized) {
                    setIsChatMaximized(false)
                }
                if (isFileExplorerMaximized) {
                    setIsFileExplorerMaximized(false)
                }
            }
        }

        if (isChatMaximized || isFileExplorerMaximized) {
            document.body.style.overflow = 'hidden'
            document.addEventListener('keydown', handleEscape)
        } else {
            document.body.style.overflow = 'unset'
        }

        return () => {
            document.body.style.overflow = 'unset'
            document.removeEventListener('keydown', handleEscape)
        }
    }, [isChatMaximized, isFileExplorerMaximized])

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
                                {user?.github_avatar ? (
                                    <img
                                        src={user.github_avatar}
                                        alt={user.display_name || user.github_username}
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
                            Select a repository from the sidebar to get started.
                        </p>
                    </div>
                </div>
            </div>)
    }
    return (
        <div className="flex-1 flex flex-col bg-gray-50 dark:bg-gray-900 duration-200 transition-colors relative h-full overflow-hidden">            {/* Chat Backdrop */}
            {isChatMaximized && (
                <div
                    className="fixed inset-0 bg-black bg-opacity-50 z-40 transition-opacity duration-300"
                    onClick={handleToggleChatMaximize}
                />
            )}
            
            {/* File Explorer Backdrop */}
            {isFileExplorerMaximized && (
                <div
                    className="fixed inset-0 bg-black bg-opacity-50 z-40 transition-opacity duration-300"
                    onClick={handleToggleFileExplorerMaximize}
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
                            {user?.github_avatar ? (
                                <img
                                    src={user.github_avatar}
                                    alt={user.display_name || user.github_username}
                                    className="w-8 h-8 rounded-full"
                                />
                            ) : (
                                <User className="w-4 h-4 text-gray-600 dark:text-gray-300" />
                            )}
                        </div>
                    </div>
                </div>
            </div>            {/* Main Content */}
            <div className="flex-1 flex flex-col overflow-hidden bg-gray-50 dark:bg-gray-900">
                <div className="flex-1 flex flex-col p-4 sm:p-6 min-h-0">                    {/* Repository Header - Condensed */}
                    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4 mb-6 flex-shrink-0">
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-4 min-w-0 flex-1">
                                <div className="min-w-0 flex-1">
                                    <div className="flex items-center gap-3 mb-1">
                                        <h1 className="text-lg font-bold text-gray-900 dark:text-white truncate">
                                            {selectedRepo.name}
                                        </h1>
                                        {selectedRepo.language && (
                                            <div className="flex items-center gap-1.5 text-sm">
                                                <div className={`w-2.5 h-2.5 rounded-full ${getLanguageColor(selectedRepo.language)}`} />
                                                <span className="text-gray-600 dark:text-gray-400 font-medium">
                                                    {selectedRepo.language}
                                                </span>
                                            </div>
                                        )}
                                    </div>
                                    <div className="flex items-center gap-4 text-sm text-gray-500 dark:text-gray-400">
                                        <div className="flex items-center gap-1">
                                            <Users className="w-3.5 h-3.5" />
                                            <span>{selectedRepo.owner.login}</span>
                                        </div>
                                        {!isUrlAddedRepo && selectedRepo.updated_at && (
                                            <div className="flex items-center gap-1">
                                                <Clock className="w-3.5 h-3.5" />
                                                <span>Updated {formatDate(selectedRepo.updated_at)}</span>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>
                            <a
                                href={selectedRepo.html_url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="flex items-center gap-2 px-3 py-1.5 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300 rounded-md transition-colors text-sm font-medium whitespace-nowrap ml-4"
                            >
                                <ExternalLink className="w-3.5 h-3.5" />
                                GitHub
                            </a>                        </div>

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
                            </div>)}
                    </div>                    {/* Tab Navigation */}
                    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 mb-6 flex-shrink-0">
                        <div className="border-b border-gray-200 dark:border-gray-700">
                            <nav className="flex space-x-8 px-6" aria-label="Tabs">                                <button
                                    onClick={() => setActiveTab('overview')}
                                    className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                                        activeTab === 'overview'
                                            ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                                            : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
                                    }`}
                                >
                                    <div className="flex items-center gap-2">
                                        <MessageSquare className="w-4 h-4" />
                                        Chat
                                    </div>
                                </button>
                                <button
                                    onClick={() => setActiveTab('analysis')}
                                    className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                                        activeTab === 'analysis'
                                            ? 'border-orange-500 text-orange-600 dark:text-orange-400'
                                            : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
                                    }`}
                                >                                    <div className="flex items-center gap-2">
                                        <AlertTriangle className="w-4 h-4" />
                                        Breaking Changes
                                    </div>
                                </button>
                                <button
                                    onClick={() => setActiveTab('risk')}
                                    className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                                        activeTab === 'risk'
                                            ? 'border-red-500 text-red-600 dark:text-red-400'
                                            : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
                                    }`}
                                >
                                    <div className="flex items-center gap-2">
                                        <Shield className="w-4 h-4" />
                                        Risk Dashboard
                                    </div>
                                </button>
                                <button
                                    onClick={() => setActiveTab('files')}
                                    className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                                        activeTab === 'files'
                                            ? 'border-green-500 text-green-600 dark:text-green-400'
                                            : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
                                    }`}
                                >
                                    <div className="flex items-center gap-2">
                                        <FileText className="w-4 h-4" />
                                        Files
                                    </div>
                                </button>
                            </nav>
                        </div>
                    </div>                    {/* Tab Content */}
                    <div className="flex-1 flex flex-col min-h-0">                        {activeTab === 'overview' && (
                            <div className="flex-1 flex flex-col min-h-0">
                                {/* Chat Interface */}
                                <div className="flex-1 min-h-0">                                    <ChatInterface
                                        repoId={repoId}
                                        onSendMessage={sendChatMessage}
                                        messages={messages}
                                        isLoading={isChatLoading}
                                        onFileSelect={setSelectedFiles}
                                        selectedFiles={selectedFiles}
                                        availableFiles={repoFiles || []}
                                        isMaximized={isChatMaximized}
                                        onToggleMaximize={handleToggleChatMaximize}
                                        user={user}
                                    />
                                </div>
                            </div>
                        )}                        {activeTab === 'analysis' && (
                            <div className="flex-1 flex flex-col min-h-0 overflow-y-auto">
                                {/* Breaking Change Analysis */}
                                <BreakingChangeAnalyzer 
                                    repoId={repoId} 
                                    repoName={selectedRepo.name} 
                                />
                            </div>
                        )}{activeTab === 'files' && (
                            <div className="flex-1 flex flex-col min-h-0">
                                {/* File Explorer - Give it full height */}
                                <div className="flex-1 min-h-0">
                                    <FileExplorer
                                        repoId={repoId}
                                        repoName={selectedRepo.name}
                                        isMaximized={isFileExplorerMaximized}
                                        onToggleMaximize={handleToggleFileExplorerMaximize}
                                    />
                                </div>
                            </div>
                        )}

                        {activeTab === 'risk' && (
                            <div className="flex-1 flex flex-col min-h-0 overflow-y-auto">
                                {/* Risk Dashboard */}                                {riskDashboard && (
                                <div className="space-y-6 mb-6">
                                    {/* Overview Stats */}
                                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                                        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
                                            <div className="flex items-center justify-between">
                                                <div>
                                                    <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Overall Risk</p>
                                                    <p className="text-2xl font-bold text-gray-900 dark:text-white">
                                                        {Math.round((riskDashboard.overall_risk_score || 0) * 100)}%
                                                    </p>
                                                </div>
                                                <div className={`p-3 rounded-full ${
                                                    (riskDashboard.overall_risk_score || 0) > 0.7 ? 'bg-red-100 dark:bg-red-900/20' :
                                                    (riskDashboard.overall_risk_score || 0) > 0.4 ? 'bg-yellow-100 dark:bg-yellow-900/20' : 'bg-green-100 dark:bg-green-900/20'
                                                }`}>
                                                    <Shield className={`w-6 h-6 ${
                                                        (riskDashboard.overall_risk_score || 0) > 0.7 ? 'text-red-600 dark:text-red-400' :
                                                        (riskDashboard.overall_risk_score || 0) > 0.4 ? 'text-yellow-600 dark:text-yellow-400' : 'text-green-600 dark:text-green-400'
                                                    }`} />
                                                </div>
                                            </div>
                                            <div className="mt-4 w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                                                <div
                                                    className={`h-2 rounded-full ${
                                                        (riskDashboard.overall_risk_score || 0) > 0.7 ? 'bg-red-500' :
                                                        (riskDashboard.overall_risk_score || 0) > 0.4 ? 'bg-yellow-500' : 'bg-green-500'
                                                    }`}
                                                    style={{ width: `${Math.max((riskDashboard.overall_risk_score || 0) * 100, 2)}%` }}
                                                />
                                            </div>
                                        </div>

                                        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
                                            <div className="flex items-center justify-between">
                                                <div>
                                                    <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Commits Analyzed</p>
                                                    <p className="text-2xl font-bold text-gray-900 dark:text-white">
                                                        {riskDashboard.total_commits_analyzed || 0}
                                                    </p>
                                                </div>
                                                <div className="p-3 bg-blue-100 dark:bg-blue-900/20 rounded-full">
                                                    <BarChart3 className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                                                </div>
                                            </div>
                                            <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                                                {riskDashboard.high_risk_commits || 0} high risk, {riskDashboard.critical_risk_commits || 0} critical
                                            </p>
                                        </div>

                                        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
                                            <div className="flex items-center justify-between">
                                                <div>
                                                    <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Breaking Changes</p>
                                                    <p className="text-2xl font-bold text-gray-900 dark:text-white">
                                                        {Object.values(riskDashboard.breaking_changes_by_type || {}).reduce((a, b) => a + b, 0)}
                                                    </p>
                                                </div>
                                                <div className="p-3 bg-red-100 dark:bg-red-900/20 rounded-full">
                                                    <AlertTriangle className="w-6 h-6 text-red-600 dark:text-red-400" />
                                                </div>
                                            </div>
                                            <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                                                {riskDashboard.intentional_breaking_changes || 0} intentional, {riskDashboard.accidental_breaking_changes || 0} accidental
                                            </p>
                                        </div>

                                        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
                                            <div className="flex items-center justify-between">
                                                <div>
                                                    <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Avg Confidence</p>
                                                    <p className="text-2xl font-bold text-gray-900 dark:text-white">
                                                        {Math.round((riskDashboard.advanced_insights?.average_confidence_score || 0) * 100)}%
                                                    </p>
                                                </div>
                                                <div className="p-3 bg-purple-100 dark:bg-purple-900/20 rounded-full">
                                                    <TrendingUp className="w-6 h-6 text-purple-600 dark:text-purple-400" />
                                                </div>
                                            </div>
                                            <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                                                Detection accuracy
                                            </p>
                                        </div>
                                    </div>

                                    {/* Risk Percentages Grid */}
                                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                                        {/* Breaking Changes Breakdown */}
                                        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
                                            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                                                <AlertTriangle className="w-5 h-5 text-red-500" />
                                                Breaking Changes Analysis
                                            </h3>
                                            
                                            {/* By Severity */}
                                            <div className="mb-6">
                                                <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">By Severity</h4>
                                                <div className="space-y-2">
                                                    {Object.entries(riskDashboard.breaking_changes_percentages?.by_severity || {}).map(([severity, data]) => (
                                                        <div key={severity} className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-700 rounded">
                                                            <span className="text-sm capitalize text-gray-700 dark:text-gray-300">{severity}</span>
                                                            <div className="flex items-center gap-2">
                                                                <span className="text-sm font-medium text-gray-900 dark:text-white">{data.count}</span>
                                                                <span className="text-xs text-gray-500 dark:text-gray-400">
                                                                    ({Math.round(data.percentage * 100)}%)
                                                                </span>
                                                            </div>
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>

                                            {/* By Type */}
                                            <div>
                                                <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">By Type</h4>
                                                <div className="space-y-2">
                                                    {Object.entries(riskDashboard.breaking_changes_percentages?.by_type || {}).map(([type, data]) => (
                                                        <div key={type} className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-700 rounded">
                                                            <span className="text-sm capitalize text-gray-700 dark:text-gray-300">
                                                                {type.replace(/_/g, ' ')}
                                                            </span>
                                                            <div className="flex items-center gap-2">
                                                                <span className="text-sm font-medium text-gray-900 dark:text-white">{data.count}</span>
                                                                <span className="text-xs text-gray-500 dark:text-gray-400">
                                                                    ({Math.round(data.percentage * 100)}%)
                                                                </span>
                                                            </div>
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>
                                        </div>

                                        {/* Commit Risk Distribution */}
                                        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
                                            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                                                <BarChart3 className="w-5 h-5 text-blue-500" />
                                                Commit Risk Distribution
                                            </h3>
                                              <div className="space-y-4">
                                                {Object.entries(riskDashboard.commit_risk_percentages || {}).map(([riskLevel, data]) => {
                                                    if (typeof data !== 'object' || !data.count) return null;
                                                    
                                                    const color = riskLevel.includes('critical') ? 'red' : 
                                                                 riskLevel.includes('high') ? 'orange' : 'green';
                                                    
                                                    // Handle both decimal (0.39) and percentage (39) formats from backend
                                                    const percentage = data.percentage > 1 ? data.percentage : data.percentage * 100;
                                                    const progressWidth = data.percentage > 1 ? data.percentage : data.percentage * 100;
                                                    
                                                    return (
                                                        <div key={riskLevel} className="space-y-2">
                                                            <div className="flex items-center justify-between">
                                                                <span className="text-sm font-medium text-gray-700 dark:text-gray-300 capitalize">
                                                                    {riskLevel.replace(/_/g, ' ')}
                                                                </span>
                                                                <span className="text-sm text-gray-600 dark:text-gray-400">
                                                                    {data.count} ({Math.round(percentage)}%)
                                                                </span>
                                                            </div>
                                                            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                                                                <div 
                                                                    className={`h-2 rounded-full bg-${color}-500`}
                                                                    style={{ width: `${Math.min(progressWidth, 100)}%` }}
                                                                />
                                                            </div>
                                                        </div>
                                                    );
                                                })}
                                            </div>
                                        </div>                                    </div>

                                    {/* Advanced Insights - Reorganized Layout */}
                                    {riskDashboard.advanced_insights && (
                                        <>
                                            {/* Files with Most Breaking Changes - Full Width */}
                                            {riskDashboard.advanced_insights.files_with_most_breaking_changes && 
                                             Object.keys(riskDashboard.advanced_insights.files_with_most_breaking_changes).length > 0 && (
                                                <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
                                                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                                                        <FileText className="w-5 h-5 text-yellow-500" />
                                                        Most Affected Files
                                                    </h3>
                                                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                                                        {Object.entries(riskDashboard.advanced_insights.files_with_most_breaking_changes)
                                                            .slice(0, 6)
                                                            .map(([file, count]) => (
                                                            <div key={file} className="flex items-center justify-between p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
                                                                <span className="text-sm font-mono text-gray-900 dark:text-white truncate mr-4 flex-1">
                                                                    <LinkifyContent repoId={selectedRepo?.repo_id}>
                                                                        {file}
                                                                    </LinkifyContent>
                                                                </span>
                                                                <span className="text-sm font-bold text-yellow-700 dark:text-yellow-400">
                                                                    {count} changes
                                                                </span>
                                                            </div>
                                                        ))}
                                                    </div>
                                                </div>
                                            )}
                                        </>
                                    )}                                    {/* High Risk Commits - Improved Layout */}
                                    {riskDashboard.recent_high_risk_commits && riskDashboard.recent_high_risk_commits.length > 0 && (
                                        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
                                            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                                                <AlertTriangle className="w-5 h-5 text-red-500" />
                                                High Risk Commits
                                            </h3>
                                            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                                                {riskDashboard.recent_high_risk_commits.slice(0, 12).map((commit, index) => (
                                                    <div key={commit.commit_hash || index} className="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
                                                        <div className="flex items-center justify-between mb-2">
                                                            <span className="text-sm font-mono text-gray-900 dark:text-white">
                                                                <LinkifyContent repoId={selectedRepo?.repo_id}>
                                                                    {commit.commit_hash?.substring(0, 8)}
                                                                </LinkifyContent>
                                                            </span>
                                                            <span className="text-sm font-bold text-red-600 dark:text-red-400">
                                                                {Math.round((commit.risk_score || 0) * 100)}%
                                                            </span>
                                                        </div>
                                                        <div className="flex items-center justify-between text-xs text-gray-600 dark:text-gray-400">
                                                            <span>{commit.breaking_changes_count || 0} breaking changes</span>
                                                            <div className="w-16 bg-gray-200 dark:bg-gray-600 rounded-full h-1">
                                                                <div
                                                                    className="h-1 rounded-full bg-red-500"
                                                                    style={{ width: `${(commit.risk_score || 0) * 100}%` }}
                                                                />
                                                            </div>
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    {/* Performance Stats */}
                                    {riskDashboard.performance_stats && (
                                        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
                                            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                                                <Activity className="w-5 h-5 text-green-500" />
                                                Analysis Performance
                                            </h3>
                                            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                                <div className="text-center p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                                                    <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                                                        {riskDashboard.performance_stats.new_commits_analyzed || 0}
                                                    </div>
                                                    <div className="text-sm text-green-600 dark:text-green-400">New Commits</div>
                                                </div>
                                                <div className="text-center p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                                                    <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                                                        {riskDashboard.performance_stats.cached_analyses_used || 0}
                                                    </div>
                                                    <div className="text-sm text-blue-600 dark:text-blue-400">Cache Hits</div>
                                                </div>
                                                <div className="text-center p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                                                    <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                                                        {Math.round((riskDashboard.performance_stats.cache_hit_rate || 0) * 100)}%
                                                    </div>
                                                    <div className="text-sm text-purple-600 dark:text-purple-400">Cache Rate</div>
                                                </div>
                                            </div>
                                        </div>
                                    )}

                                    {/* Empty State */}
                                    {(!riskDashboard.risk_trend || riskDashboard.risk_trend.length === 0) &&
                                        (!riskDashboard.advanced_insights?.files_with_most_breaking_changes || Object.keys(riskDashboard.advanced_insights.files_with_most_breaking_changes).length === 0) &&
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
                                            </div>
                                    )}
                                </div>
                            )}
                        </div>
                    )}
                    </div>
                </div>
            </div>
        </div>
    )
}

export default MainContent

import React, { useState, useEffect } from 'react'
import { AlertTriangle, Clock, GitCommit, FileText, Users, Wrench, ChevronDown, ChevronUp, ExternalLink, Copy, Calendar, User, X, Loader2 } from 'lucide-react'
import { diffSenseAPI } from '../services/api'
import LoadingSpinner from './LoadingSpinner'
import MarkdownRenderer from './MarkdownRenderer'

const SEVERITY_COLORS = {
    critical: 'bg-red-100 text-red-800 border-red-200 dark:bg-red-900/20 dark:text-red-400 dark:border-red-800',
    high: 'bg-orange-100 text-orange-800 border-orange-200 dark:bg-orange-900/20 dark:text-orange-400 dark:border-orange-800',
    medium: 'bg-yellow-100 text-yellow-800 border-yellow-200 dark:bg-yellow-900/20 dark:text-yellow-400 dark:border-yellow-800',
    low: 'bg-blue-100 text-blue-800 border-blue-200 dark:bg-blue-900/20 dark:text-blue-400 dark:border-blue-800'
}

const SEVERITY_ICONS = {
    critical: <AlertTriangle className="w-4 h-4" />,
    high: <AlertTriangle className="w-4 h-4" />,
    medium: <Clock className="w-4 h-4" />,
    low: <FileText className="w-4 h-4" />
}

const COMPLEXITY_COLORS = {
    'very_complex': 'text-red-600 dark:text-red-400',
    'complex': 'text-orange-600 dark:text-orange-400',
    'moderate': 'text-yellow-600 dark:text-yellow-400',
    'easy': 'text-green-600 dark:text-green-400',
    'trivial': 'text-blue-600 dark:text-blue-400'
}

function BreakingChangeCard({ change, repoId }) {
    const [isExpanded, setIsExpanded] = useState(false)
    
    const severityColor = SEVERITY_COLORS[change.severity] || SEVERITY_COLORS.medium
    const severityIcon = SEVERITY_ICONS[change.severity] || SEVERITY_ICONS.medium
    const complexityColor = COMPLEXITY_COLORS[change.migration_complexity] || 'text-gray-600'

    const copyToClipboard = (text) => {
        navigator.clipboard.writeText(text)
    }

    return (
        <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 mb-4 bg-white dark:bg-gray-800">
            <div className="flex items-start justify-between">
                <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                        <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium border ${severityColor}`}>
                            {severityIcon}
                            {change.severity}
                        </span>
                        <span className="text-xs text-gray-500 dark:text-gray-400">
                            {change.change_type}
                        </span>
                        <span className={`text-xs font-medium ${complexityColor}`}>
                            {change.migration_complexity} migration
                        </span>
                    </div>
                    
                    <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-1">
                        {change.affected_component}
                    </h4>
                    
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                        {change.description}
                    </p>
                    
                    <div className="flex items-center gap-4 text-xs text-gray-500 dark:text-gray-400">
                        <span className="flex items-center gap-1">
                            <FileText className="w-3 h-3" />
                            {change.file_path}
                        </span>
                        <span className="flex items-center gap-1">
                            <Users className="w-3 h-3" />
                            Affects {change.affected_users_estimate} users
                        </span>
                        <span className="flex items-center gap-1">
                            <Wrench className="w-3 h-3" />
                            {Math.round(change.confidence_score * 100)}% confidence
                        </span>
                    </div>
                </div>
                
                <button
                    onClick={() => setIsExpanded(!isExpanded)}
                    className="ml-4 p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                >
                    {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </button>
            </div>
            
            {isExpanded && (
                <div className="mt-4 border-t border-gray-200 dark:border-gray-700 pt-4">
                    <div className="grid md:grid-cols-2 gap-4">
                        <div>
                            <h5 className="font-medium text-gray-900 dark:text-gray-100 mb-2">Technical Details</h5>
                            <div className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                                {change.technical_details}
                            </div>
                            
                            {change.old_signature && (
                                <div className="mb-3">
                                    <div className="flex items-center justify-between mb-1">
                                        <span className="text-xs font-medium text-gray-700 dark:text-gray-300">Old Signature</span>
                                        <button
                                            onClick={() => copyToClipboard(change.old_signature)}
                                            className="text-xs text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                                        >
                                            <Copy className="w-3 h-3" />
                                        </button>
                                    </div>                                    <code className="block text-xs bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200 p-2 rounded font-mono">
                                        {change.old_signature}
                                    </code>
                                </div>
                            )}
                            
                            {change.new_signature && (
                                <div className="mb-3">
                                    <div className="flex items-center justify-between mb-1">
                                        <span className="text-xs font-medium text-gray-700 dark:text-gray-300">New Signature</span>
                                        <button
                                            onClick={() => copyToClipboard(change.new_signature)}
                                            className="text-xs text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                                        >
                                            <Copy className="w-3 h-3" />
                                        </button>
                                    </div>                                    <code className="block text-xs bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200 p-2 rounded font-mono">
                                        {change.new_signature}
                                    </code>
                                </div>
                            )}
                        </div>
                        
                        <div>
                            <h5 className="font-medium text-gray-900 dark:text-gray-100 mb-2">Migration Guide</h5>
                            {change.suggested_migration && (
                                <div className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                                    {change.suggested_migration}
                                </div>
                            )}
                            
                            {change.expert_recommendations && change.expert_recommendations.length > 0 && (
                                <div>
                                    <span className="text-xs font-medium text-gray-700 dark:text-gray-300 mb-1 block">Expert Recommendations</span>
                                    <ul className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
                                        {change.expert_recommendations.map((rec, idx) => (
                                            <li key={idx} className="flex items-start gap-1">
                                                <span className="text-blue-500 mt-1">•</span>
                                                <span>{rec}</span>
                                            </li>
                                        ))}
                                    </ul>
                                </div>                            )}
                        </div>
                    </div>
                    
                    {change.ai_analysis && (
                        <div className="mt-4 border-t border-gray-200 dark:border-gray-700 pt-4">
                            <h5 className="font-medium text-gray-900 dark:text-gray-100 mb-2 flex items-center gap-2">
                                <span className="w-2 h-2 bg-purple-500 rounded-full"></span>
                                AI Analysis
                            </h5>
                            <div className="prose prose-sm dark:prose-invert max-w-none">
                                <MarkdownRenderer content={change.ai_analysis} repoId={repoId} />
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    )
}

function CommitSummary({ commit }) {
    return (
        <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 mb-4 bg-white dark:bg-gray-800">
            <div className="flex items-start justify-between">
                <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                        <GitCommit className="w-4 h-4 text-gray-500" />                        <code className="text-sm font-mono bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200 px-2 py-1 rounded">
                            {commit.short_hash}
                        </code>
                        <span className="text-xs text-gray-500 dark:text-gray-400 flex items-center gap-1">
                            <User className="w-3 h-3" />
                            {commit.author}
                        </span>
                        <span className="text-xs text-gray-500 dark:text-gray-400 flex items-center gap-1">
                            <Calendar className="w-3 h-3" />
                            {new Date(commit.timestamp).toLocaleDateString()}
                        </span>
                    </div>
                    
                    <p className="text-sm text-gray-900 dark:text-gray-100 mb-2">
                        {commit.message}
                    </p>
                    
                    <div className="flex items-center gap-4 text-xs text-gray-500 dark:text-gray-400">
                        <span>{commit.files_analyzed} files analyzed</span>
                        <span>{commit.breaking_changes_count} breaking changes</span>
                        {commit.analysis_duration && (
                            <span>{Math.round(commit.analysis_duration * 1000)}ms</span>
                        )}
                    </div>
                </div>
            </div>
            
            {commit.error && (
                <div className="mt-2 text-sm text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/20 p-2 rounded">
                    Error: {commit.error}
                </div>
            )}
        </div>
    )
}

function BreakingChangeAnalyzer({ repoId, repoName }) {
    const [analysis, setAnalysis] = useState(null)
    const [isLoading, setIsLoading] = useState(false)
    const [error, setError] = useState(null)
    const [analysisType, setAnalysisType] = useState('days') // 'commits', 'since', 'days'
    const [selectedCommits, setSelectedCommits] = useState([])
    const [sinceCommit, setSinceCommit] = useState('')
    const [daysBack, setDaysBack] = useState(7)
    const [includeContext, setIncludeContext] = useState(true)
    const [aiAnalysis, setAiAnalysis] = useState(true)
    
    // Commit selection state
    const [availableCommits, setAvailableCommits] = useState([])
    const [commitsLoading, setCommitsLoading] = useState(false)
    const [showCommitDropdown, setShowCommitDropdown] = useState(false)
    const [showSinceDropdown, setShowSinceDropdown] = useState(false)
    
    const [expandedSections, setExpandedSections] = useState({
        summary: true,
        commits: false,
        changes: true,
        recommendations: true
    })    // Load available commits when component mounts or repoId changes
    useEffect(() => {
        if (repoId) {
            loadAvailableCommits()
        }
    }, [repoId])

    // Close dropdowns when clicking outside
    useEffect(() => {        const handleClickOutside = (event) => {
            if (!event.target.closest('.commit-dropdown')) {
                setShowCommitDropdown(false)
                setShowSinceDropdown(false)
            }
        }

        document.addEventListener('mousedown', handleClickOutside)
        return () => document.removeEventListener('mousedown', handleClickOutside)
    }, [])

    const loadAvailableCommits = async () => {
        setCommitsLoading(true)
        try {
            const commits = await diffSenseAPI.getRepositoryCommits(repoId, 100)
            // Ensure commits is always an array
            setAvailableCommits(Array.isArray(commits) ? commits : [])
        } catch (err) {
            console.error('Failed to load commits:', err)
            setAvailableCommits([]) // Set empty array on error
        } finally {
            setCommitsLoading(false)
        }
    }

    const formatCommitMessage = (message) => {
        return message.length > 60 ? message.substring(0, 60) + '...' : message
    }

    const formatCommitDate = (dateString) => {
        return new Date(dateString).toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        })
    }

    const handleAnalyze = async () => {
        if (isLoading) return

        // Validation
        if (analysisType === 'commits' && selectedCommits.length === 0) {
            setError('Please select at least one commit to analyze')
            return
        }
        if (analysisType === 'since' && !sinceCommit) {
            setError('Please select a starting commit')
            return
        }

        setIsLoading(true)
        setError(null)

        try {
            const options = {
                include_context: includeContext,
                ai_analysis: aiAnalysis
            }

            if (analysisType === 'commits' && selectedCommits.length > 0) {
                options.commit_hashes = selectedCommits.map(commit => commit.hash)
            } else if (analysisType === 'since' && sinceCommit) {
                options.since_commit = sinceCommit
            } else {
                options.days_back = daysBack
            }

            const result = await diffSenseAPI.analyzeBreakingChanges(repoId, options)
            setAnalysis(result)
        } catch (err) {
            setError(err.message || 'Failed to analyze breaking changes')
            console.error('Breaking change analysis error:', err)
        } finally {
            setIsLoading(false)
        }
    }

    const toggleSection = (section) => {
        setExpandedSections(prev => ({
            ...prev,
            [section]: !prev[section]
        }))
    }

    const getSeverityStats = () => {
        if (!analysis?.breaking_changes) return { critical: 0, high: 0, medium: 0, low: 0 }
        
        return analysis.breaking_changes.reduce((acc, change) => {
            acc[change.severity] = (acc[change.severity] || 0) + 1
            return acc
        }, { critical: 0, high: 0, medium: 0, low: 0 })
    }

    if (error) {
        return (
            <div className="p-6 border border-red-200 dark:border-red-800 rounded-lg bg-red-50 dark:bg-red-900/20">
                <div className="flex items-center gap-2 text-red-800 dark:text-red-400 mb-2">
                    <AlertTriangle className="w-5 h-5" />
                    <span className="font-medium">Analysis Failed</span>
                </div>
                <p className="text-red-700 dark:text-red-300">{error}</p>
                <button
                    onClick={() => setError(null)}
                    className="mt-3 px-4 py-2 text-sm bg-red-600 text-white rounded hover:bg-red-700 transition-colors"
                >
                    Try Again
                </button>
            </div>
        )
    }    return (
        <div className="space-y-6">
            {/* Analysis Configuration */}
            <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm">                <div className="p-6 border-b border-gray-200 dark:border-gray-700">
                    <div className="flex items-center justify-between">
                        <div>
                            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-3">
                                <div className="p-2 bg-orange-100 dark:bg-orange-900/20 rounded-lg">
                                    <AlertTriangle className="w-5 h-5 text-orange-600 dark:text-orange-400" />
                                </div>
                                Breaking Change Analysis
                            </h3>
                            <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                                Configure your analysis scope and options to detect breaking changes in your repository.
                            </p>
                        </div>
                        <button
                            onClick={handleAnalyze}
                            disabled={isLoading}
                            className="px-6 py-3 bg-orange-600 hover:bg-orange-700 disabled:bg-gray-400 dark:disabled:bg-gray-600 text-white rounded-lg font-semibold transition-colors duration-200 flex items-center gap-3 shadow-sm hover:shadow-md disabled:cursor-not-allowed flex-shrink-0"
                        >
                            {isLoading ? (
                                <>
                                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                                    Analyzing...
                                </>
                            ) : (
                                <>
                                    <AlertTriangle className="w-4 h-4" />
                                    Analyze Breaking Changes
                                </>
                            )}
                        </button>
                    </div>
                </div>
                  <div className="p-6">
                    <div className="grid lg:grid-cols-3 gap-6">
                        <div className="lg:col-span-2 space-y-6">
                            <div>
                                <label className="block text-sm font-semibold text-gray-900 dark:text-gray-100 mb-3">
                                    Analysis Scope
                                </label>
                                <select
                                    value={analysisType}
                                    onChange={(e) => setAnalysisType(e.target.value)}
                                    className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-orange-500 focus:border-transparent transition-all duration-200"
                                >
                                    <option value="days">Recent commits (last N days)</option>
                                    <option value="since">Commits since specific commit</option>
                                    <option value="commits">Select specific commits</option>
                                </select>
                            </div>{analysisType === 'days' && (
                                <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4 mt-4">
                                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                        Number of days
                                    </label>
                                    <input
                                        type="number"
                                        value={daysBack}
                                        onChange={(e) => setDaysBack(parseInt(e.target.value) || 7)}
                                        min="1"
                                        max="365"
                                        className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-orange-500 focus:border-transparent transition-all duration-200"
                                        placeholder="Number of days"
                                    />
                                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-2 flex items-center gap-1">
                                        <Clock className="w-3 h-3" />
                                        Analyze all commits from the last {daysBack} days
                                    </p>
                                </div>
                            )}{analysisType === 'since' && (
                                <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4 mt-4">
                                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                        Since commit
                                    </label>
                                    <div className="relative commit-dropdown">
                                        <button
                                            type="button"
                                            onClick={() => setShowSinceDropdown(!showSinceDropdown)}
                                            className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 text-left flex items-center justify-between hover:border-orange-300 dark:hover:border-orange-600 focus:ring-2 focus:ring-orange-500 focus:border-transparent transition-all duration-200"
                                            disabled={commitsLoading}
                                        >
                                            <span>
                                                {sinceCommit ? (
                                                    <div className="flex items-center gap-3">
                                                        <code className="text-sm bg-orange-100 dark:bg-orange-900/20 text-orange-800 dark:text-orange-300 px-2 py-1 rounded font-mono">
                                                            {sinceCommit.substring(0, 8)}
                                                        </code>
                                                        <span className="text-sm">
                                                            {Array.isArray(availableCommits) && availableCommits.find(c => c.hash === sinceCommit)?.message ? 
                                                                formatCommitMessage(availableCommits.find(c => c.hash === sinceCommit).message) : 
                                                                'Select commit...'}
                                                        </span>
                                                    </div>
                                                ) : (
                                                    <span className="text-gray-500 dark:text-gray-400">Select since commit...</span>
                                                )}
                                            </span>
                                            <ChevronDown className="w-4 h-4 text-gray-400" />
                                        </button>
                                        
                                        {showSinceDropdown && (
                                            <div className="absolute z-10 w-full mt-2 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg shadow-lg max-h-60 overflow-y-auto">
                                                {commitsLoading ? (
                                                    <div className="p-4 text-center text-gray-500">
                                                        <Loader2 className="w-4 h-4 animate-spin mx-auto mb-2" />
                                                        Loading commits...
                                                    </div>
                                                ) : !Array.isArray(availableCommits) || availableCommits.length === 0 ? (
                                                    <div className="p-4 text-center text-gray-500">No commits available</div>
                                                ) : (
                                                    availableCommits.map((commit) => (
                                                        <button
                                                            key={commit.hash}
                                                            type="button"
                                                            onClick={() => {
                                                                setSinceCommit(commit.hash)
                                                                setShowSinceDropdown(false)
                                                            }}
                                                            className="w-full px-4 py-3 text-left hover:bg-gray-100 dark:hover:bg-gray-600 border-b border-gray-100 dark:border-gray-600 last:border-b-0 transition-colors duration-150"
                                                        >                                                            <div className="flex items-center gap-3">
                                                                <code className="text-xs bg-gray-100 dark:bg-gray-600 text-gray-800 dark:text-gray-200 px-2 py-1 rounded font-mono">
                                                                    {commit.hash.substring(0, 8)}
                                                                </code>
                                                                <div className="flex-1 min-w-0">
                                                                    <div className="text-sm text-gray-900 dark:text-gray-100 truncate">
                                                                        {formatCommitMessage(commit.message)}
                                                                    </div>
                                                                    <div className="text-xs text-gray-500 dark:text-gray-400 flex items-center gap-2 mt-1">
                                                                        <User className="w-3 h-3" />
                                                                        {commit.author} • {formatCommitDate(commit.timestamp)}
                                                                    </div>
                                                                </div>
                                                            </div>
                                                        </button>
                                                    ))
                                                )}
                                            </div>
                                        )}
                                    </div>
                                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-2 flex items-center gap-1">
                                        <GitCommit className="w-3 h-3" />
                                        Analyze commits newer than the selected commit
                                    </p>
                                </div>
                            )}                          {analysisType === 'commits' && (
                                <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4 mt-4">
                                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                        Select commits
                                    </label>
                                    <div className="commit-dropdown space-y-3">
                                        <button
                                            type="button"
                                            onClick={() => setShowCommitDropdown(!showCommitDropdown)}
                                            className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 text-left flex items-center justify-between hover:border-orange-300 dark:hover:border-orange-600 focus:ring-2 focus:ring-orange-500 focus:border-transparent transition-all duration-200"
                                            disabled={commitsLoading}
                                        >
                                            <span>
                                                {selectedCommits.length > 0 ? 
                                                    <span className="text-gray-900 dark:text-gray-100">
                                                        {selectedCommits.length} commit{selectedCommits.length !== 1 ? 's' : ''} selected
                                                    </span> : 
                                                    <span className="text-gray-500 dark:text-gray-400">Select commits to analyze...</span>}
                                            </span>
                                            <ChevronDown className="w-4 h-4 text-gray-400" />
                                        </button>
                                        
                                        {selectedCommits.length > 0 && (
                                            <div className="flex flex-wrap gap-2">
                                                {selectedCommits.map((commit) => (
                                                    <span
                                                        key={commit.hash}
                                                        className="inline-flex items-center gap-2 px-3 py-1.5 bg-orange-100 dark:bg-orange-900/20 text-orange-800 dark:text-orange-300 rounded-lg text-sm border border-orange-200 dark:border-orange-800"
                                                    >
                                                        <code className="font-mono">{commit.hash.substring(0, 8)}</code>
                                                        <button
                                                            onClick={() => setSelectedCommits(prev => prev.filter(c => c.hash !== commit.hash))}
                                                            className="hover:bg-orange-200 dark:hover:bg-orange-800 rounded p-0.5 transition-colors duration-150"
                                                        >
                                                            <X className="w-3 h-3" />
                                                        </button>
                                                    </span>
                                                ))}
                                                <button
                                                    onClick={() => setSelectedCommits([])}
                                                    className="text-xs text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 px-2 py-1 transition-colors duration-150"
                                                >
                                                    Clear all
                                                </button>
                                            </div>
                                        )}
                                        
                                        {showCommitDropdown && (
                                            <div className="border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 max-h-60 overflow-y-auto shadow-lg">
                                                {commitsLoading ? (
                                                    <div className="p-4 text-center text-gray-500">
                                                        <Loader2 className="w-4 h-4 animate-spin mx-auto mb-2" />
                                                        Loading commits...
                                                    </div>
                                                ) : !Array.isArray(availableCommits) || availableCommits.length === 0 ? (
                                                    <div className="p-4 text-center text-gray-500">No commits available</div>
                                                ) : (
                                                    availableCommits.map((commit) => {
                                                        const isSelected = selectedCommits.some(c => c.hash === commit.hash)
                                                        return (
                                                            <label
                                                                key={commit.hash}
                                                                className="flex items-center gap-3 p-4 hover:bg-gray-50 dark:hover:bg-gray-600 border-b border-gray-100 dark:border-gray-600 last:border-b-0 cursor-pointer transition-colors duration-150"
                                                            >
                                                                <input
                                                                    type="checkbox"
                                                                    checked={isSelected}
                                                                    onChange={(e) => {
                                                                        if (e.target.checked) {
                                                                            setSelectedCommits(prev => [...prev, commit])
                                                                        } else {
                                                                            setSelectedCommits(prev => prev.filter(c => c.hash !== commit.hash))
                                                                        }
                                                                    }}
                                                                    className="rounded text-orange-600 focus:ring-orange-500"
                                                                />                                                                <div className="flex items-center gap-3 flex-1 min-w-0">
                                                                    <code className="text-xs bg-gray-100 dark:bg-gray-600 text-gray-800 dark:text-gray-200 px-2 py-1 rounded font-mono">
                                                                        {commit.hash.substring(0, 8)}
                                                                    </code>
                                                                    <div className="flex-1 min-w-0">
                                                                        <div className="text-sm text-gray-900 dark:text-gray-100 truncate">
                                                                            {formatCommitMessage(commit.message)}
                                                                        </div>
                                                                        <div className="text-xs text-gray-500 dark:text-gray-400 flex items-center gap-2 mt-1">
                                                                            <User className="w-3 h-3" />
                                                                            {commit.author} • {formatCommitDate(commit.timestamp)}
                                                                        </div>
                                                                    </div>
                                                                </div>
                                                            </label>
                                                        )
                                                    })
                                                )}
                                            </div>
                                        )}
                                    </div>
                                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-2 flex items-center gap-1">
                                        <GitCommit className="w-3 h-3" />
                                        Select multiple commits for targeted analysis
                                    </p>                                </div>
                            )}
                        </div>
                        
                        <div className="lg:col-span-1 space-y-6">
                            <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm">
                                <div className="p-4 border-b border-gray-200 dark:border-gray-700">
                                    <h4 className="text-sm font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2">
                                        <Wrench className="w-4 h-4 text-orange-600 dark:text-orange-400" />
                                        Analysis Options
                                    </h4>
                                    <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                                        Configure advanced analysis settings
                                    </p>
                                </div>
                                <div className="p-4 space-y-4">
                                    <label className="flex items-start gap-3 cursor-pointer group">
                                        <input
                                            type="checkbox"
                                            checked={includeContext}
                                            onChange={(e) => setIncludeContext(e.target.checked)}
                                            className="mt-1 rounded text-orange-600 focus:ring-orange-500 focus:ring-2 focus:ring-offset-0"
                                        />
                                        <div className="flex-1">
                                            <span className="text-sm font-medium text-gray-900 dark:text-gray-100 group-hover:text-orange-600 dark:group-hover:text-orange-400 transition-colors duration-150">
                                                Include detailed context
                                            </span>
                                            <p className="text-xs text-gray-600 dark:text-gray-400 mt-1 leading-relaxed">
                                                Provide additional context and surrounding code for better analysis
                                            </p>
                                        </div>
                                    </label>
                                    <div className="border-t border-gray-100 dark:border-gray-700 pt-4">
                                        <label className="flex items-start gap-3 cursor-pointer group">
                                            <input
                                                type="checkbox"
                                                checked={aiAnalysis}
                                                onChange={(e) => setAiAnalysis(e.target.checked)}
                                                className="mt-1 rounded text-orange-600 focus:ring-orange-500 focus:ring-2 focus:ring-offset-0"
                                            />
                                            <div className="flex-1">
                                                <span className="text-sm font-medium text-gray-900 dark:text-gray-100 group-hover:text-orange-600 dark:group-hover:text-orange-400 transition-colors duration-150">
                                                    Enable AI-powered analysis
                                                </span>
                                                <p className="text-xs text-gray-600 dark:text-gray-400 mt-1 leading-relaxed">
                                                    Use advanced AI to provide deeper insights and recommendations
                                                </p>
                                            </div>
                                        </label>
                                    </div>                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Analysis Results */}
            {analysis && (
                <div className="space-y-6">
                    {/* Summary Section */}
                    <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg">
                        <button
                            onClick={() => toggleSection('summary')}
                            className="w-full px-6 py-4 flex items-center justify-between text-left"
                        >
                            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2">
                                <FileText className="w-5 h-5 text-blue-500" />
                                Analysis Summary
                            </h3>
                            {expandedSections.summary ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
                        </button>
                        
                        {expandedSections.summary && (
                            <div className="px-6 pb-4 border-t border-gray-200 dark:border-gray-700">
                                <div className="grid md:grid-cols-4 gap-4 mb-4">
                                    <div className="text-center p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                                        <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">{analysis.commits_analyzed}</div>
                                        <div className="text-sm text-gray-600 dark:text-gray-400">Commits Analyzed</div>
                                    </div>
                                    <div className="text-center p-4 bg-red-50 dark:bg-red-900/20 rounded-lg">
                                        <div className="text-2xl font-bold text-red-600 dark:text-red-400">{analysis.summary.total_changes}</div>
                                        <div className="text-sm text-red-600 dark:text-red-400">Breaking Changes</div>
                                    </div>
                                    <div className="text-center p-4 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
                                        <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">{analysis.summary.critical_changes}</div>
                                        <div className="text-sm text-orange-600 dark:text-orange-400">Critical Issues</div>
                                    </div>
                                    <div className="text-center p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                                        <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">{analysis.summary.affected_files}</div>
                                        <div className="text-sm text-blue-600 dark:text-blue-400">Affected Files</div>
                                    </div>
                                </div>
                                
                                {analysis.summary.change_types && (
                                    <div>
                                        <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-2">Change Types Distribution</h4>
                                        <div className="grid md:grid-cols-2 gap-2">
                                            {Object.entries(analysis.summary.change_types).map(([type, count]) => (
                                                <div key={type} className="flex justify-between items-center p-2 bg-gray-50 dark:bg-gray-700 rounded">
                                                    <span className="text-sm text-gray-700 dark:text-gray-300">{type}</span>
                                                    <span className="text-sm font-medium text-gray-900 dark:text-gray-100">{count}</span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>

                    {/* Commits Section */}
                    {analysis.commits_details && analysis.commits_details.length > 0 && (
                        <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg">
                            <button
                                onClick={() => toggleSection('commits')}
                                className="w-full px-6 py-4 flex items-center justify-between text-left"
                            >
                                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2">
                                    <GitCommit className="w-5 h-5 text-gray-500" />
                                    Analyzed Commits ({analysis.commits_details.length})
                                </h3>
                                {expandedSections.commits ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
                            </button>
                            
                            {expandedSections.commits && (
                                <div className="px-6 pb-4 border-t border-gray-200 dark:border-gray-700 max-h-96 overflow-y-auto">
                                    {analysis.commits_details.map((commit, idx) => (
                                        <CommitSummary key={idx} commit={commit} />
                                    ))}
                                </div>
                            )}
                        </div>
                    )}

                    {/* Breaking Changes Section */}
                    {analysis.breaking_changes && analysis.breaking_changes.length > 0 && (
                        <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg">
                            <button
                                onClick={() => toggleSection('changes')}
                                className="w-full px-6 py-4 flex items-center justify-between text-left"
                            >
                                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2">
                                    <AlertTriangle className="w-5 h-5 text-red-500" />
                                    Breaking Changes ({analysis.breaking_changes.length})
                                </h3>
                                {expandedSections.changes ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
                            </button>
                            
                            {expandedSections.changes && (
                                <div className="px-6 pb-4 border-t border-gray-200 dark:border-gray-700 max-h-96 overflow-y-auto">
                                    {analysis.breaking_changes.map((change, idx) => (
                                        <BreakingChangeCard key={idx} change={change} repoId={repoId} />
                                    ))}
                                </div>
                            )}
                        </div>
                    )}

                    {/* Recommendations Section */}
                    {analysis.recommendations && (
                        <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg">
                            <button
                                onClick={() => toggleSection('recommendations')}
                                className="w-full px-6 py-4 flex items-center justify-between text-left"
                            >
                                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2">
                                    <Wrench className="w-5 h-5 text-green-500" />
                                    Recommendations & Migration Guide
                                </h3>
                                {expandedSections.recommendations ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
                            </button>
                            
                            {expandedSections.recommendations && (
                                <div className="px-6 pb-4 border-t border-gray-200 dark:border-gray-700">
                                    <div className="grid md:grid-cols-2 gap-6">
                                        <div>
                                            <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-3 flex items-center gap-2">
                                                <AlertTriangle className="w-4 h-4 text-red-500" />
                                                Immediate Actions
                                            </h4>
                                            <ul className="space-y-2">
                                                {analysis.recommendations.immediate_actions.map((action, idx) => (
                                                    <li key={idx} className="flex items-start gap-2 text-sm text-gray-600 dark:text-gray-400">
                                                        <span className="text-red-500 mt-1">•</span>
                                                        <span>{action}</span>
                                                    </li>
                                                ))}
                                            </ul>
                                        </div>
                                        
                                        <div>
                                            <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-3 flex items-center gap-2">
                                                <ExternalLink className="w-4 h-4 text-blue-500" />
                                                Migration Strategy
                                            </h4>
                                            <ul className="space-y-2">
                                                {analysis.recommendations.migration_strategy.map((strategy, idx) => (
                                                    <li key={idx} className="flex items-start gap-2 text-sm text-gray-600 dark:text-gray-400">
                                                        <span className="text-blue-500 mt-1">•</span>
                                                        <span>{strategy}</span>
                                                    </li>
                                                ))}
                                            </ul>
                                        </div>
                                        
                                        <div>
                                            <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-3 flex items-center gap-2">
                                                <Users className="w-4 h-4 text-purple-500" />
                                                Communication Plan
                                            </h4>
                                            <ul className="space-y-2">
                                                {analysis.recommendations.communication_plan.map((plan, idx) => (
                                                    <li key={idx} className="flex items-start gap-2 text-sm text-gray-600 dark:text-gray-400">
                                                        <span className="text-purple-500 mt-1">•</span>
                                                        <span>{plan}</span>
                                                    </li>
                                                ))}
                                            </ul>
                                        </div>
                                        
                                        <div>
                                            <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-3 flex items-center gap-2">
                                                <Wrench className="w-4 h-4 text-green-500" />
                                                Testing Recommendations
                                            </h4>
                                            <ul className="space-y-2">
                                                {analysis.recommendations.testing_recommendations.map((test, idx) => (
                                                    <li key={idx} className="flex items-start gap-2 text-sm text-gray-600 dark:text-gray-400">
                                                        <span className="text-green-500 mt-1">•</span>
                                                        <span>{test}</span>
                                                    </li>
                                                ))}
                                            </ul>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                    )}

                    {/* Analysis Metadata */}
                    <div className="text-center text-xs text-gray-500 dark:text-gray-400">
                        Analysis completed at {new Date(analysis.analysis_timestamp).toLocaleString()}
                    </div>
                </div>
            )}
        </div>
    )
}

export default BreakingChangeAnalyzer

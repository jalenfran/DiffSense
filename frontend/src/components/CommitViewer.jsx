import React, { useState, useEffect } from 'react'
import { 
    X, 
    GitCommit, 
    Calendar, 
    User, 
    Hash, 
    FileText,
    Plus,
    Minus,
    Edit,
    Loader2,
    AlertTriangle,
    Clock
} from 'lucide-react'
import PortableFileViewer from './PortableFileViewer'
import { diffSenseAPI } from '../services/api'

/**
 * CommitViewer - A modal component for viewing all files changed in a commit
 * 
 * Features:
 * - Shows commit metadata (hash, message, author, date)
 * - Lists all files changed in the commit with diff stats
 * - Displays file content with diff view in the main area
 * - Allows navigation between files in the commit
 * - Uses PortableFileViewer for consistent file rendering
 */
const CommitViewer = ({ 
    isOpen, 
    onClose, 
    repoId, 
    commitHash,
    title = null 
}) => {
    const [commitData, setCommitData] = useState(null)
    const [commitFiles, setCommitFiles] = useState([])
    const [selectedFile, setSelectedFile] = useState(null)
    const [isLoading, setIsLoading] = useState(false)
    const [error, setError] = useState(null)
    const [breakingChanges, setBreakingChanges] = useState([])
    const [isLoadingBreakingChanges, setIsLoadingBreakingChanges] = useState(false)    // Load commit data and files when opened
    useEffect(() => {
        if (isOpen && repoId && commitHash) {
            loadCommitData()
            loadBreakingChanges()
        } else {
            // Reset state when closed
            setCommitData(null)
            setCommitFiles([])
            setSelectedFile(null)
            setError(null)
            setBreakingChanges([])
        }
    }, [isOpen, repoId, commitHash])

    const loadCommitData = async () => {
        setIsLoading(true)
        setError(null)

        try {
            // Load commit files and metadata
            const response = await diffSenseAPI.getCommitFiles(repoId, commitHash, {
                include_diff_stats: true,
                include_metadata: true
            })
            
            console.log('CommitViewer: Full API response', response)
            
            // Extract commit metadata
            const metadata = response.commit_info || response.commit_metadata || response.metadata || {}
            const summary = response.summary || {}
            
            console.log('CommitViewer: Extracted metadata', metadata)
            console.log('CommitViewer: Extracted summary', summary)
            
            // Get files array
            const filesRaw = response.files_changed || response.files || []
            console.log('CommitViewer: Raw files data', filesRaw)
            
            // Calculate total stats from files if not in summary
            let totalAdditions = 0
            let totalDeletions = 0
            
            filesRaw.forEach(file => {
                console.log('CommitViewer: Processing file', file)
                const additions = file.insertions || file.additions || file.lines_added || 
                                file.added_lines || file.linesAdded || file.stats?.additions ||
                                file.diff_stats?.additions || file.diffStats?.additions || 0
                const deletions = file.deletions || file.lines_removed || file.deleted_lines || 
                                file.linesRemoved || file.stats?.deletions ||
                                file.diff_stats?.deletions || file.diffStats?.deletions || 0
                totalAdditions += additions
                totalDeletions += deletions
                console.log('CommitViewer: File stats - additions:', additions, 'deletions:', deletions)
            })
            
            console.log('CommitViewer: Calculated totals - additions:', totalAdditions, 'deletions:', totalDeletions)
            
            setCommitData({
                hash: commitHash,
                message: metadata.message || metadata.commit_message || 'No commit message',
                author: metadata.author || metadata.author_name || 'Unknown',
                date: metadata.date || metadata.commit_date || metadata.authored_date,
                stats: {
                    additions: summary.total_additions || metadata.additions || metadata.lines_added || totalAdditions || 0,
                    deletions: summary.total_deletions || metadata.deletions || metadata.lines_removed || totalDeletions || 0,
                    files_changed: summary.files_changed || metadata.files_changed || filesRaw.length || 0
                }
            })

            // Normalize file data
            const files = filesRaw.map(file => {
                const additions = file.insertions || file.additions || file.lines_added || 
                                file.added_lines || file.linesAdded || file.stats?.additions ||
                                file.diff_stats?.additions || file.diffStats?.additions || 0
                const deletions = file.deletions || file.lines_removed || file.deleted_lines || 
                                file.linesRemoved || file.stats?.deletions ||
                                file.diff_stats?.deletions || file.diffStats?.deletions || 0
                
                return {
                    path: file.file_path || file.path,
                    status: file.change_type || file.status || file.action || 'M',
                    stats: {
                        additions: additions,
                        deletions: deletions,
                        changes: file.changes || file.total_changes || (additions + deletions) || 0
                    },
                    size: file.size || file.file_size || 0,
                    rawFile: file
                }
            })
            
            setCommitFiles(files)
            
            // Auto-select first file if available
            if (files.length > 0) {
                setSelectedFile(files[0])
            }
            
        } catch (err) {
            console.error('Error loading commit data:', err)
            setError(err.message || 'Failed to load commit data')
        } finally {
            setIsLoading(false)
        }
    }

    const loadBreakingChanges = async () => {
        setIsLoadingBreakingChanges(true)
        
        try {
            console.log('CommitViewer: Loading breaking changes for commit:', commitHash, 'repoId:', repoId)
            
            // Use the advanced breaking change analysis endpoint with specific commit
            const options = {
                commit_hashes: [commitHash],
                include_context: true,
                ai_analysis: false // Set to false for faster response in commit viewer
            }
            
            console.log('CommitViewer: Calling analyzeBreakingChanges with options:', options)
            const response = await diffSenseAPI.analyzeBreakingChanges(repoId, options)
            console.log('CommitViewer: Breaking changes full response', response)
              // Extract breaking changes from response - try multiple possible fields
            const changes = response.breaking_changes || response.changes || response.analysis?.breaking_changes || []
            console.log('CommitViewer: Extracted breaking changes:', changes.length, changes)
            
            // Log the file paths in the breaking changes for debugging
            if (changes.length > 0) {
                console.log('CommitViewer: Breaking change file paths:', changes.map(c => c.file_path || c.affected_file))
            }
            
            // Also log all available fields in the response
            console.log('CommitViewer: Available response fields:', Object.keys(response))
            
            setBreakingChanges(changes)
            
        } catch (err) {
            console.error('Error loading breaking changes:', err)
            console.error('Error details:', err.response?.data || err.message)
            // Don't show error for breaking changes - it's optional
            setBreakingChanges([])
        } finally {
            setIsLoadingBreakingChanges(false)
        }
    }    // Check if a file has breaking changes
    const getFileBreakingChanges = (filePath) => {
        console.log('CommitViewer: Matching file', filePath, 'against breaking changes')
        const matches = breakingChanges.filter(change => {
            const changeFile = change.file_path || change.affected_file
            console.log('CommitViewer: Comparing', filePath, 'with', changeFile)
            
            // Try multiple matching strategies
            const exactMatch = changeFile === filePath
            const endsWithMatch = filePath.endsWith(changeFile) || changeFile.endsWith(filePath)
            const normalizedMatch = filePath.replace(/^\.\//, '') === changeFile.replace(/^\.\//, '')
            
            const isMatch = exactMatch || endsWithMatch || normalizedMatch
            
            if (isMatch) {
                console.log('CommitViewer: MATCH found for', filePath, 'with change', change)
            }
            
            return isMatch
        })
        
        console.log('CommitViewer: Found', matches.length, 'breaking changes for file', filePath)
        return matches
    }

    // Get the highest severity for a file
    const getFileSeverity = (filePath) => {
        const fileChanges = getFileBreakingChanges(filePath)
        if (fileChanges.length === 0) return null
        
        const severities = { critical: 4, high: 3, medium: 2, low: 1 }
        const highestSeverity = fileChanges.reduce((max, change) => {
            const severity = severities[change.severity] || 0
            return severity > max.value ? { severity: change.severity, value: severity } : max
        }, { severity: null, value: 0 })
        
        return highestSeverity.severity
    }    // Get warning icon for breaking changes
    const getBreakingChangeIcon = (filePath) => {
        const severity = getFileSeverity(filePath)
        const fileChanges = getFileBreakingChanges(filePath)
        console.log('CommitViewer: Getting icon for file:', filePath, 'severity:', severity, 'changes:', fileChanges.length, 'total changes:', breakingChanges.length)
        
        if (!severity || fileChanges.length === 0) return null
        
        const colors = {
            critical: 'text-red-500',
            high: 'text-orange-500', 
            medium: 'text-yellow-500',
            low: 'text-blue-500'
        }
        
        console.log('CommitViewer: Rendering breaking change icon with severity:', severity, 'color:', colors[severity])
        
        return (
            <AlertTriangle 
                className={`w-4 h-4 ${colors[severity] || 'text-yellow-500'} flex-shrink-0`}
                title={`${fileChanges.length} breaking change${fileChanges.length !== 1 ? 's' : ''} detected (${severity} severity)`}
            />
        )
    }

    const handleFileSelect = (file) => {
        setSelectedFile(file)
    }

    const formatDate = (dateString) => {
        if (!dateString) return 'Unknown date'
        return new Date(dateString).toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        })
    }

    const getStatusIcon = (status) => {
        switch (status?.toUpperCase()) {
            case 'A': return <Plus className="w-3 h-3 text-green-500" />
            case 'D': return <Minus className="w-3 h-3 text-red-500" />
            case 'M': return <Edit className="w-3 h-3 text-blue-500" />
            case 'R': return <Hash className="w-3 h-3 text-purple-500" />
            default: return <Edit className="w-3 h-3 text-gray-500" />
        }
    }

    const getStatusColor = (status) => {
        switch (status?.toUpperCase()) {
            case 'A': return 'text-green-600 dark:text-green-400'
            case 'D': return 'text-red-600 dark:text-red-400'
            case 'M': return 'text-blue-600 dark:text-blue-400'
            case 'R': return 'text-purple-600 dark:text-purple-400'
            default: return 'text-gray-600 dark:text-gray-400'
        }
    }

    const getStatusText = (status) => {
        switch (status?.toUpperCase()) {
            case 'A': return 'Added'
            case 'D': return 'Deleted'
            case 'M': return 'Modified'
            case 'R': return 'Renamed'
            default: return 'Changed'
        }
    }

    if (!isOpen) return null

    const displayTitle = title || (commitHash ? `Commit ${commitHash.substring(0, 8)}` : 'Commit Viewer')

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 dark:bg-black dark:bg-opacity-70 flex items-center justify-center z-50">
            <div className="bg-white dark:bg-gray-800 rounded-lg w-full h-full max-w-[98vw] max-h-[98vh] flex flex-col shadow-2xl">
                {/* Header */}
                <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50">
                    <div className="flex items-center gap-3 min-w-0 flex-1">
                        <GitCommit className="w-5 h-5 text-green-500 flex-shrink-0" />
                        <div className="min-w-0 flex-1">
                            <h2 className="text-lg font-semibold text-gray-900 dark:text-white truncate">
                                {displayTitle}
                            </h2>
                            {commitData && (
                                <div className="flex items-center gap-4 mt-1 text-sm text-gray-600 dark:text-gray-400">
                                    <span className="flex items-center gap-1">
                                        <User className="w-3 h-3" />
                                        {commitData.author}
                                    </span>
                                    <span className="flex items-center gap-1">
                                        <Calendar className="w-3 h-3" />
                                        {formatDate(commitData.date)}
                                    </span>                                    {commitFiles.length > 0 && (
                                        <span className="flex items-center gap-1">
                                            <FileText className="w-3 h-3" />
                                            {commitFiles.length} file{commitFiles.length !== 1 ? 's' : ''}
                                        </span>
                                    )}
                                    {breakingChanges.length > 0 && (
                                        <span className="flex items-center gap-1 text-orange-600 dark:text-orange-400">
                                            <AlertTriangle className="w-3 h-3" />
                                            {breakingChanges.length} breaking change{breakingChanges.length !== 1 ? 's' : ''}
                                        </span>
                                    )}
                                </div>
                            )}
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors flex-shrink-0"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                {/* Commit Message */}
                {commitData?.message && (
                    <div className="px-4 py-3 bg-gray-50 dark:bg-gray-700/30 border-b border-gray-200 dark:border-gray-700">
                        <p className="text-sm text-gray-700 dark:text-gray-300">
                            {commitData.message}
                        </p>
                        {commitData.stats && (
                            <div className="flex items-center gap-4 mt-2 text-xs text-gray-500 dark:text-gray-400">
                                <span className="text-green-600 dark:text-green-400">
                                    +{commitData.stats.additions} additions
                                </span>
                                <span className="text-red-600 dark:text-red-400">
                                    -{commitData.stats.deletions} deletions
                                </span>
                            </div>
                        )}
                    </div>
                )}

                {/* Content */}
                <div className="flex-1 flex overflow-hidden">
                    {/* Files Sidebar */}
                    <div className="w-80 border-r border-gray-200 dark:border-gray-700 flex flex-col bg-gray-50 dark:bg-gray-800/30">                        <div className="p-4 border-b border-gray-200 dark:border-gray-700">
                            <h3 className="font-semibold text-gray-900 dark:text-white flex items-center gap-2">
                                <FileText className="w-4 h-4 text-blue-500" />
                                Files Changed ({commitFiles.length})
                                {breakingChanges.length > 0 && (
                                    <span className="flex items-center gap-1 text-orange-600 dark:text-orange-400 text-sm font-normal">
                                        <AlertTriangle className="w-3 h-3" />
                                        {breakingChanges.length} breaking
                                    </span>
                                )}
                            </h3>
                        </div>
                        
                        <div className="flex-1 overflow-y-auto">                                {isLoading ? (
                                <div className="flex items-center justify-center h-32">
                                    <div className="text-center">
                                        <Loader2 className="w-6 h-6 animate-spin mx-auto mb-2 text-blue-500" />
                                        <p className="text-sm text-gray-600 dark:text-gray-400">Loading commit...</p>
                                    </div>
                                </div>
                            ) : error ? (
                                <div className="flex items-center justify-center h-32">
                                    <div className="text-center">
                                        <AlertTriangle className="w-6 h-6 mx-auto mb-2 text-red-500" />
                                        <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
                                    </div>
                                </div>
                            ) : commitFiles.length === 0 ? (
                                <div className="flex items-center justify-center h-32">
                                    <div className="text-center">
                                        <FileText className="w-6 h-6 mx-auto mb-2 text-gray-400" />
                                        <p className="text-sm text-gray-600 dark:text-gray-400">No files found</p>
                                    </div>
                                </div>
                            ) : (
                                <div className="p-2 space-y-1">
                                    {commitFiles.map((file, index) => {
                                        const isSelected = selectedFile?.path === file.path
                                        
                                        return (
                                            <button
                                                key={file.path || index}
                                                onClick={() => handleFileSelect(file)}
                                                className={`w-full text-left p-3 rounded-lg transition-colors ${
                                                    isSelected 
                                                        ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300' 
                                                        : 'hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300'
                                                }`}
                                            >                                                <div className="flex items-center justify-between mb-1">
                                                    <span className="text-xs font-mono truncate flex-1" title={file.path}>
                                                        {file.path.split('/').pop()}
                                                    </span>
                                                    <div className="flex items-center gap-1 ml-2">
                                                        {getBreakingChangeIcon(file.path)}
                                                        {getStatusIcon(file.status)}
                                                        <span className={`text-xs font-medium ${getStatusColor(file.status)}`}>
                                                            {file.status?.toUpperCase()}
                                                        </span>
                                                    </div>
                                                </div>
                                                <div className="text-xs text-gray-500 dark:text-gray-400 truncate mb-1">
                                                    {file.path}
                                                </div>
                                                <div className="text-xs text-gray-500 dark:text-gray-400">
                                                    <span className={getStatusColor(file.status)}>
                                                        {getStatusText(file.status)}
                                                    </span>
                                                    {file.stats.additions > 0 && (
                                                        <span className="text-green-600 dark:text-green-400 ml-2">
                                                            +{file.stats.additions}
                                                        </span>
                                                    )}                                                    {file.stats.deletions > 0 && (
                                                        <span className="text-red-600 dark:text-red-400 ml-1">
                                                            -{file.stats.deletions}
                                                        </span>
                                                    )}
                                                </div>
                                            </button>
                                        )
                                    })}
                                </div>
                            )}
                        </div>
                    </div>

                    {/* File Content */}
                    <div className="flex-1">
                        {selectedFile ? (
                            <PortableFileViewer
                                repoId={repoId}
                                filePath={selectedFile.path}
                                commitHash={commitHash}
                                showDiff={true} // Always show diff in commit view
                                height="100%"
                                className="h-full border-0 rounded-none"
                                showHeader={true}
                                showControls={true}
                            />
                        ) : (
                            <div className="flex items-center justify-center h-full text-gray-500 dark:text-gray-400">
                                <div className="text-center">
                                    <GitCommit className="w-12 h-12 mx-auto mb-3 text-gray-300 dark:text-gray-600" />
                                    <p className="text-lg font-medium mb-1">Select a file to view changes</p>
                                    <p className="text-sm">Choose a file from the sidebar to see the diff</p>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    )
}

export default CommitViewer

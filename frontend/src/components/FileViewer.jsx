import { useState, useEffect, useRef, useCallback } from 'react'
import {
    X,
    ChevronLeft,
    ChevronRight,
    Eye,
    FileText,
    GitCommit,
    Calendar,
    User,
    Hash,
    Copy,
    ExternalLink,
    Code,
    Clock,
    Download,
    RotateCcw
} from 'lucide-react'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneLight, oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { diffSenseAPI } from '../services/api'

const FileViewer = ({ isOpen, onClose, repoId, filePath, commitHash, viewMode = 'file' }) => {
    // Detect language from file extension
    const detectLanguage = (filepath) => {
        if (!filepath) return 'text'
        const ext = filepath.split('.').pop()?.toLowerCase()
        
        const languageMap = {
            'js': 'javascript',
            'jsx': 'jsx',
            'ts': 'typescript',
            'tsx': 'tsx',
            'py': 'python',
            'java': 'java',
            'cpp': 'cpp',
            'c': 'c',
            'h': 'c',
            'hpp': 'cpp',
            'cs': 'csharp',
            'php': 'php',
            'rb': 'ruby',
            'go': 'go',
            'rs': 'rust',
            'sql': 'sql',
            'json': 'json',
            'xml': 'xml',
            'html': 'html',
            'css': 'css',
            'scss': 'scss',
            'sass': 'sass',
            'md': 'markdown',
            'yml': 'yaml',
            'yaml': 'yaml',
            'sh': 'bash',
            'bash': 'bash',
            'zsh': 'bash',
            'ps1': 'powershell',
            'dockerfile': 'dockerfile'
        }
        
        return languageMap[ext] || 'text'
    }

    const [fileContent, setFileContent] = useState('')
    const [fileHistory, setFileHistory] = useState([])
    const [commitFiles, setCommitFiles] = useState([])
    const [currentCommitIndex, setCurrentCommitIndex] = useState(0)
    const [showDiff, setShowDiff] = useState(false)
    const [isLoading, setIsLoading] = useState(false)
    const [error, setError] = useState(null)
    const [fileInfo, setFileInfo] = useState(null)
    const [selectedFile, setSelectedFile] = useState(filePath)
    const containerRef = useRef(null)

    // Update selectedFile when filePath prop changes
    useEffect(() => {
        if (filePath) {
            setSelectedFile(filePath)
        }
    }, [filePath])    // Load file content
    const loadFileContent = async (path = filePath || selectedFile, commit = null) => {
        if (!repoId || !path) {
            console.log('loadFileContent: Missing repoId or path', { repoId, path })
            return
        }

        console.log('loadFileContent: Starting request', { repoId, path, commit, showDiff, viewMode })
        setIsLoading(true)
        setError(null)

        try {
            const options = {}
            
            // Always pass the commit if we have one
            if (commit) {
                options.commitHash = commit
            } else if (viewMode === 'file' && fileHistory.length > 0 && currentCommitIndex < fileHistory.length) {
                // For file view, use the current commit from history
                const currentCommit = fileHistory[currentCommitIndex]
                if (currentCommit?.hash) {
                    options.commitHash = currentCommit.hash
                }
            } else if (viewMode === 'commit' && commitHash) {
                // For commit view, use the provided commit hash
                options.commitHash = commitHash
            }
            
            // Show diff when requested
            if (showDiff) {
                options.showDiff = true
            }

            console.log('loadFileContent: Calling API with options', options)
            const response = await diffSenseAPI.getFileContent(repoId, path, options)
            console.log('loadFileContent: API response', response)
            setFileContent(response.content || response.diff || '')
            setFileInfo(response.metadata || null)
        } catch (err) {
            console.error('Error loading file content:', err)
            setError(err.message || 'Failed to load file content')
        } finally {
            setIsLoading(false)
        }
    }

    // Load file history
    const loadFileHistory = async (path = filePath || selectedFile) => {
        if (!repoId || !path) {
            console.log('loadFileHistory: Missing repoId or path', { repoId, path })
            return
        }

        console.log('loadFileHistory: Starting request', { repoId, path })
        try {
            const response = await diffSenseAPI.getFileHistory(repoId, path, {
                limit: 50,
                showDiffs: false
            })
            console.log('loadFileHistory: API response', response)
            setFileHistory(response.commits || response.history || [])
        } catch (err) {
            console.error('Error loading file history:', err)
        }
    }

    // Load commit files (for commit view mode)
    const loadCommitFiles = async (commit = commitHash) => {
        if (!repoId || !commit || viewMode !== 'commit') {
            console.log('loadCommitFiles: Missing params or wrong mode', { repoId, commit, viewMode })
            return
        }

        console.log('loadCommitFiles: Starting request', { repoId, commit })
        try {
            const response = await diffSenseAPI.getCommitFiles(repoId, commit, {
                includeDiffStats: true
            })
            console.log('loadCommitFiles: API response', response)
            
            // Normalize the file data structure - try all possible field names
            const files = (response.files_changed || response.files || []).map(file => {
                // Try multiple possible field names for diff stats from raw API response
                const additions = file.insertions || file.additions || file.lines_added || 
                                file.added_lines || file.linesAdded || file.stats?.additions ||
                                file.diff_stats?.additions || file.diffStats?.additions || 0
                const deletions = file.deletions || file.lines_removed || file.deleted_lines || 
                                file.linesRemoved || file.stats?.deletions ||
                                file.diff_stats?.deletions || file.diffStats?.deletions || 0
                
                return {
                    path: file.file_path || file.path,
                    file_path: file.file_path || file.path,
                    status: file.change_type || file.status || file.action || 'M',
                    stats: {
                        additions: additions,
                        deletions: deletions,
                        changes: file.changes || file.total_changes || (additions + deletions) || 0
                    },
                    size: file.size || file.file_size || 0,
                    rawFile: file // Keep original for debugging and fallback
                }
            })
            console.log('Normalized files:', files)
            console.log('Sample file data:', files[0])
            setCommitFiles(files)
        } catch (err) {
            console.error('Error loading commit files:', err)
        }
    }

    // Initialize data on open
    useEffect(() => {
        console.log('FileViewer useEffect triggered', { isOpen, repoId, filePath, commitHash, viewMode })
        if (isOpen) {
            // Clear previous state
            setFileContent('')
            setFileHistory([])
            setCommitFiles([])
            setCurrentCommitIndex(0)
            setError(null)
            setFileInfo(null)
            
            if (viewMode === 'commit' && commitHash) {
                console.log('Loading commit view')
                loadCommitFiles()
                if (filePath) {
                    loadFileContent(filePath, commitHash)
                } else {
                    // For commit view without specific file, just load the commit files list
                    setFileContent('')
                }
            } else if (viewMode === 'file' && filePath) {
                console.log('Loading file view')
                loadFileHistory()
                loadFileContent()
            }
        }
    }, [isOpen, repoId, filePath, commitHash, viewMode])

    // Reload content when showDiff or current commit changes
    useEffect(() => {
        if (isOpen && selectedFile) {
            const currentCommit = fileHistory[currentCommitIndex]?.hash || commitHash
            if (viewMode === 'file') {
                // For file view, load with diff option based on showDiff state
                loadFileContent(selectedFile, currentCommit)
            } else {
                // For commit view, always load the specific commit
                loadFileContent(selectedFile, commitHash)
            }
        }
    }, [showDiff, currentCommitIndex, selectedFile])

    // Navigation functions
    const navigateToPreviousCommit = () => {
        if (currentCommitIndex < fileHistory.length - 1) {
            setCurrentCommitIndex(currentCommitIndex + 1)
        }
    }

    const navigateToNextCommit = () => {
        if (currentCommitIndex > 0) {
            setCurrentCommitIndex(currentCommitIndex - 1)
        }
    }

    const handleFileSelect = (file) => {
        setSelectedFile(file.path || file.file_path)
        setCurrentCommitIndex(0)
        if (viewMode === 'commit') {
            loadFileContent(file.path || file.file_path, commitHash)
        } else {
            loadFileHistory(file.path || file.file_path)
            loadFileContent(file.path || file.file_path)
        }
    }

    const copyToClipboard = (text) => {
        navigator.clipboard.writeText(text)
    }

    const formatFileSize = (bytes) => {
        if (!bytes) return 'Unknown size'
        const sizes = ['B', 'KB', 'MB', 'GB']
        const i = Math.floor(Math.log(bytes) / Math.log(1024))
        return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${sizes[i]}`
    }

    // Filter out binary file metadata and diff headers - optimized with useCallback
    const renderDiffContent = useCallback((content) => {
        if (!content) return null

        const language = detectLanguage(selectedFile)
        const isDarkMode = document.documentElement.classList.contains('dark')        // Filter out binary file metadata and diff headers
        const filterDiffMetadata = (diffContent) => {
            const lines = diffContent.split('\n')
            const filteredLines = []
            let inBinarySection = false
            let inFileHeader = false
            
            for (let i = 0; i < lines.length; i++) {
                const line = lines[i]
                
                // Skip binary file metadata
                if (line.includes('=======================================================')) {
                    inBinarySection = true
                    continue
                }
                
                // Detect file header sections (filename alone on a line)
                if (line.trim() && !line.startsWith('+') && !line.startsWith('-') && !line.startsWith('@@') && 
                    !line.startsWith('---') && !line.startsWith('+++') && !line.startsWith('diff ') &&
                    !line.startsWith('index ') && !line.includes('|') && 
                    (line.includes('.') || line.match(/^[a-zA-Z0-9_.-]+$/))) {
                    // Check if next few lines are metadata
                    let isFileHeaderSection = false
                    for (let j = i + 1; j < Math.min(i + 4, lines.length); j++) {
                        if (lines[j].startsWith('lhs:') || lines[j].startsWith('rhs:') || 
                            lines[j].includes('file added') || lines[j].includes('file deleted') ||
                            lines[j].includes('file modified')) {
                            isFileHeaderSection = true
                            break
                        }
                    }
                    if (isFileHeaderSection) {
                        inFileHeader = true
                        continue
                    }
                }
                
                // Skip file header metadata lines
                if (inFileHeader && (
                    line.startsWith('lhs:') || 
                    line.startsWith('rhs:') ||
                    line.includes('file added') ||
                    line.includes('file deleted') ||
                    line.includes('file modified') ||
                    line.match(/^[a-f0-9]{6,}\s+\|\s+[a-f0-9]{40}$/) ||
                    line.includes(' | ') && line.match(/^[a-f0-9]+/)
                )) {
                    continue
                }
                
                // End of file header section when we hit a diff marker
                if (inFileHeader && line.startsWith('---')) {
                    inFileHeader = false
                }
                
                // Skip binary file info lines
                if (inBinarySection && (
                    line.startsWith('lhs:') || 
                    line.startsWith('rhs:') ||
                    line.match(/^[a-f0-9]{6,}\s+\|\s+[a-f0-9]{40}$/)
                )) {
                    continue
                }
                
                // End of binary section
                if (inBinarySection && line.startsWith('---')) {
                    inBinarySection = false
                }
                
                // Skip if still in binary section
                if (inBinarySection) {
                    continue
                }
                
                // Skip index lines and similarity index
                if (line.startsWith('index ') || 
                    line.startsWith('similarity index ') ||
                    line.startsWith('rename from ') ||
                    line.startsWith('rename to ') ||
                    line.startsWith('new file mode ') ||
                    line.startsWith('deleted file mode ')) {
                    continue
                }
                
                filteredLines.push(line)
            }
            
            return filteredLines.join('\n')
        }

        const filteredContent = filterDiffMetadata(content)

        // Helper function to render syntax-highlighted code line - memoized for performance
        const renderCodeLine = (lineContent, lineType) => {
            // Skip empty lines or non-code lines
            if (!lineContent || lineContent.trim() === '' || 
                lineContent.startsWith('@@') || 
                lineContent.startsWith('+++') || 
                lineContent.startsWith('---')) {
                return <span className="whitespace-pre">{lineContent}</span>
            }

            // Get the diff prefix and clean content
            let diffPrefix = ''
            let cleanContent = lineContent
            
            if (lineType === 'addition' && lineContent.startsWith('+')) {
                diffPrefix = '+'
                cleanContent = lineContent.slice(1)
            } else if (lineType === 'deletion' && lineContent.startsWith('-')) {
                diffPrefix = '-'
                cleanContent = lineContent.slice(1)
            }

            try {
                // Use SyntaxHighlighter for the clean content
                const highlightedContent = (
                    <SyntaxHighlighter
                        language={language}
                        style={isDarkMode ? oneDark : oneLight}
                        customStyle={{
                            margin: 0,
                            padding: 0,
                            background: 'transparent',
                            fontSize: 'inherit',
                            lineHeight: 'inherit'
                        }}
                        codeTagProps={{
                            style: { 
                                background: 'transparent',
                                padding: 0,
                                fontSize: 'inherit'
                            }
                        }}
                        PreTag={({ children }) => <span>{children}</span>}
                    >
                        {cleanContent}
                    </SyntaxHighlighter>
                )

                // Return the diff prefix + highlighted content
                return (
                    <span className="whitespace-pre">
                        {diffPrefix && (
                            <span className={`inline-block w-4 ${
                                lineType === 'addition' ? 'text-green-600 dark:text-green-400' : 
                                lineType === 'deletion' ? 'text-red-600 dark:text-red-400' : ''
                            }`}>
                                {diffPrefix}
                            </span>
                        )}
                        {highlightedContent}
                    </span>
                )
            } catch (e) {
                // Fallback to plain text if highlighting fails
                return <span className="whitespace-pre">{lineContent}</span>
            }
        }

        // For diff content, we'll use a custom styling approach with syntax highlighting
        return (
            <div className="font-mono text-sm">
                {filteredContent.split('\n').map((line, index) => {
                    let lineClass = 'flex items-start min-h-[1.5rem]'
                    let bgClass = ''
                    let borderClass = 'border-l-2 border-transparent'
                    let lineType = 'normal'
                    
                    if (line.startsWith('+') && !line.startsWith('+++')) {
                        bgClass = 'bg-green-50 dark:bg-green-900/20'
                        borderClass = 'border-l-2 border-green-400 dark:border-green-500'
                        lineType = 'addition'
                    } else if (line.startsWith('-') && !line.startsWith('---')) {
                        bgClass = 'bg-red-50 dark:bg-red-900/20'
                        borderClass = 'border-l-2 border-red-400 dark:border-red-500'
                        lineType = 'deletion'
                    } else if (line.startsWith('@@')) {
                        bgClass = 'bg-blue-50 dark:bg-blue-900/20'
                        borderClass = 'border-l-2 border-blue-400 dark:border-blue-500'
                        lineType = 'header'
                    } else {
                        bgClass = 'hover:bg-gray-50 dark:hover:bg-gray-700/50'
                        borderClass = 'border-l-2 border-transparent hover:border-gray-300 dark:hover:border-gray-600'
                        lineType = 'normal'
                    }

                    return (
                        <div key={index} className={`${lineClass} ${bgClass} ${borderClass}`}>
                            <span className="text-gray-400 dark:text-gray-500 text-right mr-0 flex-shrink-0 select-none px-2 py-1 min-w-[3em] bg-gray-50 dark:bg-gray-900 border-r border-gray-200 dark:border-gray-700">
                                {index + 1}
                            </span>
                            <div className="flex-1 min-w-0 px-4 py-1">
                                {lineType === 'normal' || lineType === 'addition' || lineType === 'deletion' 
                                    ? renderCodeLine(line, lineType)
                                    : <span className="whitespace-pre text-gray-700 dark:text-gray-300 font-medium">{line || ' '}</span>
                                }
                            </div>
                        </div>
                    )
                })}
            </div>
        )
    }, [selectedFile])

    const renderFileContent = (content) => {
        if (!content) return null

        if (showDiff && content.includes('@@')) {
            return renderDiffContent(content)
        }

        const language = detectLanguage(selectedFile)
        const isDarkMode = document.documentElement.classList.contains('dark')

        return (
            <div className="font-mono text-sm">
                <SyntaxHighlighter
                    language={language}
                    style={isDarkMode ? oneDark : oneLight}
                    showLineNumbers={true}
                    wrapLines={true}
                    wrapLongLines={true}
                    lineNumberStyle={{
                        minWidth: '3em',
                        paddingRight: '1em',
                        paddingLeft: '0.5em',
                        textAlign: 'right',
                        color: isDarkMode ? '#6b7280' : '#9ca3af',
                        borderRight: `1px solid ${isDarkMode ? '#374151' : '#e5e7eb'}`,
                        marginRight: '0',
                        backgroundColor: isDarkMode ? '#111827' : '#f9fafb'
                    }}
                    customStyle={{
                        margin: 0,
                        padding: '1rem',
                        backgroundColor: 'transparent',
                        fontSize: '14px',
                        lineHeight: '1.5'
                    }}
                    codeTagProps={{
                        style: { 
                            backgroundColor: 'transparent',
                            fontSize: 'inherit'
                        }
                    }}
                >
                    {content}
                </SyntaxHighlighter>
            </div>
        )
    }

    if (!isOpen) return null

    const currentCommit = fileHistory[currentCommitIndex]
    const currentFile = commitFiles.find(f => (f.path || f.file_path) === selectedFile)

    return (
        <div className="fixed inset-2 bg-black bg-opacity-50 dark:bg-black dark:bg-opacity-70 flex items-center justify-center z-[70]">
            <div className="bg-white dark:bg-gray-800 rounded-lg w-full h-full max-w-[95vw] max-h-[95vh] flex flex-col shadow-2xl">
                {/* Header */}
                <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
                    <div className="flex items-center gap-3">
                        <FileText className="w-5 h-5 text-blue-500" />
                        <div>
                            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                                {selectedFile || 'File Viewer'}
                            </h2>
                            {currentCommit && (
                                <p className="text-sm text-gray-500 dark:text-gray-400">
                                    {viewMode === 'commit' ? 'Commit View' : `Commit ${currentCommitIndex + 1} of ${fileHistory.length}`}
                                </p>
                            )}
                        </div>
                    </div>
                    
                    <div className="flex items-center gap-2">
                        {/* Toggle Diff View */}
                        <button
                            onClick={() => setShowDiff(!showDiff)}
                            className={`px-3 py-1.5 text-sm font-medium rounded-lg transition-colors ${
                                showDiff 
                                    ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400' 
                                    : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                            }`}
                            title={showDiff ? 'Hide diff' : 'Show diff'}
                        >
                            <Eye className="w-4 h-4" />
                        </button>

                        {/* Copy button */}
                        <button
                            onClick={() => copyToClipboard(fileContent)}
                            className="p-2 text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
                            title="Copy content"
                        >
                            <Copy className="w-4 h-4" />
                        </button>

                        {/* Close button */}
                        <button
                            onClick={onClose}
                            className="p-2 text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
                        >
                            <X className="w-5 h-5" />
                        </button>
                    </div>
                </div>

                {/* Content */}
                <div className="flex-1 flex overflow-hidden">
                    {/* Sidebar for commit files or file history */}
                    {((viewMode === 'commit' && commitFiles.length > 0) || (viewMode === 'file' && fileHistory.length > 0)) && (
                        <div className="w-80 border-r border-gray-200 dark:border-gray-700 flex flex-col">
                            <div className="p-4 border-b border-gray-200 dark:border-gray-700">
                                <h3 className="font-semibold text-gray-900 dark:text-white flex items-center gap-2">
                                    {viewMode === 'commit' ? (
                                        <>
                                            <GitCommit className="w-4 h-4 text-green-500" />
                                            Files Changed ({commitFiles.length})
                                        </>
                                    ) : (
                                        <>
                                            <Clock className="w-4 h-4 text-blue-500" />
                                            File History ({fileHistory.length})
                                        </>
                                    )}
                                </h3>
                            </div>
                            
                            <div className="flex-1 overflow-y-auto">
                                {viewMode === 'commit' ? (
                                    <div className="p-2 space-y-1">
                                        {commitFiles.map((file, index) => {
                                            const isSelected = (file.path || file.file_path) === selectedFile
                                            const statusColors = {
                                                'A': 'text-green-600 dark:text-green-400',
                                                'M': 'text-blue-600 dark:text-blue-400',
                                                'D': 'text-red-600 dark:text-red-400',
                                                'R': 'text-purple-600 dark:text-purple-400'
                                            }
                                            
                                            return (
                                                <button
                                                    key={index}
                                                    onClick={() => handleFileSelect(file)}
                                                    className={`w-full text-left p-2 rounded-lg transition-colors ${
                                                        isSelected 
                                                            ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300' 
                                                            : 'hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300'
                                                    }`}
                                                >
                                                    <div className="flex items-center justify-between mb-1">
                                                        <span className="text-xs font-mono truncate" title={file.path || file.file_path}>
                                                            {(file.path || file.file_path).split('/').pop()}
                                                        </span>
                                                        <span className={`text-xs font-bold ${statusColors[file.status] || 'text-gray-500'}`}>
                                                            {file.status}
                                                        </span>
                                                    </div>
                                                    <div className="text-xs text-gray-500 dark:text-gray-400 truncate">
                                                        {file.path || file.file_path}
                                                    </div>
                                                    {file.stats && (
                                                        <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                                                            <span className="text-green-600 dark:text-green-400">+{file.stats.additions}</span>
                                                            {' '}
                                                            <span className="text-red-600 dark:text-red-400">-{file.stats.deletions}</span>
                                                        </div>
                                                    )}
                                                </button>
                                            )
                                        })}
                                    </div>
                                ) : (
                                    <div className="p-2 space-y-1">
                                        {fileHistory.map((commit, index) => {
                                            const isSelected = index === currentCommitIndex
                                            
                                            return (
                                                <button
                                                    key={commit.hash || index}
                                                    onClick={() => setCurrentCommitIndex(index)}
                                                    className={`w-full text-left p-2 rounded-lg transition-colors ${
                                                        isSelected 
                                                            ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300' 
                                                            : 'hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300'
                                                    }`}
                                                >
                                                    <div className="flex items-center justify-between mb-1">
                                                        <span className="text-xs font-mono">
                                                            {commit.hash?.substring(0, 8) || 'Unknown'}
                                                        </span>
                                                        <span className="text-xs text-gray-500 dark:text-gray-400">
                                                            {commit.date ? new Date(commit.date).toLocaleDateString() : ''}
                                                        </span>
                                                    </div>
                                                    <div className="text-xs text-gray-600 dark:text-gray-300 truncate">
                                                        {commit.message || commit.subject || 'No message'}
                                                    </div>
                                                    {commit.author && (
                                                        <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                                                            {commit.author}
                                                        </div>
                                                    )}
                                                </button>
                                            )
                                        })}
                                    </div>
                                )}
                            </div>
                        </div>
                    )}

                    {/* Main content area */}
                    <div className="flex-1 flex flex-col overflow-hidden">
                        {/* Navigation and file info */}
                        {viewMode === 'file' && fileHistory.length > 0 && (
                            <div className="p-3 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-700/50">
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-4">
                                        <div className="flex items-center gap-1">
                                            <button
                                                onClick={navigateToPreviousCommit}
                                                disabled={currentCommitIndex >= fileHistory.length - 1}
                                                className="p-1 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600 rounded disabled:opacity-50 disabled:cursor-not-allowed"
                                                title="Previous commit"
                                            >
                                                <ChevronLeft className="w-4 h-4" />
                                            </button>
                                            <span className="text-sm text-gray-600 dark:text-gray-300 font-mono">
                                                {currentCommitIndex + 1} / {fileHistory.length}
                                            </span>
                                            <button
                                                onClick={navigateToNextCommit}
                                                disabled={currentCommitIndex <= 0}
                                                className="p-1 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600 rounded disabled:opacity-50 disabled:cursor-not-allowed"
                                                title="Next commit"
                                            >
                                                <ChevronRight className="w-4 h-4" />
                                            </button>
                                        </div>
                                        
                                        {currentCommit && (
                                            <div className="text-sm text-gray-600 dark:text-gray-300">
                                                <span className="font-mono">{currentCommit.hash?.substring(0, 8)}</span>
                                                {currentCommit.date && (
                                                    <span className="ml-2">{new Date(currentCommit.date).toLocaleDateString()}</span>
                                                )}
                                            </div>
                                        )}
                                    </div>
                                    
                                    {fileInfo && (
                                        <div className="text-sm text-gray-500 dark:text-gray-400">
                                            {formatFileSize(fileInfo.size)}
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}

                        {/* File content */}
                        <div className="flex-1 overflow-auto bg-white dark:bg-gray-800" ref={containerRef}>
                            {isLoading ? (
                                <div className="flex items-center justify-center h-64">
                                    <div className="text-gray-500 dark:text-gray-400">Loading...</div>
                                </div>
                            ) : error ? (
                                <div className="flex items-center justify-center h-64">
                                    <div className="text-red-500 dark:text-red-400">Error: {error}</div>
                                </div>
                            ) : (
                                <div className="bg-white dark:bg-gray-800">
                                    {renderFileContent(fileContent)}
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default FileViewer

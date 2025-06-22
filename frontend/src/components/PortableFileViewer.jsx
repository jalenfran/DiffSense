import React, { useState, useEffect, useMemo } from 'react'
import { 
    Copy, 
    Download, 
    Eye, 
    EyeOff, 
    Code, 
    Image, 
    Archive, 
    FileText,
    Loader2,
    AlertTriangle,
    ChevronLeft,
    ChevronRight
} from 'lucide-react'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneLight, oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { diffSenseAPI } from '../services/api'

/**
 * PortableFileViewer - A standalone, reusable component for displaying file content
 * 
 * Features:
 * - Syntax highlighting for code files
 * - Diff view support with toggle
 * - Binary/image file handling
 * - Copy and download functionality
 * - Flexible styling and size options
 * - Can be embedded anywhere (Files tab, modals, inline)
 *  * Props:
 * - repoId: Repository ID
 * - filePath: Path to the file
 * - commitHash: Optional commit hash for specific version
 * - showDiff: Boolean to show diff view initially
 * - className: Additional CSS classes
 * - style: Inline styles
 * - height: Height of the content area ('auto', '400px', '100%', etc.)
 * - showHeader: Whether to show the file header with name and controls
 * - showControls: Whether to show action buttons (copy, download, diff toggle)
 * - showDiffToggle: Whether to show the diff/content toggle button
 * - showCopyButton: Whether to show the copy button
 * - showDownloadButton: Whether to show the download button
 * - showCommitNavigation: Whether to show commit navigation controls
 * - onToggleDiff: Callback when diff is toggled
 * - onCopy: Callback when content is copied
 * - onDownload: Callback when file is downloaded
 * - onNavigateCommit: Callback for commit navigation (previous/next)
 * - initialContent: Pre-loaded content (skips API call)
 * - compact: Whether to use compact styling
 */
const PortableFileViewer = ({ 
    repoId,
    filePath, 
    commitHash = null,
    showDiff = false,
    className = '',
    style = {},
    height = 'auto',
    showHeader = true,
    showControls = true,
    showDiffToggle = true,
    showCopyButton = true,
    showDownloadButton = true,
    showCommitNavigation = false,
    onToggleDiff = null,
    onCopy = null,
    onDownload = null,
    onNavigateCommit = null,
    initialContent = null,
    compact = false
}) => {    const [content, setContent] = useState(initialContent || '')
    const [diffData, setDiffData] = useState(null)
    const [isLoading, setIsLoading] = useState(false)
    const [error, setError] = useState(null)
    const [fileInfo, setFileInfo] = useState(null)
    const [isDiffMode, setIsDiffMode] = useState(showDiff)

    // Detect file language from extension
    const detectLanguage = (filepath) => {
        if (!filepath) return 'text'
        const ext = filepath.toLowerCase().split('.').pop()
        
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

    const getFileType = (filepath) => {
        if (!filepath) return 'text'
        const ext = filepath.toLowerCase().split('.').pop()
        
        if (['png', 'jpg', 'jpeg', 'gif', 'svg', 'webp', 'bmp', 'ico'].includes(ext)) {
            return 'image'
        }
        if (['zip', 'tar', 'gz', 'rar', '7z', 'exe', 'bin', 'dll'].includes(ext)) {
            return 'binary'
        }
        return 'text'
    }

    const formatFileSize = (bytes) => {
        if (!bytes) return ''
        const sizes = ['B', 'KB', 'MB', 'GB']
        const i = Math.floor(Math.log(bytes) / Math.log(1024))
        return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${sizes[i]}`
    }    // Load content based on mode
    const loadContent = async () => {
        if (!repoId || !filePath) return

        setIsLoading(true)
        setError(null)

        try {
            const options = {}
            if (commitHash) options.commitHash = commitHash
              let response
            if (isDiffMode) {
                // Load both current content and diff
                const [contentResponse, diffResponse] = await Promise.all([
                    diffSenseAPI.getFileContent(repoId, filePath, options),
                    diffSenseAPI.getFileContent(repoId, filePath, { ...options, showDiff: true })
                ])
                response = contentResponse
                setContent(contentResponse?.content || '')
                setDiffData(diffResponse?.diff || diffResponse?.content || '')
            } else {
                response = await diffSenseAPI.getFileContent(repoId, filePath, options)
                setContent(response?.content || '')
                setDiffData(null)
            }
            
            setFileInfo(response)
        } catch (err) {
            console.error('Error loading file content:', err)
            setError(err.message || 'Failed to load file content')
        } finally {
            setIsLoading(false)
        }
    }

    // Load content when props change
    useEffect(() => {
        if (initialContent) {
            setContent(initialContent)
        } else if (repoId && filePath) {
            loadContent()
        }
    }, [repoId, filePath, commitHash, isDiffMode, initialContent])

    // Update diff mode when prop changes
    useEffect(() => {
        setIsDiffMode(showDiff)
    }, [showDiff])

    const handleToggleDiff = () => {
        const newDiffMode = !isDiffMode
        setIsDiffMode(newDiffMode)
        if (onToggleDiff) {
            onToggleDiff(newDiffMode)
        }
    }

    const handleCopy = () => {
        navigator.clipboard.writeText(content)
        if (onCopy) {
            onCopy(content)
        }
    }

    const handleDownload = () => {
        const blob = new Blob([content], { type: 'text/plain' })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = filePath ? filePath.split('/').pop() : 'file.txt'
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
        URL.revokeObjectURL(url)
        
        if (onDownload) {
            onDownload(content)
        }
    }    // Parse diff data to extract change information
    const parseDiffData = (diffContent) => {
        if (!diffContent || !diffContent.includes('@@')) return { ranges: [], changes: {} }
        
        const lines = diffContent.split('\n')
        const ranges = []
        const changes = {}
        let currentRange = null
        let lineNumber = 0
        
        for (const line of lines) {
            if (line.startsWith('@@')) {
                // Parse hunk header like @@ -1,4 +1,6 @@
                const match = line.match(/@@ -(\d+),?(\d+)? \+(\d+),?(\d+)? @@/)
                if (match) {
                    currentRange = {
                        oldStart: parseInt(match[1]),
                        oldCount: parseInt(match[2]) || 1,
                        newStart: parseInt(match[3]),
                        newCount: parseInt(match[4]) || 1
                    }
                    ranges.push(currentRange)
                    lineNumber = currentRange.newStart
                }
            } else if (line.startsWith('+') && !line.startsWith('+++')) {
                // Added line
                if (currentRange) {
                    changes[lineNumber] = { type: 'added', content: line.substring(1) }
                    lineNumber++
                }
            } else if (line.startsWith('-') && !line.startsWith('---')) {
                // Removed line - don't increment line number for new file
                if (currentRange) {
                    changes[lineNumber] = { type: 'removed', content: line.substring(1) }
                }
            } else if (line.startsWith(' ')) {
                // Context line
                lineNumber++
            }
        }
        
        return { ranges, changes }
    }    // Render enhanced diff view - syntax highlighted content with diff overlays
    const renderEnhancedDiffContent = useMemo(() => {
        if (!content || !diffData) return null

        const language = detectLanguage(filePath)
        const isDarkMode = document.documentElement.classList.contains('dark')
        const { changes } = parseDiffData(diffData)

        return (
            <SyntaxHighlighter
                language={language}
                style={isDarkMode ? oneDark : oneLight}
                showLineNumbers={true}
                wrapLines={true}
                wrapLongLines={true}
                lineProps={(lineNumber) => {
                    const change = changes[lineNumber]
                    let style = {}
                    let className = ''
                    
                    if (change) {
                        if (change.type === 'added') {
                            style.backgroundColor = isDarkMode ? 'rgba(34, 197, 94, 0.2)' : 'rgba(34, 197, 94, 0.1)'
                            style.borderLeft = '2px solid rgb(34, 197, 94)'
                            className = 'diff-added'
                        } else if (change.type === 'removed') {
                            style.backgroundColor = isDarkMode ? 'rgba(239, 68, 68, 0.2)' : 'rgba(239, 68, 68, 0.1)'
                            style.borderLeft = '2px solid rgb(239, 68, 68)'
                            className = 'diff-removed'
                        }
                    }
                    
                    return { style, className }
                }}
                lineNumberStyle={{
                    minWidth: compact ? '2.5em' : '3em',
                    paddingRight: compact ? '0.5em' : '1em',
                    paddingLeft: compact ? '0.25em' : '0.5em',
                    textAlign: 'right',
                    color: isDarkMode ? '#6b7280' : '#9ca3af',
                    borderRight: `1px solid ${isDarkMode ? '#374151' : '#e5e7eb'}`,
                    marginRight: '0',
                    backgroundColor: isDarkMode ? '#111827' : '#f9fafb'
                }}
                customStyle={{
                    margin: 0,
                    padding: compact ? '0.5rem' : '1rem',
                    backgroundColor: 'transparent',
                    fontSize: compact ? '12px' : '14px',
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
        )
    }, [content, diffData, filePath, compact])

    // Render regular file content with syntax highlighting
    const renderFileContent = useMemo(() => {
        if (!content) return null

        const language = detectLanguage(filePath)
        const isDarkMode = document.documentElement.classList.contains('dark')

        return (
            <SyntaxHighlighter
                language={language}
                style={isDarkMode ? oneDark : oneLight}
                showLineNumbers={true}
                wrapLines={true}
                wrapLongLines={true}
                lineNumberStyle={{
                    minWidth: compact ? '2.5em' : '3em',
                    paddingRight: compact ? '0.5em' : '1em',
                    paddingLeft: compact ? '0.25em' : '0.5em',
                    textAlign: 'right',
                    color: isDarkMode ? '#6b7280' : '#9ca3af',
                    borderRight: `1px solid ${isDarkMode ? '#374151' : '#e5e7eb'}`,
                    marginRight: '0',
                    backgroundColor: isDarkMode ? '#111827' : '#f9fafb'
                }}
                customStyle={{
                    margin: 0,
                    padding: compact ? '0.5rem' : '1rem',
                    backgroundColor: 'transparent',
                    fontSize: compact ? '12px' : '14px',
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
        )
    }, [content, filePath, compact])

    const fileType = getFileType(filePath)
    const fileName = filePath ? filePath.split('/').pop() : 'Unknown file'

    return (
        <div 
            className={`bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden ${className}`}
            style={style}
        >
            {/* Header */}
            {showHeader && (
                <div className={`flex items-center justify-between ${compact ? 'p-2' : 'p-3'} border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50`}>
                    <div className="flex items-center gap-2 min-w-0 flex-1">
                        {fileType === 'image' ? (
                            <Image className={`${compact ? 'w-3 h-3' : 'w-4 h-4'} text-pink-500 flex-shrink-0`} />
                        ) : fileType === 'binary' ? (
                            <Archive className={`${compact ? 'w-3 h-3' : 'w-4 h-4'} text-gray-500 flex-shrink-0`} />
                        ) : (
                            <Code className={`${compact ? 'w-3 h-3' : 'w-4 h-4'} text-blue-500 flex-shrink-0`} />
                        )}
                        <span className={`${compact ? 'text-sm' : 'text-base'} font-medium text-gray-900 dark:text-gray-100 truncate`}>
                            {fileName}
                        </span>
                        {fileInfo?.size_bytes && (
                            <span className={`${compact ? 'text-xs' : 'text-sm'} text-gray-500 dark:text-gray-400 flex-shrink-0`}>
                                ({formatFileSize(fileInfo.size_bytes)})
                            </span>
                        )}
                        {commitHash && (
                            <span className={`${compact ? 'text-xs px-1 py-0.5' : 'text-xs px-2 py-1'} font-mono text-gray-500 dark:text-gray-400 bg-gray-100 dark:bg-gray-700 rounded flex-shrink-0`}>
                                {commitHash.substring(0, 8)}
                            </span>
                        )}
                    </div>                    {/* Controls */}
                    {showControls && (
                        <div className={`flex items-center ${compact ? 'gap-0.5 ml-2' : 'gap-1 ml-4'}`}>                            {/* Diff toggle - only show for text files and when not using initialContent */}
                            {showDiffToggle && fileType === 'text' && !initialContent && (
                                <button
                                    onClick={handleToggleDiff}
                                    className={`flex items-center gap-2 px-3 py-1.5 rounded-lg transition-colors text-sm font-medium ${
                                        isDiffMode
                                            ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 border border-green-300 dark:border-green-700'
                                            : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600 border border-gray-300 dark:border-gray-600'
                                    }`}
                                    title={isDiffMode ? 'Switch to content view' : 'Switch to diff view'}
                                >
                                    {isDiffMode ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                                    {isDiffMode ? 'View Content' : 'View Diff'}
                                </button>
                            )}

                            {/* Copy button */}
                            {showCopyButton && (
                                <button
                                    onClick={handleCopy}
                                    className={`${compact ? 'p-1' : 'p-2'} text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors`}
                                    title="Copy content"
                                >
                                    <Copy className={`${compact ? 'w-3 h-3' : 'w-4 h-4'}`} />
                                </button>
                            )}

                            {/* Download button */}
                            {showDownloadButton && (
                                <button
                                    onClick={handleDownload}
                                    className={`${compact ? 'p-1' : 'p-2'} text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors`}
                                    title="Download file"
                                >
                                    <Download className={`${compact ? 'w-3 h-3' : 'w-4 h-4'}`} />
                                </button>
                            )}

                            {/* Commit navigation - show when enabled and callback provided */}
                            {showCommitNavigation && onNavigateCommit && (
                                <>
                                    <div className="w-px h-4 bg-gray-300 dark:bg-gray-600 mx-1" />
                                    <button
                                        onClick={() => onNavigateCommit('previous')}
                                        className={`${compact ? 'p-1' : 'p-2'} text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors`}
                                        title="Previous commit"
                                    >
                                        <ChevronLeft className={`${compact ? 'w-3 h-3' : 'w-4 h-4'}`} />
                                    </button>
                                    <button
                                        onClick={() => onNavigateCommit('next')}
                                        className={`${compact ? 'p-1' : 'p-2'} text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors`}
                                        title="Next commit"
                                    >
                                        <ChevronRight className={`${compact ? 'w-3 h-3' : 'w-4 h-4'}`} />
                                    </button>
                                </>
                            )}
                        </div>
                    )}
                </div>
            )}

            {/* Content */}
            <div 
                className="overflow-auto"
                style={{ height: height !== 'auto' ? height : undefined }}
            >
                {isLoading ? (
                    <div className={`flex items-center justify-center ${compact ? 'h-24' : 'h-32'}`}>
                        <div className="text-center">
                            <Loader2 className={`${compact ? 'w-4 h-4' : 'w-6 h-6'} animate-spin mx-auto mb-2 text-blue-500`} />
                            <p className={`${compact ? 'text-xs' : 'text-sm'} text-gray-600 dark:text-gray-400`}>Loading file...</p>
                        </div>
                    </div>
                ) : error ? (
                    <div className={`flex items-center justify-center ${compact ? 'h-24' : 'h-32'}`}>
                        <div className="text-center">
                            <AlertTriangle className={`${compact ? 'w-4 h-4' : 'w-6 h-6'} mx-auto mb-2 text-red-500`} />
                            <p className={`${compact ? 'text-xs' : 'text-sm'} text-red-600 dark:text-red-400`}>{error}</p>
                        </div>
                    </div>
                ) : fileType === 'image' ? (
                    <div className={`${compact ? 'p-4' : 'p-8'} text-center`}>
                        <Image className={`${compact ? 'w-8 h-8' : 'w-16 h-16'} mx-auto mb-3 text-gray-400`} />
                        <p className={`${compact ? 'text-sm' : 'text-base'} text-gray-600 dark:text-gray-400`}>Image file</p>
                        <p className={`${compact ? 'text-xs' : 'text-sm'} text-gray-500 dark:text-gray-500`}>{fileName}</p>
                    </div>
                ) : fileType === 'binary' ? (
                    <div className={`${compact ? 'p-4' : 'p-8'} text-center`}>
                        <Archive className={`${compact ? 'w-8 h-8' : 'w-16 h-16'} mx-auto mb-3 text-gray-400`} />
                        <p className={`${compact ? 'text-sm' : 'text-base'} text-gray-600 dark:text-gray-400`}>Binary file</p>
                        <p className={`${compact ? 'text-xs' : 'text-sm'} text-gray-500 dark:text-gray-500`}>{fileName}</p>
                    </div>
                ) : !content ? (
                    <div className={`${compact ? 'p-4' : 'p-8'} text-center`}>
                        <FileText className={`${compact ? 'w-8 h-8' : 'w-16 h-16'} mx-auto mb-3 text-gray-400`} />
                        <p className={`${compact ? 'text-sm' : 'text-base'} text-gray-600 dark:text-gray-400`}>No content available</p>
                    </div>                ) : (
                    <div className="bg-white dark:bg-gray-800">
                        {isDiffMode && diffData ? renderEnhancedDiffContent : renderFileContent}
                    </div>
                )}
            </div>
        </div>
    )
}

export default PortableFileViewer

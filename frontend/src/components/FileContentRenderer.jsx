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
    AlertTriangle
} from 'lucide-react'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneLight, oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { diffSenseAPI } from '../services/api'

const FileContentRenderer = ({ 
    repoId,
    filePath, 
    commitHash = null,
    showDiff = false,
    className = '',
    height = 'auto',
    showHeader = true,
    showControls = true,
    onToggleDiff = null,
    onCopy = null,
    onDownload = null,
    initialContent = null
}) => {
    const [content, setContent] = useState(initialContent || '')
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
    }

    // Load file content from API
    const loadContent = async () => {
        if (!repoId || !filePath) return

        setIsLoading(true)
        setError(null)

        try {
            const options = {}
            if (commitHash) options.commitHash = commitHash
            if (isDiffMode) options.showDiff = true

            const response = await diffSenseAPI.getFileContent(repoId, filePath, options)
            setContent(response?.content || response?.diff || '')
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
        }    }

    // Render diff content with syntax highlighting
    const renderDiffContent = useMemo(() => {
        if (!content || !content.includes('@@')) return null

        const language = detectLanguage(filePath)
        const isDarkMode = document.documentElement.classList.contains('dark')

        // Filter out binary metadata and git headers
        const filterDiffMetadata = (diffContent) => {
            const lines = diffContent.split('\n')
            return lines.filter(line => {
                // Skip binary file metadata, index lines, and file mode changes
                return !line.startsWith('index ') &&
                       !line.startsWith('similarity index ') &&
                       !line.startsWith('rename from ') &&
                       !line.startsWith('rename to ') &&
                       !line.startsWith('new file mode ') &&
                       !line.startsWith('deleted file mode ') &&
                       !line.includes('=======================================================')
            }).join('\n')
        }

        const filteredContent = filterDiffMetadata(content)

        // Parse diff to extract actual code lines for syntax highlighting
        const parseAndHighlightDiff = () => {
            const lines = filteredContent.split('\n')
            const processedLines = []
            
            lines.forEach((line, index) => {
                let lineType = 'context'
                let actualCode = line
                let bgStyle = {}
                let borderStyle = {}
                
                if (line.startsWith('+') && !line.startsWith('+++')) {
                    lineType = 'added'
                    actualCode = line.substring(1) // Remove the + prefix
                    bgStyle.backgroundColor = isDarkMode ? 'rgba(34, 197, 94, 0.2)' : 'rgba(34, 197, 94, 0.1)'
                    borderStyle.borderLeft = '2px solid rgb(34, 197, 94)'
                } else if (line.startsWith('-') && !line.startsWith('---')) {
                    lineType = 'removed'
                    actualCode = line.substring(1) // Remove the - prefix
                    bgStyle.backgroundColor = isDarkMode ? 'rgba(239, 68, 68, 0.2)' : 'rgba(239, 68, 68, 0.1)'
                    borderStyle.borderLeft = '2px solid rgb(239, 68, 68)'
                } else if (line.startsWith('@@')) {
                    lineType = 'hunk'
                    bgStyle.backgroundColor = isDarkMode ? 'rgba(59, 130, 246, 0.2)' : 'rgba(59, 130, 246, 0.1)'
                    borderStyle.borderLeft = '2px solid rgb(59, 130, 246)'
                } else if (line.startsWith(' ')) {
                    actualCode = line.substring(1) // Remove the space prefix
                }
                
                processedLines.push({
                    original: line,
                    code: actualCode,
                    type: lineType,
                    bgStyle,
                    borderStyle,
                    lineNumber: index + 1
                })
            })
            
            return processedLines
        }

        const processedLines = parseAndHighlightDiff()
        
        return (
            <SyntaxHighlighter
                language={language}
                style={isDarkMode ? oneDark : oneLight}
                showLineNumbers={true}
                wrapLines={true}
                wrapLongLines={true}
                lineProps={(lineNumber) => {
                    const lineData = processedLines[lineNumber - 1]
                    if (!lineData) return {}
                    
                    return {
                        style: { ...lineData.bgStyle, ...lineData.borderStyle },
                        className: `diff-line diff-${lineData.type}`
                    }
                }}
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
                        backgroundColor: 'transparent',                        fontSize: 'inherit'
                    }
                }}
            >
                {processedLines.map(line => line.code).join('\n')}
            </SyntaxHighlighter>
        )
    }, [content, filePath])

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
        )
    }, [content, filePath])

    const fileType = getFileType(filePath)
    const fileName = filePath ? filePath.split('/').pop() : 'Unknown file'

    return (
        <div className={`bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden ${className}`}>
            {/* Header */}
            {showHeader && (
                <div className="flex items-center justify-between p-3 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50">
                    <div className="flex items-center gap-2 min-w-0 flex-1">
                        {fileType === 'image' ? (
                            <Image className="w-4 h-4 text-pink-500 flex-shrink-0" />
                        ) : fileType === 'binary' ? (
                            <Archive className="w-4 h-4 text-gray-500 flex-shrink-0" />
                        ) : (
                            <Code className="w-4 h-4 text-blue-500 flex-shrink-0" />
                        )}
                        <span className="font-medium text-gray-900 dark:text-gray-100 truncate">
                            {fileName}
                        </span>
                        {fileInfo?.size_bytes && (
                            <span className="text-sm text-gray-500 dark:text-gray-400 flex-shrink-0">
                                ({formatFileSize(fileInfo.size_bytes)})
                            </span>
                        )}
                        {commitHash && (
                            <span className="text-xs font-mono text-gray-500 dark:text-gray-400 bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded flex-shrink-0">
                                {commitHash.substring(0, 8)}
                            </span>
                        )}
                    </div>

                    {/* Controls */}
                    {showControls && (
                        <div className="flex items-center gap-1 ml-4">                            {/* Diff toggle - only show for text files and when not using initialContent */}
                            {fileType === 'text' && !initialContent && (
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
                            <button
                                onClick={handleCopy}
                                className="p-2 text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
                                title="Copy content"
                            >
                                <Copy className="w-4 h-4" />
                            </button>

                            {/* Download button */}
                            <button
                                onClick={handleDownload}
                                className="p-2 text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
                                title="Download file"
                            >
                                <Download className="w-4 h-4" />
                            </button>
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
                    <div className="flex items-center justify-center h-32">
                        <div className="text-center">
                            <Loader2 className="w-6 h-6 animate-spin mx-auto mb-2 text-blue-500" />
                            <p className="text-sm text-gray-600 dark:text-gray-400">Loading file...</p>
                        </div>
                    </div>
                ) : error ? (
                    <div className="flex items-center justify-center h-32">
                        <div className="text-center">
                            <AlertTriangle className="w-6 h-6 mx-auto mb-2 text-red-500" />
                            <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
                        </div>
                    </div>
                ) : fileType === 'image' ? (
                    <div className="p-8 text-center">
                        <Image className="w-16 h-16 mx-auto mb-3 text-gray-400" />
                        <p className="text-gray-600 dark:text-gray-400">Image file</p>
                        <p className="text-sm text-gray-500 dark:text-gray-500">{fileName}</p>
                    </div>
                ) : fileType === 'binary' ? (
                    <div className="p-8 text-center">
                        <Archive className="w-16 h-16 mx-auto mb-3 text-gray-400" />
                        <p className="text-gray-600 dark:text-gray-400">Binary file</p>
                        <p className="text-sm text-gray-500 dark:text-gray-500">{fileName}</p>
                    </div>
                ) : !content ? (
                    <div className="p-8 text-center">
                        <FileText className="w-16 h-16 mx-auto mb-3 text-gray-400" />
                        <p className="text-gray-600 dark:text-gray-400">No content available</p>
                    </div>
                ) : (
                    <div className="bg-white dark:bg-gray-800">
                        {isDiffMode && content.includes('@@') ? renderDiffContent : renderFileContent}
                    </div>
                )}
            </div>
        </div>
    )
}

export default FileContentRenderer

import React, { useState, useEffect } from 'react'
import {
    Folder, 
    FolderOpen, 
    File, 
    Code, 
    Image, 
    FileText, 
    Settings,
    Database,
    Archive,
    ChevronRight,
    ChevronDown,
    GitCommit,
    Calendar,
    User,
    Eye,
    Download,
    Copy,
    Search,
    Filter,
    RefreshCw,
    Loader2,
    Maximize2,
    Minimize2
} from 'lucide-react'
import { diffSenseAPI } from '../services/api'
import FileContentRenderer from './FileContentRenderer'

// File type icons mapping
const FILE_ICONS = {
    // Code files
    'js': Code,
    'jsx': Code,
    'ts': Code,
    'tsx': Code,
    'py': Code,
    'java': Code,
    'cpp': Code,
    'c': Code,
    'cs': Code,
    'php': Code,
    'rb': Code,
    'go': Code,
    'rs': Code,
    'swift': Code,
    'kt': Code,
    'scala': Code,
    'sh': Code,
    'bat': Code,
    'ps1': Code,
    
    // Web files
    'html': Code,
    'htm': Code,
    'css': Code,
    'scss': Code,
    'sass': Code,
    'less': Code,
    'vue': Code,
    'svelte': Code,
    
    // Config/Data files
    'json': Settings,
    'yaml': Settings,
    'yml': Settings,
    'xml': Settings,
    'toml': Settings,
    'ini': Settings,
    'cfg': Settings,
    'conf': Settings,
    'env': Settings,
    'properties': Settings,
    
    // Database
    'sql': Database,
    'db': Database,
    'sqlite': Database,
    
    // Archives
    'zip': Archive,
    'tar': Archive,
    'gz': Archive,
    'rar': Archive,
    '7z': Archive,
    
    // Images
    'png': Image,
    'jpg': Image,
    'jpeg': Image,
    'gif': Image,
    'svg': Image,
    'ico': Image,
    'bmp': Image,
    'webp': Image,
    
    // Documents
    'md': FileText,
    'txt': FileText,
    'pdf': FileText,
    'doc': FileText,
    'docx': FileText,
    'rtf': FileText,
    'readme': FileText,
    'license': FileText,
    'changelog': FileText,
}

const getFileIcon = (filename) => {
    const extension = filename.toLowerCase().split('.').pop()
    const IconComponent = FILE_ICONS[extension] || File
    return IconComponent
}

const getFileTypeColor = (filename) => {
    const extension = filename.toLowerCase().split('.').pop()
    const colors = {
        'js': 'text-yellow-600',
        'jsx': 'text-blue-600',
        'ts': 'text-blue-700',
        'tsx': 'text-blue-700',
        'py': 'text-green-600',
        'java': 'text-orange-600',
        'cpp': 'text-purple-600',
        'c': 'text-purple-600',
        'cs': 'text-purple-700',
        'php': 'text-indigo-600',
        'rb': 'text-red-600',
        'go': 'text-cyan-600',
        'rs': 'text-orange-700',
        'html': 'text-orange-500',
        'css': 'text-blue-500',
        'json': 'text-gray-600',
        'md': 'text-gray-700',
        'txt': 'text-gray-500',
        'png': 'text-pink-500',
        'jpg': 'text-pink-500',
        'jpeg': 'text-pink-500',
        'gif': 'text-pink-500',
        'svg': 'text-green-500',
    }
    return colors[extension] || 'text-gray-600'
}

const formatFileSize = (bytes) => {
    if (!bytes) return ''
    const sizes = ['B', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(1024))
    return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${sizes[i]}`
}

// File tree item component
function FileTreeItem({ item, level = 0, onFileSelect, selectedFile, expandedFolders, onToggleFolder }) {
    const isFolder = item.type === 'tree'
    const isExpanded = expandedFolders.has(item.path)
    const isSelected = selectedFile?.path === item.path
    const IconComponent = isFolder ? (isExpanded ? FolderOpen : Folder) : getFileIcon(item.name)
    const fileColor = isFolder ? 'text-blue-600 dark:text-blue-400' : getFileTypeColor(item.name)
    
    const handleClick = () => {
        if (isFolder) {
            onToggleFolder(item.path)
        } else {
            onFileSelect(item)
        }
    }
    
    return (
        <div>
            <div
                className={`flex items-center gap-2 px-3 py-1.5 hover:bg-gray-100 dark:hover:bg-gray-700 cursor-pointer transition-colors ${
                    isSelected ? 'bg-blue-50 dark:bg-blue-900/20 border-r-2 border-blue-500' : ''
                }`}
                style={{ paddingLeft: `${12 + level * 20}px` }}
                onClick={handleClick}
            >
                {isFolder && (
                    <div className="flex-shrink-0">
                        {isExpanded ? (
                            <ChevronDown className="w-4 h-4 text-gray-400" />
                        ) : (
                            <ChevronRight className="w-4 h-4 text-gray-400" />
                        )}
                    </div>
                )}
                <IconComponent className={`w-4 h-4 flex-shrink-0 ${fileColor}`} />
                <span className={`text-sm truncate ${isSelected ? 'font-medium text-blue-700 dark:text-blue-300' : 'text-gray-900 dark:text-gray-100'}`}>
                    {item.name}
                </span>
                {!isFolder && item.size && (
                    <span className="text-xs text-gray-400 ml-auto">
                        {formatFileSize(item.size)}
                    </span>
                )}
            </div>
            
            {isFolder && isExpanded && item.children && (
                <div>
                    {item.children.map((child) => (
                        <FileTreeItem
                            key={child.path}
                            item={child}
                            level={level + 1}
                            onFileSelect={onFileSelect}
                            selectedFile={selectedFile}
                            expandedFolders={expandedFolders}
                            onToggleFolder={onToggleFolder}
                        />
                    ))}                </div>
            )}
        </div>
    )
}

function FileExplorer({ repoId, repoName, isMaximized = false, onToggleMaximize }) {
    const [fileTree, setFileTree] = useState(null)
    const [selectedFile, setSelectedFile] = useState(null)
    const [isLoading, setIsLoading] = useState(false)
    const [expandedFolders, setExpandedFolders] = useState(new Set())
    const [selectedCommit, setSelectedCommit] = useState(null)
    const [availableCommits, setAvailableCommits] = useState([])
    const [commitsLoading, setCommitsLoading] = useState(false)
    const [showCommitDropdown, setShowCommitDropdown] = useState(false)
    const [searchQuery, setSearchQuery] = useState('')
    const [error, setError] = useState(null)
    
    useEffect(() => {
        if (repoId) {
            loadFileTree()
            loadAvailableCommits()
        }
    }, [repoId])
    
    useEffect(() => {
        if (repoId && selectedCommit) {
            loadFileTree(selectedCommit.hash)
        }
    }, [selectedCommit])
    
    // Close dropdown when clicking outside
    useEffect(() => {
        const handleClickOutside = (event) => {
            if (!event.target.closest('.commit-dropdown')) {
                setShowCommitDropdown(false)
            }
        }
        
        document.addEventListener('mousedown', handleClickOutside)
        return () => document.removeEventListener('mousedown', handleClickOutside)
    }, [])
    
    const loadAvailableCommits = async () => {
        setCommitsLoading(true)
        try {
            const commits = await diffSenseAPI.getRepositoryCommits(repoId, 50)
            setAvailableCommits(Array.isArray(commits) ? commits : [])
        } catch (err) {
            console.error('Failed to load commits:', err)
            setAvailableCommits([])        } finally {
            setCommitsLoading(false)
        }
    }
    
    const loadFileTree = async (commitHash = null) => {
        setIsLoading(true)
        setError(null)
        try {            let response
            let files = []
            
            if (commitHash) {
                // Get files that existed at the specific commit
                response = await diffSenseAPI.getCommitFiles(repoId, commitHash, { includeDiffStats: false })
                files = response?.files || []
            } else {
                // Get current repository files (HEAD)
                response = await diffSenseAPI.getRepositoryFiles(repoId, { all_files: true })
                files = response?.files || []
            }
            
            const tree = buildFileTree(Array.isArray(files) ? files : [])
            setFileTree(tree)
            
            // Auto-expand root level folders
            const rootFolders = tree.filter(item => item.type === 'tree').map(item => item.path)
            setExpandedFolders(new Set(rootFolders))
        } catch (err) {
            console.error('Failed to load file tree:', err)
            setError(`Failed to load file tree${commitHash ? ` at commit ${commitHash.substring(0, 8)}` : ''}`)
        } finally {
            setIsLoading(false)
        }    }
    
    const buildFileTree = (files) => {
        const tree = []
        const pathMap = new Map()
        
        // Sort files by path for consistent ordering
        files.sort((a, b) => (a.path || a.file_path).localeCompare(b.path || b.file_path))
        
        files.forEach(file => {
            // Handle both 'path' and 'file_path' properties
            const filePath = file.path || file.file_path
            if (!filePath) return // Skip files without a path
            
            const parts = filePath.split('/')
            let currentPath = ''
            
            parts.forEach((part, index) => {
                const parentPath = currentPath
                currentPath = currentPath ? `${currentPath}/${part}` : part
                
                if (!pathMap.has(currentPath)) {
                    const isFile = index === parts.length - 1
                    const item = {
                        name: part,
                        path: currentPath,
                        type: isFile ? 'blob' : 'tree',
                        size: isFile ? file.size : undefined,
                        children: isFile ? undefined : []
                    }
                    
                    pathMap.set(currentPath, item)
                    
                    if (parentPath) {
                        const parent = pathMap.get(parentPath)
                        if (parent && parent.children) {
                            parent.children.push(item)
                        }
                    } else {
                        tree.push(item)
                    }
                }
            })
        })
        
        // Sort children (folders first, then files)
        const sortItems = (items) => {
            items.sort((a, b) => {
                if (a.type !== b.type) {
                    return a.type === 'tree' ? -1 : 1
                }
                return a.name.localeCompare(b.name)
            })
            
            items.forEach(item => {
                if (item.children) {
                    sortItems(item.children)
                }
            })
        }
        
        sortItems(tree)
        return tree
    }
      const handleFileSelect = (file) => {
        if (file.type !== 'blob') return
        setSelectedFile(file)
    }
    
    const handleToggleFolder = (path) => {
        const newExpanded = new Set(expandedFolders)
        if (newExpanded.has(path)) {
            newExpanded.delete(path)
        } else {
            newExpanded.add(path)
        }
        setExpandedFolders(newExpanded)
    }
    
    const formatCommitMessage = (message) => {
        return message.length > 50 ? message.substring(0, 50) + '...' : message
    }
    
    const formatCommitDate = (dateString) => {
        return new Date(dateString).toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        })
    }
    
    const filterTree = (items, query) => {
        if (!query) return items
        
        const filtered = []
        
        items.forEach(item => {
            if (item.name.toLowerCase().includes(query.toLowerCase())) {
                filtered.push(item)
            } else if (item.children) {
                const filteredChildren = filterTree(item.children, query)
                if (filteredChildren.length > 0) {
                    filtered.push({
                        ...item,
                        children: filteredChildren
                    })
                }
            }
        })
        
        return filtered
    }
      const filteredTree = searchQuery ? filterTree(fileTree || [], searchQuery) : fileTree
    
    return (
        <div className={`flex flex-col bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg transition-all duration-300 ease-in-out ${
            isMaximized ? 'fixed inset-2 z-[60] shadow-2xl max-h-[calc(100vh-1rem)]' : 'h-full overflow-hidden'
        }`}>
            {/* Header */}
            <div className="border-b border-gray-200 dark:border-gray-700 p-4 flex-shrink-0"><div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2">
                        <Folder className="w-5 h-5 text-blue-500" />
                        File Explorer
                        {selectedCommit && (
                            <span className="text-sm font-normal text-gray-500 dark:text-gray-400">
                                @ {selectedCommit.hash.substring(0, 8)}
                            </span>
                        )}
                    </h3>
                      <div className="flex items-center gap-2">
                        {onToggleMaximize && (
                            <button
                                onClick={onToggleMaximize}
                                className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
                                title={isMaximized ? "Exit fullscreen" : "Maximize"}
                            >
                                {isMaximized ? (
                                    <Minimize2 className="w-4 h-4" />
                                ) : (
                                    <Maximize2 className="w-4 h-4" />
                                )}
                            </button>
                        )}
                        <button
                            onClick={() => loadFileTree(selectedCommit?.hash)}
                            disabled={isLoading}
                            className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors disabled:opacity-50"
                            title="Refresh"
                        >
                            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
                        </button>
                    </div>
                </div>
                
                {/* Commit Selector */}
                <div className="mb-4">
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        View files at commit:
                    </label>
                    <div className="relative commit-dropdown">
                        <button
                            type="button"
                            onClick={() => setShowCommitDropdown(!showCommitDropdown)}
                            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 text-left flex items-center justify-between"
                            disabled={commitsLoading}
                        >
                            <span>
                                {selectedCommit ? (                                    <div className="flex items-center gap-2">
                                        <code className="text-xs bg-gray-100 dark:bg-gray-600 text-gray-900 dark:text-gray-100 px-2 py-1 rounded">
                                            {selectedCommit.hash.substring(0, 8)}
                                        </code>
                                        <span className="text-sm">
                                            {formatCommitMessage(selectedCommit.message)}
                                        </span>
                                    </div>
                                ) : (
                                    'Latest (HEAD)'
                                )}
                            </span>
                            <ChevronDown className="w-4 h-4" />
                        </button>
                        
                        {showCommitDropdown && (
                            <div className="absolute z-10 w-full mt-1 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md shadow-lg max-h-60 overflow-y-auto">
                                {/* Latest option */}
                                <button
                                    type="button"
                                    onClick={() => {
                                        setSelectedCommit(null)
                                        setShowCommitDropdown(false)
                                    }}
                                    className="w-full px-3 py-2 text-left hover:bg-gray-100 dark:hover:bg-gray-600 border-b border-gray-100 dark:border-gray-600"
                                >
                                    <div className="flex items-center gap-2">
                                        <div className="text-sm text-gray-900 dark:text-gray-100 font-medium">
                                            Latest (HEAD)
                                        </div>
                                    </div>
                                </button>
                                
                                {commitsLoading ? (
                                    <div className="p-3 text-center text-gray-500">
                                        <Loader2 className="w-4 h-4 animate-spin mx-auto mb-1" />
                                        Loading commits...
                                    </div>
                                ) : !Array.isArray(availableCommits) || availableCommits.length === 0 ? (
                                    <div className="p-3 text-center text-gray-500">No commits available</div>
                                ) : (
                                    availableCommits.map((commit) => (
                                        <button
                                            key={commit.hash}
                                            type="button"
                                            onClick={() => {
                                                setSelectedCommit(commit)
                                                setShowCommitDropdown(false)
                                            }}
                                            className="w-full px-3 py-2 text-left hover:bg-gray-100 dark:hover:bg-gray-600 border-b border-gray-100 dark:border-gray-600 last:border-b-0"
                                        >                                            <div className="flex items-center gap-2">
                                                <code className="text-xs bg-gray-100 dark:bg-gray-600 text-gray-900 dark:text-gray-100 px-2 py-1 rounded">
                                                    {commit.hash.substring(0, 8)}
                                                </code>
                                                <div className="flex-1 min-w-0">
                                                    <div className="text-sm text-gray-900 dark:text-gray-100 truncate">
                                                        {formatCommitMessage(commit.message)}
                                                    </div>
                                                    <div className="text-xs text-gray-500 dark:text-gray-400">
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
                </div>
                
                {/* Search */}
                <div className="relative">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                    <input
                        type="text"
                        placeholder="Search files..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                </div>            </div>
            
            <div className="flex flex-1 min-h-0">
                {/* File Tree */}
                <div className="w-1/3 border-r border-gray-200 dark:border-gray-700 overflow-y-auto">
                    {isLoading ? (
                        <div className="flex items-center justify-center h-32">
                            <div className="text-center">
                                <Loader2 className="w-6 h-6 animate-spin mx-auto mb-2 text-blue-500" />
                                <p className="text-sm text-gray-600 dark:text-gray-400">Loading files...</p>
                            </div>
                        </div>
                    ) : error ? (
                        <div className="p-4 text-center text-red-600 dark:text-red-400">
                            <FileText className="w-8 h-8 mx-auto mb-2" />
                            <p>{error}</p>
                        </div>
                    ) : filteredTree && filteredTree.length > 0 ? (
                        <div className="py-2">
                            {filteredTree.map((item) => (
                                <FileTreeItem
                                    key={item.path}
                                    item={item}
                                    onFileSelect={handleFileSelect}
                                    selectedFile={selectedFile}
                                    expandedFolders={expandedFolders}
                                    onToggleFolder={handleToggleFolder}
                                />
                            ))}
                        </div>
                    ) : (
                        <div className="flex items-center justify-center h-32 text-gray-500 dark:text-gray-400">
                            <div className="text-center">
                                <Folder className="w-8 h-8 mx-auto mb-2" />
                                <p className="text-sm">No files found</p>
                            </div>
                        </div>
                    )}
                </div>
                  {/* File Content */}
                <div className="flex-1">
                    {selectedFile ? (
                        <FileContentRenderer
                            repoId={repoId}
                            filePath={selectedFile.path}
                            commitHash={selectedCommit?.hash}
                            height="100%"
                            className="h-full border-0 rounded-none"
                        />
                    ) : (
                        <div className="flex items-center justify-center h-full text-gray-500 dark:text-gray-400">
                            <div className="text-center">
                                <File className="w-12 h-12 mx-auto mb-3 text-gray-300 dark:text-gray-600" />
                                <p>Select a file to view its content</p>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}

export default FileExplorer

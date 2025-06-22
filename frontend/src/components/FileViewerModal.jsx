import React, { useEffect, useState } from 'react'
import { X, ChevronLeft, ChevronRight, Eye, EyeOff } from 'lucide-react'
import PortableFileViewer from './PortableFileViewer'

/**
 * FileViewerModal - A modal overlay for displaying file content
 * 
 * This is a lightweight modal wrapper around PortableFileViewer
 * for use in contexts where you need to show file content in an overlay
 */
const FileViewerModal = ({ 
    isOpen, 
    onClose, 
    repoId, 
    filePath, 
    commitHash = null,
    title = null,
    showDiff = false,
    availableCommits = [],
    currentCommitIndex = -1,
    onNavigateCommit = null
}) => {
    const [isDiffMode, setIsDiffMode] = useState(showDiff)

    const displayTitle = title || (filePath ? filePath.split('/').pop() : 'File Viewer')
    const canNavigatePrevious = availableCommits.length > 1 && currentCommitIndex > 0
    const canNavigateNext = availableCommits.length > 1 && currentCommitIndex < availableCommits.length - 1

    const handleNavigateCommit = (direction) => {
        if (onNavigateCommit) {
            onNavigateCommit(direction)
        }
    }

    const handleToggleDiff = () => {
        setIsDiffMode(!isDiffMode)
    }

    // Handle keyboard shortcuts
    useEffect(() => {
        const handleKeyDown = (e) => {
            if (!isOpen) return
            
            // Arrow keys for commit navigation
            if (e.key === 'ArrowLeft' && canNavigatePrevious) {
                e.preventDefault()
                handleNavigateCommit('previous')
            } else if (e.key === 'ArrowRight' && canNavigateNext) {
                e.preventDefault()
                handleNavigateCommit('next')
            }
            // 'D' key for diff toggle
            else if ((e.key === 'd' || e.key === 'D') && commitHash) {
                e.preventDefault()
                handleToggleDiff()
            }
            // Escape to close
            else if (e.key === 'Escape') {
                e.preventDefault()
                onClose()
            }
        }

        if (isOpen) {
            document.addEventListener('keydown', handleKeyDown)
        }

        return () => {
            document.removeEventListener('keydown', handleKeyDown)
        }
    }, [isOpen, canNavigatePrevious, canNavigateNext, onClose, commitHash])

    // Early return after all hooks are called
    if (!isOpen) return null

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 dark:bg-black dark:bg-opacity-70 flex items-center justify-center z-50">
            <div className="bg-white dark:bg-gray-800 rounded-lg w-full h-full max-w-[95vw] max-h-[95vh] flex flex-col shadow-2xl">                {/* Modal Header */}
                <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50">
                    <div className="flex items-center gap-4">
                        <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                            {displayTitle}
                        </h2>
                        {availableCommits.length > 1 && (
                            <div className="flex items-center gap-2">
                                <button
                                    onClick={() => handleNavigateCommit('previous')}
                                    disabled={!canNavigatePrevious}
                                    className="p-1 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
                                    title="Previous commit (←)"
                                >
                                    <ChevronLeft className="w-4 h-4" />
                                </button>
                                <span className="text-sm text-gray-600 dark:text-gray-400">
                                    {currentCommitIndex + 1} of {availableCommits.length}
                                </span>
                                <button
                                    onClick={() => handleNavigateCommit('next')}
                                    disabled={!canNavigateNext}
                                    className="p-1 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
                                    title="Next commit (→)"
                                >
                                    <ChevronRight className="w-4 h-4" />
                                </button>
                            </div>
                        )}
                    </div>
                    <div className="flex items-center gap-2">
                        {/* Diff Toggle */}
                        {commitHash && (
                            <button
                                onClick={handleToggleDiff}
                                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg transition-colors text-sm font-medium ${
                                    isDiffMode
                                        ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 border border-green-300 dark:border-green-700'
                                        : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600 border border-gray-300 dark:border-gray-600'
                                }`}
                                title={isDiffMode ? 'Switch to content view (D)' : 'Switch to diff view (D)'}
                            >
                                {isDiffMode ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                                {isDiffMode ? 'View Content' : 'View Diff'}
                            </button>
                        )}
                        <button
                            onClick={onClose}
                            className="p-2 text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
                        >
                            <X className="w-5 h-5" />
                        </button>
                    </div>
                </div>                {/* Content */}
                <div className="flex-1 overflow-hidden">
                    <PortableFileViewer
                        repoId={repoId}
                        filePath={filePath}
                        commitHash={commitHash}
                        showDiff={isDiffMode}
                        height="100%"
                        className="h-full border-0 rounded-none"
                        showHeader={false} // Header is handled by the modal
                        showControls={true} // Enable controls
                        showDiffToggle={false} // Diff toggle is handled by the modal
                        showCopyButton={true} // Enable copy button
                        showDownloadButton={true} // Enable download button
                        showCommitNavigation={false} // Commit navigation is handled by the modal
                        onNavigateCommit={onNavigateCommit}
                    />
                </div>
            </div>
        </div>
    )
}

export default FileViewerModal

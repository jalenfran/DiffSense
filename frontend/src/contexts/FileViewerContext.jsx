import { createContext, useContext, useState } from 'react'
import FileViewerModal from '../components/FileViewerModal'
import { diffSenseAPI } from '../services/api'

const FileViewerContext = createContext()

export const FileViewerProvider = ({ children }) => {
    const [isOpen, setIsOpen] = useState(false)
    const [viewerConfig, setViewerConfig] = useState({
        repoId: null,
        filePath: null,
        commitHash: null,
        showDiff: false,
        title: null,
        availableCommits: [],
        currentCommitIndex: -1
    })

    const openFileViewer = (config) => {
        setViewerConfig({
            repoId: config.repoId,
            filePath: config.filePath,
            commitHash: config.commitHash || null,
            showDiff: config.showDiff || false,
            title: config.title || null,
            availableCommits: config.availableCommits || [],            currentCommitIndex: config.currentCommitIndex || -1
        })
        setIsOpen(true)
    }

    const closeFileViewer = () => {
        setIsOpen(false)
        // Clear config after a short delay to allow for smooth closing
        setTimeout(() => {
            setViewerConfig({
                repoId: null,
                filePath: null,
                commitHash: null,
                showDiff: false,
                title: null,
                availableCommits: [],
                currentCommitIndex: -1
            })
        }, 300)
    }    // Load commits for navigation - only commits where the file actually changed
    const loadCommitsForFile = async (repoId, filePath) => {
        try {
            // Use the file history endpoint to get only commits that touched this file
            const fileHistory = await diffSenseAPI.getFileHistory(repoId, filePath, { limit: 100 })
            
            // The API returns commits in the 'history' array, need to transform to match expected format
            const commits = (fileHistory.history || []).map(historyItem => ({
                hash: historyItem.commit_hash,
                message: historyItem.message,
                author: historyItem.author,
                timestamp: historyItem.date,
                short_hash: historyItem.short_hash
            }))
            
            return commits
        } catch (error) {
            console.error('Failed to load file history:', error)
            return []
        }
    }

    // Navigate to previous/next commit
    const navigateCommit = async (direction) => {
        const { availableCommits, currentCommitIndex, filePath, repoId, showDiff } = viewerConfig
        
        if (!availableCommits.length) return

        let newIndex
        if (direction === 'previous') {
            newIndex = Math.max(0, currentCommitIndex - 1)
        } else {
            newIndex = Math.min(availableCommits.length - 1, currentCommitIndex + 1)
        }

        if (newIndex !== currentCommitIndex) {
            const newCommit = availableCommits[newIndex]
            setViewerConfig(prev => ({
                ...prev,
                commitHash: newCommit.hash,
                currentCommitIndex: newIndex,
                title: filePath ? `${filePath} @ ${newCommit.hash.substring(0, 8)}` : null
            }))
        }
    }    // Convenience function for opening files with commit navigation
    const openFileWithCommitNavigation = async (repoId, filePath, commitHash = null, showDiff = false) => {
        const commits = await loadCommitsForFile(repoId, filePath)
        
        let currentIndex = 0  // Default to first commit (most recent)
        let actualCommitHash = commits[0]?.hash  // Default to first commit hash
        
        if (commitHash && commits.length > 0) {
            // If a specific commit is provided, find its index
            const foundIndex = commits.findIndex(c => c.hash === commitHash)
            if (foundIndex !== -1) {
                currentIndex = foundIndex
                actualCommitHash = commitHash
            }
            // If not found, we keep the defaults (index 0, most recent commit)
        }
        
        openFileViewer({
            repoId,
            filePath,
            commitHash: actualCommitHash,
            showDiff,
            title: filePath ? `${filePath}${actualCommitHash ? ` @ ${actualCommitHash.substring(0, 8)}` : ''}` : null,
            availableCommits: commits,
            currentCommitIndex: currentIndex
        })
    }

    // Convenience function for opening files
    const openFile = (repoId, filePath, commitHash = null, showDiff = false) => {
        openFileViewer({
            repoId,
            filePath,
            commitHash,
            showDiff,
            title: filePath ? `${filePath}${commitHash ? ` @ ${commitHash.substring(0, 8)}` : ''}` : null
        })
    }

    return (
        <FileViewerContext.Provider value={{
            isOpen,
            openFileViewer,
            closeFileViewer,
            openFile,
            openFileWithCommitNavigation,
            navigateCommit
        }}>
            {children}
            <FileViewerModal
                isOpen={isOpen}
                onClose={closeFileViewer}
                repoId={viewerConfig.repoId}
                filePath={viewerConfig.filePath}
                commitHash={viewerConfig.commitHash}
                showDiff={viewerConfig.showDiff}
                title={viewerConfig.title}
                availableCommits={viewerConfig.availableCommits}
                currentCommitIndex={viewerConfig.currentCommitIndex}
                onNavigateCommit={navigateCommit}
            />
        </FileViewerContext.Provider>
    )
}

export const useFileViewer = () => {
    const context = useContext(FileViewerContext)
    if (!context) {
        throw new Error('useFileViewer must be used within a FileViewerProvider')
    }
    return context
}
